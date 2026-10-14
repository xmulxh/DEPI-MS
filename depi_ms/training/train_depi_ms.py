"""Train DEPI-MS with participant-level five-fold cross-validation."""

import argparse
import os
import random
from glob import glob
from typing import Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    auc as sk_auc,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


CLASS_NAMES = ["C", "ABEM"]

CLASS_TO_IDX = {"C": 0, "A": 1, "B": 1, "E": 1, "M": 1}

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def random_rt_warp(
    images: torch.Tensor,

    mean_shift_sec: float = 60.0,
    std_shift_sec: float = 10.0,
    rt_min: float = 50.0,
    rt_max: float = 900.0,
    jitter_ratio: float = 0.3,
    kernel_size: int = 31,
) -> torch.Tensor:
    """
    Apply a smooth warp along the image height, which represents retention time.

    Each image receives a global RT shift sampled from a normal distribution,
    followed by a smaller smooth local perturbation. The resulting displacement
    field is applied with grid_sample.
      - images: [B, C, H, W]
    """

    if images.ndim != 4:
        raise ValueError(f"images should be 4D [B,C,H,W], got {images.shape}")

    B, C, H, W = images.shape
    if H <= 1:
        return images

    device = images.device


    rt_range = rt_max - rt_min

    pix_per_sec = H / float(rt_range)

    mean_shift_pix = mean_shift_sec * pix_per_sec
    std_shift_pix  = std_shift_sec  * pix_per_sec


    global_shift_pix = torch.normal(
        mean=torch.full((B, 1, 1), mean_shift_pix, device=device),
        std=std_shift_pix
    )


    noise = torch.randn(B, 1, H, device=device)
    disp_jitter = F.avg_pool1d(
        noise,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )


    max_abs = disp_jitter.abs().max(dim=2, keepdim=True)[0] + 1e-6
    disp_jitter = disp_jitter / max_abs
    disp_jitter = disp_jitter * (std_shift_pix * jitter_ratio)


    disp_pix = global_shift_pix + disp_jitter


    disp_norm = disp_pix * (2.0 / H)


    disp_norm = disp_norm.permute(0, 2, 1).expand(B, H, W)


    base_y = torch.linspace(-1.0, 1.0, H, device=device).view(1, H, 1).expand(B, H, W)
    base_x = torch.linspace(-1.0, 1.0, W, device=device).view(1, 1, W).expand(B, H, W)

    y_new = torch.clamp(base_y + disp_norm, -1.0, 1.0)
    grid = torch.stack([base_x, y_new], dim=-1)


    warped = F.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    return warped


def parse_sample_id(id_str: str) -> Tuple[str, str]:
    """
    Parse sample identifiers used by the internal and public datasets.

    Examples:
      - 'A-12' -> ('A', '12')
      - 'A_SAM03341' -> ('A', 'SAM03341')
      - 'A_BPL1415-0491' -> ('A', 'BPL1415-0491')
    """
    id_str = str(id_str)


    if "_" in id_str:
        parts = id_str.split("_", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid id format: {id_str}")
        cls = parts[0]
        sid = parts[1]
        return cls, sid


    parts = id_str.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid id format: {id_str}")
    cls = parts[0]
    sid = parts[1]
    return cls, sid


def build_image_info_for_ids(
    ids: List[str],
    img_dir: str,
    use_aug: bool,
    mode: str
) -> List[Dict]:
    """
    Build list of image entries for a list of base IDs.

    Each entry is dict(path, label, id_str, suffix).

    mode:
      - "train": use suffix 0 or 0-5 depending on use_aug
      - "val0":  only suffix 0
      - "val6":  only suffix 6

    Support the filename conventions used by the internal and public datasets.

    IDs such as A-8 map to A-8_0.png or A_8_0.png. Public-dataset IDs such as
    A_SAM03341 are resolved with a suffix-aware wildcard match.
    """
    image_info = []
    for id_str in ids:
        cls, sid = parse_sample_id(id_str)
        if cls not in CLASS_TO_IDX:
            raise ValueError(f"Unknown class {cls} in id {id_str}")

        label = CLASS_TO_IDX[cls]

        if mode == "train":
            if use_aug:
                suffixes = list(range(0, 6))
            else:
                suffixes = [0]
        elif mode == "val0":
            suffixes = [0]
        elif mode == "val6":
            suffixes = [6]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        for suf in suffixes:

            candidates = [
                f"{id_str}_{suf}.png",
                f"{cls}_{sid}_{suf}.png",
            ]

            found_path = None
            for fname in candidates:
                path = os.path.join(img_dir, fname)
                if os.path.exists(path):
                    found_path = path
                    break


            if found_path is None and isinstance(sid, str):
                pattern = os.path.join(img_dir, f"*{sid}*_{suf}.png")
                matches = glob(pattern)
                if len(matches) > 0:

                    found_path = sorted(matches)[0]

            if found_path is None:

                if suf in (0, 6):
                    print(
                        f"[Warning] Missing image for id {id_str}, suffix {suf}: "
                        f"tried {candidates[0]} / {candidates[1]} and pattern '*_{sid}_{suf}.png'"
                    )
                continue

            image_info.append(
                {"path": found_path, "label": label, "id": id_str, "suffix": suf}
            )

    return image_info

def get_transforms(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


class BinaryImageDataset(Dataset):
    def __init__(self, image_info: List[Dict], image_size: int = 224):
        self.image_info = image_info
        self.transform = get_transforms(image_size=image_size)

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        info = self.image_info[idx]
        path = info["path"]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        label = info["label"]
        return img, label, info["id"], info["suffix"], os.path.basename(path)


class SEBlock(nn.Module):
    """Squeeze-and-excitation block for channel-wise peak weighting."""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MSPeakNet_ResNet18(nn.Module):
    """
    Multi-scale peak-aware image classifier.

    When return_feat=True, the network also returns the global feature vector
    used by the RT-warp consistency loss.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(MSPeakNet_ResNet18, self).__init__()


        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)


        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4


        layers_to_freeze = [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False


        self.compress_l2 = nn.Conv2d(128, 64, kernel_size=1)
        self.compress_l3 = nn.Conv2d(256, 64, kernel_size=1)
        self.compress_l4 = nn.Conv2d(512, 64, kernel_size=1)

        fusion_channels = 64 * 4
        self.fusion_channels = fusion_channels


        self.se_peak = SEBlock(fusion_channels)


        feat_dim = fusion_channels * 2
        self.bn_final = nn.BatchNorm1d(feat_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(feat_dim, num_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the global image feature with shape [B, 2 * fusion_channels]."""

        l0 = self.layer0(x)
        l1 = self.layer1(l0)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)


        target_size = l1.shape[2:]


        l2_up = F.interpolate(self.compress_l2(l2), size=target_size, mode='nearest')
        l3_up = F.interpolate(self.compress_l3(l3), size=target_size, mode='nearest')
        l4_up = F.interpolate(self.compress_l4(l4), size=target_size, mode='nearest')


        x_fused = torch.cat([l1, l2_up, l3_up, l4_up], dim=1)


        x_att = self.se_peak(x_fused)


        x_max = F.adaptive_max_pool2d(x_att, 1).flatten(1)
        x_avg = F.adaptive_avg_pool2d(x_att, 1).flatten(1)
        x_cat = torch.cat([x_max, x_avg], dim=1)
        return x_cat

    def forward(self, x, return_feat: bool = False):
        """
        If return_feat is False, return logits only. Otherwise return
        (logits, feat), where feat is the global representation before the
        classifier and has shape [B, 2 * fusion_channels].
        """
        feat = self._forward_features(x)
        x_final = self.bn_final(feat)
        x_final = self.dropout(x_final)
        logits = self.fc(x_final)
        if return_feat:
            return logits, feat
        return logits


def youden_threshold(y_true: np.ndarray, pos_prob: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, pos_prob)
    j = tpr - fpr
    best_i = int(np.argmax(j))
    return float(thr[best_i])

def compute_metrics(
    y_true: List[int],
    y_prob: np.ndarray,
    num_classes: int,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true_np = np.array(y_true, dtype=int)
    y_prob_np = np.array(y_prob, dtype=float)

    pos_prob = y_prob_np if y_prob_np.ndim == 1 else y_prob_np[:, 1]


    y_pred_np = (pos_prob >= threshold).astype(int)

    acc = accuracy_score(y_true_np, y_pred_np)

    cm = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    auc = roc_auc_score(y_true_np, pos_prob) if len(np.unique(y_true_np)) >= 2 else float("nan")

    return {"acc": float(acc), "auc": float(auc), "sen": float(sensitivity), "spe": float(specificity)}


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    criterion: nn.Module = None,
    return_records: bool = True,
    plot_roc: bool = False,
    roc_save_path: str = None,
    roc_title: str = "ROC",
):
    model.eval()
    all_probs = []
    all_labels = []
    records = []
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in loader:
            images, labels, id_list, suffix_list, name_list = batch
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = F.softmax(logits, dim=1)

            if criterion is not None:
                loss = criterion(logits, labels)
                total_loss += loss.item() * images.size(0)
                n_samples += images.size(0)

            probs_np = probs.cpu().numpy()
            labels_np = labels.cpu().numpy().astype(int)
            preds_np = np.argmax(probs_np, axis=1)

            all_probs.append(probs_np)
            all_labels.append(labels_np)

            if return_records:
                for i in range(len(labels_np)):
                    rec = {
                        "image_name": name_list[i],
                        "id": id_list[i],
                        "suffix": int(suffix_list[i].item() if hasattr(suffix_list[i], 'item') else suffix_list[i]),
                        "true": int(labels_np[i]),
                        "pred": int(preds_np[i]),
                    }
                    for c_idx, cname in enumerate(CLASS_NAMES):
                        rec[f"prob_{cname}"] = float(probs_np[i, c_idx])
                    records.append(rec)

    if len(all_labels) == 0:
        return {"loss": float("nan"), "acc": float("nan"), "auc": float("nan"),
                "sen": float("nan"), "spe": float("nan")}, records

    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    pos_prob = all_probs[:, 1]
    thr = youden_threshold(all_labels, pos_prob)

    metrics = compute_metrics(all_labels, all_probs, num_classes, threshold=thr)
    if criterion is not None and n_samples > 0:
        metrics["loss"] = float(total_loss / n_samples)
    else:
        metrics["loss"] = float("nan")


    if plot_roc and roc_save_path is not None:
        try:
            print(f"Plotting ROC curve to {roc_save_path} ...")
            if len(np.unique(all_labels)) >= 2:
                pos_prob = all_probs[:, 1]
                fpr, tpr, _ = roc_curve(all_labels, pos_prob)
                roc_auc = sk_auc(fpr, tpr)

                plt.figure()
                plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
                plt.plot([0, 1], [0, 1], linestyle="--")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(roc_title)
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(roc_save_path, dpi=200)
                plt.close()
            else:
                print("[Warning] ROC skipped: only one class in y_true.")
        except Exception as e:
            print(f"[Warning] ROC plot failed: {e}")


    return metrics, records


def train_one_fold(
    fold_idx: int,
    train_ids: List[str],
    val_ids: List[str],
    args
):
    """Train one fold for binary classification C vs ABEM."""
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"\n========== Fold {fold_idx} on device {device} ==========")

    fold_dir = os.path.join(args.out_dir, f"fold{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    last_model_path = os.path.join(fold_dir, "model_last.pth")


    train_info = build_image_info_for_ids(
        train_ids, args.img_dir, use_aug=args.use_aug, mode="train"
    )
    val0_info = build_image_info_for_ids(
        val_ids, args.img_dir, use_aug=False, mode="val0"
    )
    val6_info = build_image_info_for_ids(
        val_ids, args.img_dir, use_aug=False, mode="val6"
    )

    if len(train_info) == 0:
        raise RuntimeError(f"Fold {fold_idx}: no training images found.")
    if len(val0_info) == 0:
        print(f"[Warning] Fold {fold_idx}: no validation-0 images found.")
    if len(val6_info) == 0:
        print(f"[Warning] Fold {fold_idx}: no validation-6 images found.")

    train_ds = BinaryImageDataset(train_info, image_size=args.image_size)
    val0_ds = BinaryImageDataset(val0_info, image_size=args.image_size)
    val6_ds = BinaryImageDataset(val6_info, image_size=args.image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    val0_loader = DataLoader(
        val0_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    val6_loader = DataLoader(
        val6_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )


    model = MSPeakNet_ResNet18(num_classes=2, pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()


    num_classes = 2


    print(f"Training fold {fold_idx} for {args.epochs} epochs. "
          f"Train images: {len(train_ds)}, Val0: {len(val0_ds)}, Val6: {len(val6_ds)}")

    history_rows = []
    pred_train_rows = []
    pred_val0_rows = []
    pred_val6_rows = []


    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)


    for epoch in range(1, args.epochs + 1):
        need_records = (epoch == args.epochs)
        model.train()


        model.layer0.eval()
        model.layer1.eval()
        model.layer2.eval()
        model.layer3.eval()
        model.layer4.eval()

        epoch_loss = 0.0
        all_train_probs = []
        all_train_labels = []
        train_records_epoch = []

        for batch in train_loader:
            images, labels, id_list, suffix_list, name_list = batch
            images = images.to(device)
            labels = labels.to(device)


            images_shift = random_rt_warp(
                                            images,
                                            mean_shift_sec=10, std_shift_sec=2,
                            rt_min=50, rt_max=900, jitter_ratio=0.3, kernel_size=31)

            optimizer.zero_grad()


            logits1, feat1 = model(images, return_feat=True)

            logits2, feat2 = model(images_shift, return_feat=True)


            loss_cls1 = criterion(logits1, labels)
            loss_cls2 = criterion(logits2, labels)

            loss_cons = F.mse_loss(feat1, feat2)

            loss = loss_cls1 + loss_cls2 + args.lambda_cons * loss_cons
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)

            probs = F.softmax(logits1, dim=1)
            probs_np = probs.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy().astype(int)
            preds_np = np.argmax(probs_np, axis=1)

            all_train_probs.append(probs_np)
            all_train_labels.append(labels_np)
            if need_records:
                for i in range(len(labels_np)):
                    rec = {
                        "epoch": epoch,
                        "image_name": name_list[i],
                        "id": id_list[i],
                        "suffix": int(suffix_list[i].item() if hasattr(suffix_list[i], 'item') else suffix_list[i]),
                        "true": int(labels_np[i]),
                        "pred": int(preds_np[i]),
                    }
                    for c_idx, cname in enumerate(CLASS_NAMES):
                        rec[f"prob_{cname}"] = float(probs_np[i, c_idx])
                    train_records_epoch.append(rec)

        scheduler.step()

        epoch_loss /= len(train_ds)
        if len(all_train_labels) > 0:
            all_train_labels = np.concatenate(all_train_labels, axis=0)
            all_train_probs = np.concatenate(all_train_probs, axis=0)
            train_metrics = compute_metrics(all_train_labels, all_train_probs, num_classes)
        else:
            train_metrics = {"acc": float("nan"), "fi": float("nan"), "auc": float("nan")}


        val0_metrics, val0_records = evaluate_model(model, val0_loader, device, num_classes, criterion, return_records=need_records)

        for r in val0_records:
            r["epoch"] = epoch


        val6_metrics, val6_records = evaluate_model(model, val6_loader, device, num_classes, criterion, return_records=need_records)
        for r in val6_records:
            r["epoch"] = epoch


        if need_records:
            pred_train_rows = train_records_epoch
            pred_val0_rows = val0_records
            pred_val6_rows = val6_records


        history_rows.append({
            "epoch": epoch, "split": "train",
            "loss": epoch_loss,
            "acc": train_metrics["acc"],
            "auc": train_metrics["auc"],
            "sen": train_metrics["sen"],
            "spe": train_metrics["spe"],
        })
        history_rows.append({
            "epoch": epoch, "split": "val0",
            "loss": val0_metrics["loss"],
            "acc": val0_metrics["acc"],
            "auc": val0_metrics["auc"],
            "sen": val0_metrics["sen"],
            "spe": val0_metrics["spe"],
        })
        history_rows.append({
            "epoch": epoch, "split": "val6",
            "loss": val6_metrics["loss"],
            "acc": val6_metrics["acc"],
            "auc": val6_metrics["auc"],
            "sen": val6_metrics["sen"],
            "spe": val6_metrics["spe"],
        })


    torch.save({"model_state_dict": model.state_dict()}, last_model_path)


    history_df = pd.DataFrame(history_rows)
    train_pred_df = pd.DataFrame(pred_train_rows)
    val0_pred_df = pd.DataFrame(pred_val0_rows)
    val6_pred_df = pd.DataFrame(pred_val6_rows)

    metrics_xlsx = os.path.join(fold_dir, f"fold{fold_idx}_metrics_and_predictions.xlsx")
    with pd.ExcelWriter(metrics_xlsx, engine="openpyxl") as writer:
        history_df.to_excel(writer, sheet_name="metrics", index=False)
        train_pred_df.to_excel(writer, sheet_name="train_pred", index=False)
        val0_pred_df.to_excel(writer, sheet_name="val0_pred", index=False)
        val6_pred_df.to_excel(writer, sheet_name="val6_pred", index=False)


    def _plot_metric(df, metric, save_name, ylabel):
        plt.figure()
        for split in ["train", "val0", "val6"]:
            sub = df[df["split"] == split]
            if len(sub) == 0:
                continue
            plt.plot(sub["epoch"], sub[metric], label=split)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"Fold {fold_idx} {metric}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(fold_dir, save_name))
        plt.close()

    _plot_metric(history_df, "acc", "acc_curve.png", "ACC")
    _plot_metric(history_df, "auc", "auc_curve.png", "AUC")
    _plot_metric(history_df, "loss", "loss_curve.png", "Loss")
    _plot_metric(history_df, "sen", "sen_curve.png", "Sensitivity")
    _plot_metric(history_df, "spe", "spe_curve.png", "Specificity")


    last_epoch = args.epochs
    last_val0 = history_df[(history_df["split"] == "val0") & (history_df["epoch"] == last_epoch)].iloc[0]
    last_val6 = history_df[(history_df["split"] == "val6") & (history_df["epoch"] == last_epoch)].iloc[0]

    summary_last = {
        "fold": fold_idx,
        "epoch": last_epoch,
        "val0_acc": float(last_val0["acc"]),
        "val0_auc": float(last_val0["auc"]),
        "val0_sen": float(last_val0["sen"]),
        "val0_spe": float(last_val0["spe"]),
        "val6_acc": float(last_val6["acc"]),
        "val6_auc": float(last_val6["auc"]),
        "val6_sen": float(last_val6["sen"]),
        "val6_spe": float(last_val6["spe"]),
    }


    return summary_last


def build_argparser():
    parser = argparse.ArgumentParser(description="Train DEPI-MS with five-fold participant-level cross-validation.")

    parser.add_argument("--img-dir", type=str, default="./img224", help="Directory with images A-1_0.png etc.")
    parser.add_argument("--split-xlsx", type=str, default="./metadata/five_fold_cv_patient.xlsx", help="Path to 5-fold split Excel.")
    parser.add_argument("--out-dir", type=str, default="./binary_cls_5fold_results", help="Output dir.")

    parser.add_argument("--use-aug", action="store_true", help="Use augmented images (suffix 1-5) in training.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU id to use.")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader num_workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument("--lambda-cons", type=float, default=0.5,
                        help="Weight applied to the RT-warp feature-consistency MSE loss.")
    return parser


def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)


    if not os.path.exists(args.split_xlsx):
        raise FileNotFoundError(f"Split file not found: {args.split_xlsx}")
    xls = pd.ExcelFile(args.split_xlsx)

    folds = {}
    for fold_idx in range(1, 6):
        sheet_name = f"Fold{fold_idx}"
        df = pd.read_excel(xls, sheet_name)
        train_ids = df["train_id"].dropna().astype(str).tolist()
        val_ids = df["val_id"].dropna().astype(str).tolist()
        folds[fold_idx] = (train_ids, val_ids)

    os.makedirs(args.out_dir, exist_ok=True)

    all_last = []

    for fold_idx in range(1, 6):
        train_ids, val_ids = folds[fold_idx]
        summary_last = train_one_fold(fold_idx, train_ids, val_ids, args)
        all_last.append(summary_last)


    def average_results(summaries, label):
        keys = ["val0_acc", "val0_auc", "val0_sen", "val0_spe",
                "val6_acc", "val6_auc", "val6_sen", "val6_spe"]
        avg = {"label": label}
        for k in keys:
            vals = np.array([s[k] for s in summaries], dtype=float)
            avg[k] = float(np.nanmean(vals))
            avg[f"{k}_std"] = float(np.nanstd(vals, ddof=1))
            avg[f"{k}_mean_std"] = f"{avg[k]:.3f} ± {avg[f'{k}_std']:.3f}"
        return avg

    def ms(avg, k):
        return avg[f"{k}_mean_std"]

    avg_last = average_results(all_last, "last_epoch")


    print("\n========== 5-fold summary ==========")
    print("Last epoch (mean ± std over folds):")
    print(f"  Val0: ACC={ms(avg_last, 'val0_acc')}, AUC={ms(avg_last, 'val0_auc')}, "
        f"SEN={ms(avg_last, 'val0_sen')}, SPE={ms(avg_last, 'val0_spe')}")
    print(f"  Val6: ACC={ms(avg_last, 'val6_acc')}, AUC={ms(avg_last, 'val6_auc')}, "
        f"SEN={ms(avg_last, 'val6_sen')}, SPE={ms(avg_last, 'val6_spe')}")


    summary_path = os.path.join(args.out_dir, "summary_overall.xlsx")
    df_last = pd.DataFrame(all_last)
    df_avg = pd.DataFrame([avg_last])

    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        df_last.to_excel(writer, sheet_name="fold_last", index=False)
        df_avg.to_excel(writer, sheet_name="average", index=False)

    print("All done.")


if __name__ == "__main__":
    main()