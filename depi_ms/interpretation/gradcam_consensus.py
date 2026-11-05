"""Generate Grad-CAM maps and cohort-level consensus hotspots for DEPI-MS."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import cm as mpl_cm
from PIL import Image
from torch.utils.data import DataLoader

from depi_ms.training.train_depi_ms import (
    BinaryImageDataset,
    MSPeakNet_ResNet18,
    build_image_info_for_ids,
    set_seed,
)


class GradCAM:
    """Grad-CAM for a convolutional target layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handle = self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output
        output.register_hook(self._save_gradients)

    def _save_gradients(self, gradients):
        self.gradients = gradients

    def remove(self):
        self.handle.remove()

    def __call__(self, image: torch.Tensor, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        logits[:, class_idx].sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach(), logits.detach()


def overlay_cam(image: torch.Tensor, cam: torch.Tensor, alpha: float) -> Image.Image:
    """Overlay a normalized Grad-CAM map on an RGB tensor."""
    image_np = image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    cam_np = cam.detach().cpu().clamp(0, 1).numpy()
    heatmap = mpl_cm.get_cmap("jet")(cam_np)[..., :3]
    overlay = np.clip((1.0 - alpha) * image_np + alpha * heatmap, 0.0, 1.0)
    return Image.fromarray((overlay * 255).astype(np.uint8))


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    """Load a fold checkpoint into MSPeakNet."""
    model = MSPeakNet_ResNet18(num_classes=2, pretrained=False)
    state = torch.load(checkpoint_path, map_location=device)
    state_dict = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def run_gradcam(
    model,
    loader,
    device,
    out_dir: Path,
    alpha: float,
    target_mode: str,
):
    """Save per-sample Grad-CAM overlays, arrays, and metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    array_dir = out_dir / "_cam_npy"
    array_dir.mkdir(parents=True, exist_ok=True)

    gradcam = GradCAM(model, model.se_peak)
    rows = []

    try:
        with torch.enable_grad():
            for images, labels, ids, suffixes, names in loader:
                images = images.to(device)
                labels = labels.to(device)

                for i in range(images.size(0)):
                    image = images[i : i + 1]
                    target_class = 1 if target_mode == "positive" else None
                    cam, logits = gradcam(image, class_idx=target_class)
                    probs = F.softmax(logits, dim=1)[0]
                    pred = int(torch.argmax(probs).item())
                    true = int(labels[i].item())
                    explained_class = pred if target_class is None else target_class

                    base = Path(names[i]).stem
                    overlay_path = out_dir / f"{base}_t{true}_p{pred}_gradcam.png"
                    cam_path = array_dir / f"{base}.npy"

                    overlay_cam(images[i], cam, alpha).save(overlay_path)
                    np.save(cam_path, cam.cpu().numpy().astype(np.float32))

                    rows.append(
                        {
                            "image_name": names[i],
                            "base": base,
                            "id": str(ids[i]),
                            "suffix": int(suffixes[i].item() if hasattr(suffixes[i], "item") else suffixes[i]),
                            "true": true,
                            "pred": pred,
                            "explained_class": explained_class,
                            "p_pred": float(probs[pred].item()),
                            "p_positive": float(probs[1].item()),
                            "cam_npy": str(cam_path),
                            "overlay_png": str(overlay_path),
                        }
                    )
    finally:
        gradcam.remove()

    metadata = pd.DataFrame(rows)
    metadata.to_csv(out_dir / "cam_meta.csv", index=False)
    return metadata


def save_heatmap(array, path: Path, title: str):
    """Save a normalized heatmap."""
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    image = ax.imshow(array, cmap="hot", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_consensus(
    metadata: pd.DataFrame,
    out_dir: Path,
    quantile: float,
    freq_threshold: float,
    only_correct: bool,
    min_probability: float,
    balance_mode: str,
    per_class: int,
    seed: int,
):
    """Aggregate per-sample Grad-CAM maps into a cohort-level consensus."""
    data = metadata.copy()
    if only_correct:
        data = data[data["pred"] == data["true"]]
    if min_probability > 0:
        data = data[data["p_pred"] >= float(min_probability)]

    data = data[data["cam_npy"].map(lambda x: Path(str(x)).exists())].copy()
    if data.empty:
        raise RuntimeError("No Grad-CAM maps remain after filtering.")

    class_counts = data["true"].astype(int).value_counts().to_dict()

    if balance_mode == "none":
        data["weight"] = 1.0
    elif balance_mode == "class_weight":
        data["weight"] = data["true"].map(
            lambda label: 1.0 / float(class_counts[int(label)])
        )
    elif balance_mode == "stratified":
        rng = np.random.RandomState(seed)
        min_count = min(class_counts.values())
        sample_count = min_count if per_class < 0 else min(per_class, min_count)
        sampled = []
        for _, group in data.groupby("true"):
            chosen = rng.choice(group.index.to_numpy(), size=sample_count, replace=False)
            sampled.append(data.loc[chosen])
        data = pd.concat(sampled, ignore_index=True)
        data["weight"] = 1.0
        class_counts = data["true"].astype(int).value_counts().to_dict()
    else:
        raise ValueError(f"Unknown balance mode: {balance_mode}")

    hot_sum = None
    cam_sum = None
    total_weight = 0.0

    for row in data.itertuples(index=False):
        cam = np.load(row.cam_npy).astype(np.float32)
        weight = float(row.weight)
        if hot_sum is None:
            hot_sum = np.zeros_like(cam, dtype=np.float32)
            cam_sum = np.zeros_like(cam, dtype=np.float32)

        threshold = float(np.quantile(cam, quantile))
        hot_sum += weight * (cam >= threshold).astype(np.float32)
        cam_sum += weight * cam
        total_weight += weight

    frequency = hot_sum / max(total_weight, 1e-12)
    mean_cam = cam_sum / max(total_weight, 1e-12)
    mask = frequency >= float(freq_threshold)
    coords = np.argwhere(mask)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "freq.npy", frequency)
    np.save(out_dir / "mean_cam.npy", mean_cam)
    np.save(out_dir / "mask.npy", mask.astype(np.uint8))

    save_heatmap(
        frequency,
        out_dir / "freq_heatmap.png",
        f"Grad-CAM frequency (q={quantile:g}, threshold={freq_threshold:g})",
    )
    save_heatmap(mean_cam, out_dir / "mean_cam.png", "Mean Grad-CAM")
    save_heatmap(mask.astype(np.float32), out_dir / "mask_queue.png", "Consensus mask")

    rows = [
        {
            "x": int(x),
            "y": int(y),
            "freq": float(frequency[y, x]),
            "mean_cam": float(mean_cam[y, x]),
        }
        for y, x in coords
    ]
    queue = pd.DataFrame(rows, columns=["x", "y", "freq", "mean_cam"])
    queue.to_csv(out_dir / "heat_queue_xy.csv", index=False)

    summary = {
        "n_samples": int(len(data)),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "balance_mode": balance_mode,
        "quantile": float(quantile),
        "freq_threshold": float(freq_threshold),
        "only_correct": bool(only_correct),
        "min_probability": float(min_probability),
        "queue_size": int(len(queue)),
        "height": int(frequency.shape[0]),
        "width": int(frequency.shape[1]),
    }
    (out_dir / "consensus_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return queue


def parse_folds(value: str):
    """Parse a comma-separated fold list or 'all'."""
    if value.lower() == "all":
        return [1, 2, 3, 4, 5]
    folds = [int(item.strip()) for item in value.split(",") if item.strip()]
    invalid = [fold for fold in folds if fold not in {1, 2, 3, 4, 5}]
    if invalid:
        raise argparse.ArgumentTypeError(f"Invalid fold numbers: {invalid}")
    return folds


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM maps and consensus hotspots from trained DEPI-MS folds."
    )
    parser.add_argument("--img-dir", type=Path, required=True)
    parser.add_argument(
        "--split-xlsx",
        type=Path,
        default=Path("./metadata/five_fold_cv_patient.xlsx"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing fold1 ... fold5 training outputs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output root. Default: <checkpoint-dir>.",
    )
    parser.add_argument("--folds", type=parse_folds, default=[1, 2, 3, 4, 5])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradcam-alpha", type=float, default=0.45)
    parser.add_argument(
        "--target-mode",
        choices=["predicted", "positive"],
        default="predicted",
        help="Class used to compute each Grad-CAM map.",
    )
    parser.add_argument("--cam-quantile", type=float, default=0.90)
    parser.add_argument("--cam-freq-thr", type=float, default=0.25)
    parser.add_argument("--cam-only-correct", action="store_true")
    parser.add_argument("--cam-min-prob", type=float, default=0.0)
    parser.add_argument(
        "--cam-balance",
        choices=["none", "class_weight", "stratified"],
        default="class_weight",
    )
    parser.add_argument("--cam-per-class", type=int, default=-1)
    return parser


def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)

    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    output_root = args.out_dir or args.checkpoint_dir

    workbook = pd.ExcelFile(args.split_xlsx)
    for fold in args.folds:
        sheet = pd.read_excel(workbook, sheet_name=f"Fold{fold}")
        train_ids = sheet["train_id"].dropna().astype(str).tolist()

        image_info = build_image_info_for_ids(
            train_ids,
            str(args.img_dir),
            use_aug=False,
            mode="train",
        )
        if not image_info:
            raise RuntimeError(f"Fold {fold}: no training images found.")

        dataset = BinaryImageDataset(
            image_info,
            image_size=args.image_size,
        )
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
        )

        checkpoint = (
            args.checkpoint_dir
            / f"fold{fold}"
            / "model_last.pth"
        )
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        model = load_checkpoint(checkpoint, device)
        gradcam_dir = output_root / f"fold{fold}" / "gradcam" / "train"
        metadata = run_gradcam(
            model=model,
            loader=loader,
            device=device,
            out_dir=gradcam_dir,
            alpha=args.gradcam_alpha,
            target_mode=args.target_mode,
        )

        consensus_dir = gradcam_dir / "consensus"
        queue = build_consensus(
            metadata=metadata,
            out_dir=consensus_dir,
            quantile=args.cam_quantile,
            freq_threshold=args.cam_freq_thr,
            only_correct=args.cam_only_correct,
            min_probability=args.cam_min_prob,
            balance_mode=args.cam_balance,
            per_class=args.cam_per_class,
            seed=args.seed,
        )
        print(
            f"Fold {fold}: {len(metadata)} Grad-CAM maps, "
            f"{len(queue)} consensus pixels."
        )


if __name__ == "__main__":
    main()