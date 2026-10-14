"""Extract DEPI-MS features and generate a t-SNE visualization."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from depi_ms.training.train_depi_ms import (
    BinaryImageDataset,
    CLASS_NAMES,
    MSPeakNet_ResNet18,
    build_image_info_for_ids,
    set_seed,
)


def load_validation_ids(split_xlsx: Path, fold_idx: int):
    df = pd.read_excel(split_xlsx, sheet_name=f"Fold{fold_idx}")
    return df["val_id"].dropna().astype(str).tolist()


def load_model(checkpoint_path: Path, device: torch.device):
    model = MSPeakNet_ResNet18(num_classes=2, pretrained=False)
    state = torch.load(checkpoint_path, map_location=device)
    state_dict = (
        state["model_state_dict"]
        if isinstance(state, dict) and "model_state_dict" in state
        else state
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_features(model, loader, device):
    features = []
    labels = []
    suffixes = []
    ids = []
    names = []

    model.eval()
    for images, y, id_list, suffix_list, name_list in loader:
        images = images.to(device)
        _, feat = model(images, return_feat=True)

        features.append(feat.cpu().numpy().astype(np.float32))
        labels.append(y.numpy().astype(np.int64))
        suffixes.append(
            np.asarray(
                [
                    int(item.item()) if hasattr(item, "item") else int(item)
                    for item in suffix_list
                ],
                dtype=np.int64,
            )
        )
        ids.extend(str(item) for item in id_list)
        names.extend(str(item) for item in name_list)

    return (
        np.concatenate(features, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(suffixes, axis=0),
        np.asarray(ids, dtype=object),
        np.asarray(names, dtype=object),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract DEPI-MS embeddings and generate a t-SNE plot."
    )
    parser.add_argument("--img-dir", required=True, type=Path)
    parser.add_argument("--split-xlsx", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--fold", type=int, default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--view", choices=["val0", "val6"], default="val0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--pca-dim", type=int, default=50)
    args = parser.parse_args()

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    val_ids = load_validation_ids(args.split_xlsx, args.fold)
    image_info = build_image_info_for_ids(
        val_ids,
        str(args.img_dir),
        use_aug=False,
        mode=args.view,
    )
    if len(image_info) < 5:
        raise RuntimeError(f"Too few images for t-SNE: N={len(image_info)}")

    dataset = BinaryImageDataset(image_info, image_size=args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    checkpoint = args.checkpoint_dir / f"fold{args.fold}" / "model_last.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = load_model(checkpoint, device)
    features, labels, suffixes, ids, names = extract_features(
        model,
        loader,
        device,
    )

    np.savez_compressed(
        args.out_dir / "tsne_features_raw.npz",
        features=features,
        labels=labels,
        suffixes=suffixes,
        ids=ids,
        image_names=names,
    )

    x = features
    n_samples, n_features = x.shape
    pca_dim = min(args.pca_dim, n_features, max(2, n_samples - 1))
    if n_features > pca_dim:
        x = PCA(
            n_components=pca_dim,
            random_state=args.seed,
            svd_solver="full",
        ).fit_transform(x)

    feature_table = pd.DataFrame(
        x,
        columns=[f"f{i:04d}" for i in range(x.shape[1])],
    )
    feature_table.insert(0, "label", labels)
    feature_table.insert(0, "suffix", suffixes)
    feature_table.insert(0, "image_name", names)
    feature_table.insert(0, "id", ids)
    feature_table.to_csv(args.out_dir / "tsne_features.csv", index=False)

    perplexity = min(
        float(args.perplexity),
        float(max(5, (n_samples - 1) // 3)),
    )
    perplexity = min(perplexity, float(n_samples - 1))

    coordinates = TSNE(
        n_components=2,
        random_state=args.seed,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
    ).fit_transform(x)

    coordinate_table = pd.DataFrame(
        {
            "id": ids,
            "image_name": names,
            "suffix": suffixes,
            "label": labels,
            "tsne1": coordinates[:, 0],
            "tsne2": coordinates[:, 1],
        }
    )
    coordinate_table.to_csv(args.out_dir / "tsne_coords.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    marker_map = {0: "o", 6: "x"}
    for label in [0, 1]:
        for suffix in sorted(np.unique(suffixes).tolist()):
            mask = (labels == label) & (suffixes == suffix)
            if not mask.any():
                continue
            ax.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                s=18,
                alpha=0.85,
                marker=marker_map.get(int(suffix), "o"),
                label=f"{CLASS_NAMES[label]} | suf={int(suffix)}",
            )

    ax.set_title(f"Fold{args.fold} t-SNE {args.view}")
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    fig.savefig(args.out_dir / "tsne.png", dpi=300)
    fig.savefig(args.out_dir / "tsne.pdf", dpi=300)
    plt.close(fig)

    print(f"t-SNE outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()