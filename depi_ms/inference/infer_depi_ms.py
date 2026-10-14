"""Evaluate trained DEPI-MS fold models."""

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from depi_ms.training.train_depi_ms import (
    BinaryImageDataset,
    MSPeakNet_ResNet18,
    build_image_info_for_ids,
    evaluate_model,
    set_seed,
)


def parse_folds(value: str):
    if value.lower() == "all":
        return [1, 2, 3, 4, 5]

    folds = [int(item.strip()) for item in value.split(",") if item.strip()]
    invalid = [fold for fold in folds if fold not in {1, 2, 3, 4, 5}]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"folds must be 1-5 or 'all'; got {invalid}"
        )
    return folds


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


def evaluate_view(
    model,
    ids,
    img_dir,
    view,
    image_size,
    batch_size,
    num_workers,
    device,
):
    info = build_image_info_for_ids(
        ids,
        str(img_dir),
        use_aug=False,
        mode=view,
    )
    if not info:
        raise RuntimeError(f"No images found for {view} in {img_dir}")

    dataset = BinaryImageDataset(info, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    criterion = nn.CrossEntropyLoss()
    metrics, records = evaluate_model(
        model,
        loader,
        device,
        num_classes=2,
        criterion=criterion,
        return_records=True,
        plot_roc=False,
    )
    return metrics, pd.DataFrame(records)


def summarize_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for view, group in summary.groupby("view", sort=False):
        row = {"view": view}
        for metric in ["ACC", "AUC", "SEN", "SPE"]:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained DEPI-MS five-fold checkpoints."
    )
    parser.add_argument("--img-dir", required=True, type=Path)
    parser.add_argument(
        "--split-xlsx",
        required=True,
        type=Path,
        help="Workbook containing Fold1-Fold5 and val_id.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        type=Path,
        help="Training output directory containing fold1 ... fold5.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <checkpoint-dir>/inference.",
    )
    parser.add_argument(
        "--view",
        choices=["val0", "val6", "both"],
        default="val0",
        help="Evaluate unshifted images, Shift-60 images, or both.",
    )
    parser.add_argument(
        "--folds",
        type=parse_folds,
        default=[1, 2, 3, 4, 5],
        help="Comma-separated folds, e.g. 1,2,3, or 'all'.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    out_dir = args.out_dir or (args.checkpoint_dir / "inference")
    out_dir.mkdir(parents=True, exist_ok=True)

    views = ["val0", "val6"] if args.view == "both" else [args.view]
    summary_rows = []

    for fold_idx in args.folds:
        checkpoint = args.checkpoint_dir / f"fold{fold_idx}" / "model_last.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        val_ids = load_validation_ids(args.split_xlsx, fold_idx)
        model = load_model(checkpoint, device)

        for view in views:
            metrics, records = evaluate_view(
                model=model,
                ids=val_ids,
                img_dir=args.img_dir,
                view=view,
                image_size=args.image_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
            )
            records.insert(0, "fold", fold_idx)
            records.insert(1, "view", view)
            records.to_csv(
                out_dir / f"fold{fold_idx}_{view}_predictions.csv",
                index=False,
            )
            summary_rows.append(
                {
                    "fold": fold_idx,
                    "view": view,
                    "n": len(records),
                    "ACC": metrics["acc"],
                    "AUC": metrics["auc"],
                    "SEN": metrics["sen"],
                    "SPE": metrics["spe"],
                    "loss": metrics["loss"],
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "fold_metrics.csv", index=False)

    aggregate = summarize_metrics(summary)
    aggregate.to_csv(out_dir / "mean_std_metrics.csv", index=False)

    print(f"Inference results saved to: {out_dir}")


if __name__ == "__main__":
    main()