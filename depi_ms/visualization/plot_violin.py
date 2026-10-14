"""Summarize consensus-pixel intensities and draw violin/AUC panels."""

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_CLASS_ORDER = ["C", "E", "A", "P"]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize every pixel in heat_queue_xy.csv and draw violin/AUC "
            "panels for a configurable number of pixels."
        )
    )
    parser.add_argument("--queue-csv", type=Path, required=True)
    parser.add_argument(
        "--img-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Pseudo-image directories. Only images ending with _0 are used.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-pix-per-row",
        type=int,
        default=4,
        help="Number of pixels per row in the violin/AUC figure.",
    )
    parser.add_argument(
        "--max-plot-pixels",
        type=int,
        default=20,
        help=(
            "Maximum number of queue pixels drawn in the figure. "
            "All pixels are still included in the output tables. Use 0 for no limit."
        ),
    )
    parser.add_argument("--auc-vmin", type=float, default=0.5)
    parser.add_argument("--auc-vmax", type=float, default=1.0)
    return parser.parse_args()


def load_queue_coords(queue_csv: Path) -> np.ndarray:
    if not queue_csv.exists():
        raise FileNotFoundError(f"Queue CSV not found: {queue_csv}")

    df = pd.read_csv(queue_csv)
    missing = [col for col in ["x", "y"] if col not in df.columns]
    if missing:
        raise ValueError(f"Queue CSV is missing columns: {missing}")

    xs = df["x"].astype(int).to_numpy()
    ys = df["y"].astype(int).to_numpy()
    return np.stack([ys, xs], axis=1)


def map_raw_prefix_to_group(raw_prefix: str) -> str:
    raw = str(raw_prefix).strip()
    if raw == "C":
        return "C"
    if raw == "E":
        return "E"
    if raw in {"A", "B", "M"}:
        return "A"
    if raw in {"postRT", "postCT"}:
        return "P"
    return ""


def infer_group_from_stem(stem: str) -> Tuple[str, str]:
    s = str(stem).strip()
    if not s.endswith("_0"):
        return "", ""

    core = s[:-2]
    match = re.match(r"^(postRT|postCT|[ABCEM])(?:[_-]|$)", core)
    if match is None:
        return "", ""

    raw_prefix = match.group(1)
    return map_raw_prefix_to_group(raw_prefix), raw_prefix


def scan_image_entries(img_dirs: List[Path]) -> Dict[str, List[Dict]]:
    entries = {key: [] for key in DEFAULT_CLASS_ORDER}
    seen = set()

    for root in img_dirs:
        if not root.exists():
            print(f"[Warning] Image directory does not exist: {root}")
            continue

        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in VALID_EXTS:
                continue

            group, raw_prefix = infer_group_from_stem(path.stem)
            if not group:
                continue

            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)

            entries[group].append(
                {
                    "class": group,
                    "raw_prefix": raw_prefix,
                    "path": resolved,
                    "image_name": path.name,
                }
            )

    for group in entries:
        entries[group] = sorted(entries[group], key=lambda item: item["image_name"])
    return entries


def load_gray_image(path: str, image_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize(
        (image_size, image_size),
        resample=Image.BILINEAR,
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array.mean(axis=2)


def validate_coords(coords_yx: np.ndarray, image_size: int) -> np.ndarray:
    keep = (
        (coords_yx[:, 0] >= 0)
        & (coords_yx[:, 0] < image_size)
        & (coords_yx[:, 1] >= 0)
        & (coords_yx[:, 1] < image_size)
    )
    valid = coords_yx[keep]
    if len(valid) == 0:
        raise RuntimeError("No queue pixels are within the image bounds.")
    return valid


def compute_pixel_class_means(
    coords_yx: np.ndarray,
    entries_by_cls: Dict[str, List[Dict]],
    image_size: int,
    class_order: List[str],
):
    coords_yx = validate_coords(coords_yx, image_size)
    ys = coords_yx[:, 0]
    xs = coords_yx[:, 1]

    sums = np.zeros((len(class_order), len(xs)), dtype=np.float64)
    counts = np.zeros(len(class_order), dtype=np.int64)

    for class_index, cls in enumerate(class_order):
        for entry in entries_by_cls.get(cls, []):
            gray = load_gray_image(entry["path"], image_size)
            sums[class_index] += gray[ys, xs]
            counts[class_index] += 1

    means = np.full_like(sums, np.nan, dtype=np.float64)
    for class_index, count in enumerate(counts):
        if count > 0:
            means[class_index] = sums[class_index] / float(count)

    table = pd.DataFrame({"x": xs.astype(int), "y": ys.astype(int)})
    for class_index, cls in enumerate(class_order):
        table[cls] = means[class_index].astype(np.float32)
        table[f"n_{cls}"] = int(counts[class_index])

    return table, means.T.astype(np.float32)


def collect_pixel_values(
    entries_by_cls: Dict[str, List[Dict]],
    pixels_xy: List[Tuple[int, int]],
    class_order: List[str],
    image_size: int,
) -> pd.DataFrame:
    rows = []

    for cls in class_order:
        for entry in entries_by_cls.get(cls, []):
            gray = load_gray_image(entry["path"], image_size)

            for pixel_idx, (x, y) in enumerate(pixels_xy):
                if not (0 <= x < image_size and 0 <= y < image_size):
                    continue
                rows.append(
                    {
                        "pixel_idx": int(pixel_idx),
                        "x": int(x),
                        "y": int(y),
                        "cls": cls,
                        "value": float(gray[y, x]),
                        "image_name": entry["image_name"],
                        "path": entry["path"],
                        "raw_prefix": entry["raw_prefix"],
                    }
                )

    return pd.DataFrame(rows)


def summarize_pixel_medians(
    df_long: pd.DataFrame,
    class_order: List[str],
) -> pd.DataFrame:
    medians = (
        df_long.groupby(["pixel_idx", "x", "y", "cls"], as_index=False)["value"]
        .median()
        .rename(columns={"value": "median"})
    )
    counts = (
        df_long.groupby(["pixel_idx", "x", "y", "cls"], as_index=False)["value"]
        .size()
        .rename(columns={"size": "n"})
    )

    medians = medians.pivot_table(
        index=["pixel_idx", "x", "y"],
        columns="cls",
        values="median",
        aggfunc="first",
    )
    counts = counts.pivot_table(
        index=["pixel_idx", "x", "y"],
        columns="cls",
        values="n",
        aggfunc="first",
    )

    for cls in class_order:
        if cls not in medians.columns:
            medians[cls] = np.nan
        if cls not in counts.columns:
            counts[cls] = 0

    medians = medians[class_order]
    counts = counts[class_order].fillna(0).astype(int)
    counts.columns = [f"n_{cls}" for cls in counts.columns]
    return pd.concat([medians, counts], axis=1).reset_index()


def plot_pixel_class_heatmap(
    df_means: pd.DataFrame,
    matrix: np.ndarray,
    class_order: List[str],
    save_path: Path,
    max_yticks: int = 25,
):
    n_pixels, n_classes = matrix.shape
    fig, ax = plt.subplots(figsize=(1.8 + n_classes, 6.0))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(n_classes))
    ax.set_xticklabels(class_order)
    ax.set_xlabel("Class")

    if n_pixels <= max_yticks:
        ticks = np.arange(n_pixels)
    else:
        ticks = np.linspace(0, n_pixels - 1, max_yticks).astype(int)

    labels = [
        f"({int(df_means.loc[i, 'x'])},{int(df_means.loc[i, 'y'])})"
        for i in ticks
    ]
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylabel("Pixel (x,y)")
    ax.set_title("Consensus pixels: mean intensity per class")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def pairwise_auc_matrix(
    df_pixel: pd.DataFrame,
    class_order: List[str],
) -> np.ndarray:
    matrix = np.full(
        (len(class_order), len(class_order)),
        np.nan,
        dtype=np.float32,
    )
    values = {
        cls: df_pixel.loc[df_pixel["cls"] == cls, "value"].to_numpy(
            dtype=np.float32
        )
        for cls in class_order
    }

    for i in range(len(class_order)):
        for j in range(i + 1, len(class_order)):
            a = values[class_order[i]]
            b = values[class_order[j]]
            if len(a) == 0 or len(b) == 0:
                continue

            y = np.concatenate(
                [np.zeros(len(a), dtype=int), np.ones(len(b), dtype=int)]
            )
            score = np.concatenate([a, b])
            auc_value = float(roc_auc_score(y, score))
            auc_value = max(auc_value, 1.0 - auc_value)
            matrix[i, j] = auc_value
            matrix[j, i] = auc_value

    return matrix


def build_pairwise_auc_table(
    df_long: pd.DataFrame,
    pixels_xy: List[Tuple[int, int]],
    class_order: List[str],
) -> pd.DataFrame:
    rows = []

    for pixel_idx, subset in df_long.groupby("pixel_idx", sort=False):
        pixel_idx = int(pixel_idx)
        x, y = pixels_xy[pixel_idx]
        matrix = pairwise_auc_matrix(subset, class_order)

        for i in range(len(class_order)):
            for j in range(i + 1, len(class_order)):
                value = matrix[i, j]
                rows.append(
                    {
                        "pixel_idx": pixel_idx,
                        "x": x,
                        "y": y,
                        "pair": f"{class_order[i]}_vs_{class_order[j]}",
                        "auc": float(value) if np.isfinite(value) else np.nan,
                    }
                )

    return pd.DataFrame(rows)


def plot_violin_and_auc(
    df_long: pd.DataFrame,
    pixels_xy: List[Tuple[int, int]],
    class_order: List[str],
    save_png: Path,
    save_pdf: Path,
    n_pix_per_row: int,
    auc_vmin: float,
    auc_vmax: float,
):
    n_pixels = len(pixels_xy)
    if n_pixels == 0:
        return

    n_pix_per_row = max(1, int(n_pix_per_row))
    nrows = int(math.ceil(n_pixels / n_pix_per_row))
    ncols = 2 * n_pix_per_row

    fig = plt.figure(figsize=(3.1 * ncols, 2.8 * nrows))
    width_ratios = []
    for _ in range(n_pix_per_row):
        width_ratios.extend([1.25, 0.95])

    grid = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols,
        width_ratios=width_ratios,
        wspace=0.40,
        hspace=0.55,
    )

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    for pixel_idx, (x, y) in enumerate(pixels_xy):
        row = pixel_idx // n_pix_per_row
        pair_index = pixel_idx % n_pix_per_row
        violin_col = 2 * pair_index
        auc_col = violin_col + 1

        subset = df_long[df_long["pixel_idx"] == pixel_idx]
        data = [
            subset.loc[subset["cls"] == cls, "value"].to_numpy(dtype=np.float32)
            for cls in class_order
        ]

        ax_violin = fig.add_subplot(grid[row, violin_col])
        positions = np.arange(1, len(class_order) + 1)
        violin = ax_violin.violinplot(
            [
                values
                if len(values) >= 2
                else np.array([np.nan], dtype=np.float32)
                for values in data
            ],
            positions=positions,
            widths=0.85,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for body in violin["bodies"]:
            body.set_facecolor("none")
            body.set_edgecolor("black")
            body.set_linewidth(1.0)
            body.set_alpha(1.0)
        if "cmedians" in violin:
            violin["cmedians"].set_color("black")
            violin["cmedians"].set_linewidth(1.2)

        for index, values in enumerate(data):
            if len(values) == 1:
                ax_violin.scatter([positions[index]], [values[0]], s=18)

        ax_violin.set_xticks(positions)
        ax_violin.set_xticklabels(class_order, fontsize=9)
        ax_violin.set_xlim(0.5, len(class_order) + 0.5)
        ax_violin.set_title(f"Pixel ({x},{y})", fontsize=11)
        ax_violin.spines["top"].set_visible(False)
        ax_violin.spines["right"].set_visible(False)
        ax_violin.grid(True, axis="y", linestyle="--", alpha=0.25)

        ax_auc = fig.add_subplot(grid[row, auc_col])
        auc_matrix = pairwise_auc_matrix(subset, class_order)
        image = ax_auc.imshow(
            auc_matrix,
            vmin=auc_vmin,
            vmax=auc_vmax,
            aspect="equal",
            cmap="Blues",
        )
        ax_auc.set_xticks(np.arange(len(class_order)))
        ax_auc.set_yticks(np.arange(len(class_order)))
        ax_auc.set_xticklabels(
            class_order,
            fontsize=8,
            rotation=45,
            ha="right",
        )
        ax_auc.set_yticklabels(class_order, fontsize=8)
        ax_auc.set_title("AUC", fontsize=10)

        for i in range(len(class_order)):
            for j in range(len(class_order)):
                value = auc_matrix[i, j]
                if np.isfinite(value) and i != j:
                    ax_auc.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                    )

        ax_auc.spines["top"].set_visible(False)
        ax_auc.spines["right"].set_visible(False)

        if pair_index == n_pix_per_row - 1 or pixel_idx == n_pixels - 1:
            colorbar = fig.colorbar(
                image,
                ax=ax_auc,
                fraction=0.046,
                pad=0.02,
            )
            colorbar.ax.tick_params(labelsize=8)

    for pixel_idx in range(n_pixels, nrows * n_pix_per_row):
        row = pixel_idx // n_pix_per_row
        pair_index = pixel_idx % n_pix_per_row
        for column in [2 * pair_index, 2 * pair_index + 1]:
            ax = fig.add_subplot(grid[row, column])
            ax.axis("off")

    fig.savefig(save_png, dpi=600, bbox_inches="tight")
    fig.savefig(save_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.out_dir)

    coords_yx = load_queue_coords(args.queue_csv)
    coords_yx = validate_coords(coords_yx, args.image_size)
    pixels_xy = [(int(x), int(y)) for y, x in coords_yx]

    entries_by_cls = scan_image_entries(args.img_dirs)
    class_order = list(DEFAULT_CLASS_ORDER)
    class_counts = {
        cls: len(entries_by_cls.get(cls, []))
        for cls in class_order
    }
    print(f"[Scan] Class counts: {class_counts}")
    print(f"[Queue] Pixels: {len(pixels_xy)}")

    df_means, mean_matrix = compute_pixel_class_means(
        coords_yx=coords_yx,
        entries_by_cls=entries_by_cls,
        image_size=args.image_size,
        class_order=class_order,
    )
    df_means.to_csv(
        args.out_dir / "pixel_class_means.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_pixel_class_heatmap(
        df_means,
        mean_matrix,
        class_order,
        args.out_dir / "pixel_class_means_heatmap.png",
    )

    df_long = collect_pixel_values(
        entries_by_cls=entries_by_cls,
        pixels_xy=pixels_xy,
        class_order=class_order,
        image_size=args.image_size,
    )
    if df_long.empty:
        raise RuntimeError(
            "No pixel values were collected. Check image directories and naming rules."
        )

    df_long.to_csv(
        args.out_dir / "violin_pixel_values_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    df_medians = summarize_pixel_medians(df_long, class_order)
    df_medians.to_csv(
        args.out_dir / "violin_pixel_medians.csv",
        index=False,
        encoding="utf-8-sig",
    )

    auc_table = build_pairwise_auc_table(
        df_long,
        pixels_xy,
        class_order,
    )
    auc_table.to_csv(
        args.out_dir / "pairwise_auc_per_pixel_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if args.max_plot_pixels < 0:
        raise ValueError("--max-plot-pixels must be >= 0")

    if args.max_plot_pixels == 0:
        n_plot = len(pixels_xy)
    else:
        n_plot = min(args.max_plot_pixels, len(pixels_xy))

    plot_pixels = pixels_xy[:n_plot]
    plot_values = df_long[df_long["pixel_idx"] < n_plot].copy()

    metadata = {
        "class_order": class_order,
        "pixels_xy": pixels_xy,
        "n_pixels": len(pixels_xy),
        "n_plotted_pixels": n_plot,
        "max_plot_pixels": args.max_plot_pixels,
        "image_size": args.image_size,
        "suffix": 0,
        "value_def": "mean(R,G,B) in [0,1]",
        "class_counts": class_counts,
        "img_dirs": [str(path.resolve()) for path in args.img_dirs],
    }
    (args.out_dir / "violin_pixel_values_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    plot_violin_and_auc(
        df_long=plot_values,
        pixels_xy=plot_pixels,
        class_order=class_order,
        save_png=args.out_dir / "violin_auc_plotted_pixels.png",
        save_pdf=args.out_dir / "violin_auc_plotted_pixels.pdf",
        n_pix_per_row=args.n_pix_per_row,
        auc_vmin=args.auc_vmin,
        auc_vmax=args.auc_vmax,
    )

    print(
        f"[Done] Tables include all {len(pixels_xy)} queue pixels; "
        f"the figure includes {n_plot} pixels."
    )


if __name__ == "__main__":
    main()
