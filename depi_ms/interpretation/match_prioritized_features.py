#!/usr/bin/env python


"""Match LC-MS feature coordinates to consensus Grad-CAM RT-m/z hotspots."""

import argparse
import os
from typing import Iterable, List

import numpy as np
import pandas as pd


REQUIRED_HEAT_COLUMNS = [
    "freq",
    "mean_cam",
    "mz_low",
    "mz_high",
    "mz_center",
    "rt_low_sec",
    "rt_high_sec",
    "rt_center_sec",
    "rt_center_min",
]


def load_feature_table(path: str, sheet: str = None) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name=sheet)
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported feature-table format: {ext}")


def ensure_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")
    return pd.to_numeric(df[col], errors="coerce")


def sanitize_thr_name(thr: float) -> str:
    text = f"{thr:.6f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def parse_thresholds(values: Iterable[float]) -> List[float]:
    thresholds = sorted(set(float(v) for v in values))
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    for value in thresholds:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Threshold must be within [0, 1], got {value}")
    return thresholds


def prepare_heat_table(heat: pd.DataFrame) -> pd.DataFrame:
    heat = heat.copy()

    if "_x0" not in heat.columns:
        if "x" not in heat.columns:
            raise ValueError("Heat table must contain '_x0' or 'x'.")
        heat["_x0"] = pd.to_numeric(heat["x"], errors="coerce")

    if "_y0" not in heat.columns:
        if "y" not in heat.columns:
            raise ValueError("Heat table must contain '_y0' or 'y'.")
        heat["_y0"] = pd.to_numeric(heat["y"], errors="coerce")

    required = REQUIRED_HEAT_COLUMNS + ["_x0", "_y0"]
    missing = [col for col in required if col not in heat.columns]
    if missing:
        raise ValueError(
            "heat_queue_rt_mz.csv is missing required columns: "
            + ", ".join(missing)
        )

    for col in required:
        heat[col] = pd.to_numeric(heat[col], errors="coerce")

    heat = heat.dropna(subset=required).copy()

    invalid = (
        (heat["mz_high"] <= heat["mz_low"])
        | (heat["rt_high_sec"] <= heat["rt_low_sec"])
        | (heat["freq"] < 0)
        | (heat["freq"] > 1)
    )
    if invalid.any():
        raise ValueError(
            f"Heat table contains {int(invalid.sum())} invalid interval/frequency rows."
        )

    heat["_x0"] = heat["_x0"].astype(np.int32)
    heat["_y0"] = heat["_y0"].astype(np.int32)
    return (
        heat.sort_values("freq", ascending=False)
        .drop_duplicates(subset=["_x0", "_y0"], keep="first")
        .reset_index(drop=True)
    )


def prepare_feature_table(
    features: pd.DataFrame,
    id_col: str,
    rt_col: str,
    mz_col: str,
) -> pd.DataFrame:
    required = [id_col, rt_col, mz_col]
    missing = [col for col in required if col not in features.columns]
    if missing:
        raise ValueError(
            "Feature table is missing required columns: " + ", ".join(missing)
        )

    features = features.copy()
    features["_feature_row_id"] = np.arange(len(features), dtype=np.int64)
    features[rt_col] = ensure_numeric(features, rt_col)
    features[mz_col] = ensure_numeric(features, mz_col)

    valid = features[rt_col].notna() & features[mz_col].notna()
    return features.loc[valid].copy()


def match_features_to_hotspots(
    features: pd.DataFrame,
    heat: pd.DataFrame,
    rt_col: str,
    mz_col: str,
) -> pd.DataFrame:
    """Match using left-closed, right-open physical RT and m/z intervals."""
    output_parts = []
    metadata_columns = [
        "_x0",
        "_y0",
        "freq",
        "mean_cam",
        "mz_low",
        "mz_high",
        "mz_center",
        "rt_low_sec",
        "rt_high_sec",
        "rt_center_sec",
        "rt_center_min",
    ]

    rt_values = features[rt_col]
    mz_values = features[mz_col]

    for hot in heat[metadata_columns].itertuples(index=False, name=None):
        (
            x0,
            y0,
            freq,
            mean_cam,
            mz_low,
            mz_high,
            mz_center,
            rt_low,
            rt_high,
            rt_center,
            rt_center_min,
        ) = hot

        mask = (
            (rt_values >= rt_low)
            & (rt_values < rt_high)
            & (mz_values >= mz_low)
            & (mz_values < mz_high)
        )
        if not mask.any():
            continue

        matched = features.loc[mask].copy()
        matched["hotspot_x0"] = int(x0)
        matched["hotspot_y0"] = int(y0)
        matched["hotspot_freq"] = float(freq)
        matched["hotspot_mean_cam"] = float(mean_cam)
        matched["hotspot_mz_low"] = float(mz_low)
        matched["hotspot_mz_high"] = float(mz_high)
        matched["hotspot_mz_center"] = float(mz_center)
        matched["hotspot_rt_low_sec"] = float(rt_low)
        matched["hotspot_rt_high_sec"] = float(rt_high)
        matched["hotspot_rt_center_sec"] = float(rt_center)
        matched["hotspot_rt_center_min"] = float(rt_center_min)
        output_parts.append(matched)

    if not output_parts:
        return pd.DataFrame(columns=list(features.columns))

    result = pd.concat(output_parts, ignore_index=True)
    return (
        result.sort_values("hotspot_freq", ascending=False)
        .drop_duplicates(subset=["_feature_row_id"], keep="first")
        .reset_index(drop=True)
    )


def drop_internal_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "_feature_row_id" in df.columns:
        return df.drop(columns=["_feature_row_id"])
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Match LC-MS feature RT and m/z values to consensus hotspot intervals."
    )
    parser.add_argument("--heat-csv", type=str, required=True)
    parser.add_argument("--feature-table", type=str, required=True)
    parser.add_argument("--sheet", type=str, default=None)
    parser.add_argument("--out-dir", type=str, required=True)

    parser.add_argument("--id-col", type=str, default="No.")
    parser.add_argument("--rt-col", type=str, default="RT (s)")
    parser.add_argument("--mz-col", type=str, default="m/z")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.25, 0.45],
        help="Retain hotspot pixels with freq > threshold.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.heat_csv):
        raise FileNotFoundError(f"Heat CSV not found: {args.heat_csv}")
    if not os.path.exists(args.feature_table):
        raise FileNotFoundError(f"Feature table not found: {args.feature_table}")

    os.makedirs(args.out_dir, exist_ok=True)

    heat = prepare_heat_table(pd.read_csv(args.heat_csv))
    features = prepare_feature_table(
        load_feature_table(args.feature_table, sheet=args.sheet),
        id_col=args.id_col,
        rt_col=args.rt_col,
        mz_col=args.mz_col,
    )
    thresholds = parse_thresholds(args.thresholds)

    for threshold in thresholds:
        threshold_name = sanitize_thr_name(threshold)
        heat_thr = heat.loc[heat["freq"] > threshold].copy()

        matched = match_features_to_hotspots(
            features=features,
            heat=heat_thr,
            rt_col=args.rt_col,
            mz_col=args.mz_col,
        )
        matched = drop_internal_columns(matched)

        out_path = os.path.join(
            args.out_dir,
            f"matched_features_freq_gt_{threshold_name}.csv",
        )
        matched.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(
            f"[freq>{threshold:g}] hotspot_pixels={len(heat_thr)} | "
            f"matched_features={len(matched)}"
        )

    print(f"Results saved to: {args.out_dir}")


if __name__ == "__main__":
    main()