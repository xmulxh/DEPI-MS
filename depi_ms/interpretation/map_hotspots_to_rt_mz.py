import argparse
import os
import numpy as np
import pandas as pd

def make_uniform_edges(x_min: float, x_max: float, pixel: int) -> np.ndarray:
    return np.linspace(float(x_min), float(x_max), int(pixel) + 1)

def load_edges_mz_npz(edges_npz: str, mz_min: float, mz_max: float, mz_pixel: int,
                      strict_endpoints: bool = True) -> np.ndarray:
    z = np.load(edges_npz, allow_pickle=True)
    if "edges_mz" not in z:
        raise ValueError(f"NPZ must contain 'edges_mz', got keys={list(z.keys())}")

    edges = np.asarray(z["edges_mz"], dtype=np.float64)
    if edges.ndim != 1 or edges.size != int(mz_pixel) + 1:
        raise ValueError(f"edges_mz length mismatch: got {edges.size}, expected {int(mz_pixel)+1}")

    if not np.all(np.diff(edges) > 0):
        raise ValueError("edges_mz must be strictly increasing")

    if strict_endpoints:
        a0, a1 = float(edges[0]), float(edges[-1])
        b0, b1 = float(mz_min), float(mz_max)
        tol = max(1e-3, 1e-6 * max(abs(b0), abs(b1), 1.0))
        if abs(a0 - b0) > tol or abs(a1 - b1) > tol:
            raise ValueError(
                "edges_mz endpoints do not match mz_min/mz_max. "
                f"edges=({a0:.6f},{a1:.6f}), mz_range=({b0:.6f},{b1:.6f})"
            )
    return edges

def map_xy_edges(df: pd.DataFrame,
                 x_col: str, y_col: str,
                 mz_edges: np.ndarray,
                 rt_min: float, rt_max: float, rt_pixel: int,
                 index_base: int = 0) -> pd.DataFrame:
    if index_base not in (0, 1):
        raise ValueError("index_base must be 0 or 1")

    x = df[x_col].astype(int).to_numpy()
    y = df[y_col].astype(int).to_numpy()
    if index_base == 1:
        x = x - 1
        y = y - 1

    mz_pixel = int(len(mz_edges) - 1)

    if x.min() < 0 or x.max() >= mz_pixel:
        raise ValueError(f"x out of range after base-fix: [{x.min()}, {x.max()}], expected [0, {mz_pixel-1}]")
    if y.min() < 0 or y.max() >= rt_pixel:
        raise ValueError(f"y out of range after base-fix: [{y.min()}, {y.max()}], expected [0, {rt_pixel-1}]")


    mz_low = mz_edges[x]
    mz_high = mz_edges[x + 1]
    mz_center = (mz_low + mz_high) / 2.0


    rt_edges = make_uniform_edges(rt_min, rt_max, rt_pixel)
    rt_low = rt_edges[y]
    rt_high = rt_edges[y + 1]
    rt_center_sec = (rt_low + rt_high) / 2.0
    rt_center_min = rt_center_sec / 60.0

    out = df.copy()
    out["_x0"] = x
    out["_y0"] = y
    out["mz_low"] = mz_low
    out["mz_high"] = mz_high
    out["mz_center"] = mz_center
    out["rt_low_sec"] = rt_low
    out["rt_high_sec"] = rt_high
    out["rt_center_sec"] = rt_center_sec
    out["rt_center_min"] = rt_center_min
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heat-csv", required=True)
    ap.add_argument("--out-csv", required=True)

    ap.add_argument("--x-col", default="x")
    ap.add_argument("--y-col", default="y")
    ap.add_argument("--index-base", type=int, default=0, choices=[0, 1])

    ap.add_argument("--mz-min", type=float, default=220.0)
    ap.add_argument("--mz-max", type=float, default=1000.0)
    ap.add_argument("--rt-min", type=float, default=50.0)
    ap.add_argument("--rt-max", type=float, default=900.0)

    ap.add_argument("--mz-pixel", type=int, default=224)
    ap.add_argument("--rt-pixel", type=int, default=224)


    ap.add_argument("--edges-npz", type=str, default="", help="NPZ containing edges_mz (adaptive m/z). If empty, use uniform.")

    args = ap.parse_args()

    df = pd.read_csv(args.heat_csv)
    if args.x_col not in df.columns or args.y_col not in df.columns:
        raise ValueError(f"CSV must contain columns '{args.x_col}' and '{args.y_col}', got {list(df.columns)}")

    if args.edges_npz:
        if not os.path.exists(args.edges_npz):
            raise FileNotFoundError(f"--edges-npz not found: {args.edges_npz}")
        mz_edges = load_edges_mz_npz(args.edges_npz, args.mz_min, args.mz_max, args.mz_pixel, strict_endpoints=True)
    else:
        mz_edges = make_uniform_edges(args.mz_min, args.mz_max, args.mz_pixel)

    out = map_xy_edges(
        df, args.x_col, args.y_col,
        mz_edges=mz_edges,
        rt_min=args.rt_min, rt_max=args.rt_max, rt_pixel=args.rt_pixel,
        index_base=args.index_base
    )
    out.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv} (rows={len(out)})")

if __name__ == "__main__":
    main()