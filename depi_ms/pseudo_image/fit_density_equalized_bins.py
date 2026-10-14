"""Fit density-equalized m/z bin boundaries from QC LC-MS runs.

The m/z axis is partitioned by weighted quantiles of log(m/z), using log1p
intensity as peak mass. Retention time remains uniformly discretized.
"""

import json
import math
import argparse
from pathlib import Path
import numpy as np
from pyteomics import mzxml
try:
    from pyteomics import mzml
except Exception:
    mzml = None


def _open_ms_reader(fp: Path):
    """Open an mzXML or mzML reader with pyteomics."""
    ext = fp.suffix.lower()
    if ext == ".mzxml":
        return mzxml.read(str(fp))
    if ext == ".mzml":
        if mzml is None:
            raise RuntimeError("pyteomics.mzml not available but file is .mzML")
        return mzml.read(str(fp))
    raise ValueError(f"Unsupported file type: {ext}")


def _get_ms_level(spec) -> int:
    """Return the MS level from common mzXML/mzML field names."""
    ms_level = spec.get("msLevel", spec.get("ms level", spec.get("mslevel", 1)))
    try:
        return int(ms_level)
    except Exception:
        return 1


def _get_arrays(spec):
    """Return m/z and intensity arrays from a spectrum record."""
    mz_arr = spec.get("m/z array", None)
    if mz_arr is None:
        mz_arr = spec.get("mz array", spec.get("m/zArray", []))

    in_arr = spec.get("intensity array", None)
    if in_arr is None:
        in_arr = spec.get("intensityArray", [])

    mz_arr = np.asarray(mz_arr, dtype=np.float64)
    in_arr = np.asarray(in_arr, dtype=np.float64)
    return mz_arr, in_arr


def _hist_add_weighted(values, weights, edges, out_hist):
    """Accumulate weighted values into histogram bins in place."""
    if values.size == 0:
        return
    idx = np.searchsorted(edges, values, side="right") - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    np.add.at(out_hist, idx, weights)


def build_equal_mass_edges_from_hist(x_edges, hist_mass, nbins_target, eps_mass=1e-12):
    """Convert a weighted histogram into equal-mass bin boundaries."""
    mass = np.asarray(hist_mass, dtype=np.float64)
    mass = np.maximum(mass, 0.0)


    mass = mass + float(eps_mass)

    total = float(mass.sum())
    if total <= 0:
        raise RuntimeError("Histogram mass is zero; check mz_range / noise_threshold.")

    p = mass / total
    cdf = np.cumsum(p)
    cdf[-1] = 1.0

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])


    qs = np.linspace(0.0, 1.0, nbins_target + 1)


    edges_out = np.interp(qs, cdf, x_centers)


    edges_out[0] = float(x_centers[0])
    edges_out[-1] = float(x_centers[-1])
    return edges_out, cdf, x_centers


def enforce_min_width_edges_1d(edges_init, target_bins, min_width,
                               x_centers_for_inv, cdf_for_inv):
    """Enforce a minimum bin width while preserving the target bin count."""
    edges = np.asarray(edges_init, dtype=np.float64).copy()

    def cdf_at(x):
        return float(np.interp(x, x_centers_for_inv, cdf_for_inv, left=0.0, right=1.0))

    def inv_cdf(q):
        q = float(np.clip(q, 0.0, 1.0))
        return float(np.interp(q, cdf_for_inv, x_centers_for_inv))


    while True:
        widths = np.diff(edges)
        bad = np.where(widths < min_width)[0]
        if bad.size == 0:
            break
        i = int(bad[0])

        if i + 1 < len(edges) - 1:
            edges = np.delete(edges, i + 1)
        elif i > 0:
            edges = np.delete(edges, i)
        else:
            break


    while (len(edges) - 1) < target_bins:
        widths = np.diff(edges)
        candidates = np.where(widths >= 2.0 * min_width)[0]
        if candidates.size == 0:
            break


        masses = np.array([cdf_at(edges[i+1]) - cdf_at(edges[i]) for i in range(len(edges) - 1)], dtype=np.float64)
        best = int(candidates[np.argmax(masses[candidates])])

        left = edges[best]
        right = edges[best + 1]
        q_mid = 0.5 * (cdf_at(left) + cdf_at(right))
        x_mid = inv_cdf(q_mid)


        if not (left + min_width <= x_mid <= right - min_width):
            x_mid = 0.5 * (left + right)

        edges = np.insert(edges, best + 1, x_mid)


    while (len(edges) - 1) > target_bins:
        widths = np.diff(edges)
        i = int(np.argmin(widths))
        if 0 < (i + 1) < (len(edges) - 1):
            edges = np.delete(edges, i + 1)
        else:
            edges = np.delete(edges, max(1, min(len(edges) - 2, i)))


    eps = 1e-12
    edges[0] = edges_init[0]
    edges[-1] = edges_init[-1]
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + eps

    return edges


def fit_mz_adaptive_edges_from_qc(
    qc_dir: str,
    mz_min: float,
    mz_max: float,
    W: int,
    noise_threshold: float,
    u_hist_bins: int = 20000,
    min_mz_width_ppm: float = 20.0,
    eps_mass: float = 1e-12,
):
    """
    Learn m/z adaptive edges (W bins) from ALL QC files.

    Steps:
      - accumulate weighted histogram in u=log(mz)
      - equal-mass quantiles -> edges_u -> edges_mz
      - enforce minimum ppm width in u space

    Returns:
      edges_mz (len W+1), meta dict
    """
    qc_dir = Path(qc_dir)
    files = sorted([p for p in qc_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in (".mzxml", ".mzml")])
    if len(files) == 0:
        raise FileNotFoundError(f"No mzXML/mzML files found under: {qc_dir}")


    u_min = float(np.log(mz_min))
    u_max = float(np.log(mz_max))
    u_edges_fine = np.linspace(u_min, u_max, int(u_hist_bins) + 1, dtype=np.float64)
    u_mass = np.zeros(int(u_hist_bins), dtype=np.float64)

    n_files = 0
    n_scans_ms1_used = 0
    n_points_used = 0

    for fp in files:
        n_files += 1
        try:
            with _open_ms_reader(fp) as reader:
                for spec in reader:
                    if _get_ms_level(spec) != 1:
                        continue

                    mz_arr, in_arr = _get_arrays(spec)
                    if mz_arr.size == 0:
                        continue


                    mask = (mz_arr >= mz_min) & (mz_arr <= mz_max)
                    if not np.any(mask):
                        continue
                    mz_f = mz_arr[mask]
                    in_f = in_arr[mask]


                    mask2 = in_f > noise_threshold
                    if not np.any(mask2):
                        continue
                    mz_f = mz_f[mask2]
                    in_f = in_f[mask2]


                    u = np.log(mz_f)
                    w = np.log1p(in_f)
                    _hist_add_weighted(u, w, u_edges_fine, u_mass)

                    n_scans_ms1_used += 1
                    n_points_used += int(u.size)

        except Exception as e:
            print(f"[WARN] Failed reading {fp.name}: {e}")
            continue

        if (n_files % 10) == 0:
            print(f"[INFO] files={n_files}, ms1_scans_used={n_scans_ms1_used}, points_used={n_points_used}")

    if u_mass.sum() <= 0:
        raise RuntimeError("No valid m/z mass accumulated. Check mz range / threshold / file format.")


    edges_u_init, cdf_u, u_centers = build_equal_mass_edges_from_hist(
        x_edges=u_edges_fine,
        hist_mass=u_mass,
        nbins_target=W,
        eps_mass=eps_mass
    )


    min_u_width = math.log1p(float(min_mz_width_ppm) * 1e-6)

    edges_u = enforce_min_width_edges_1d(
        edges_init=edges_u_init,
        target_bins=W,
        min_width=min_u_width,
        x_centers_for_inv=u_centers,
        cdf_for_inv=cdf_u
    )

    if len(edges_u) != int(W) + 1:
        raise RuntimeError(
            f"Could not preserve {W} adaptive m/z bins after enforcing "
            f"the minimum width; got {len(edges_u) - 1} bins."
        )

    edges_mz = np.exp(edges_u)
    edges_mz[0] = mz_min
    edges_mz[-1] = mz_max


    eps = 1e-12
    for i in range(1, len(edges_mz)):
        if edges_mz[i] <= edges_mz[i - 1]:
            edges_mz[i] = edges_mz[i - 1] + eps

    meta = {
        "scheme": "mz_adaptive_only__rt_uniform__no_smoothing",
        "qc_dir": str(qc_dir),
        "mz_min": float(mz_min),
        "mz_max": float(mz_max),
        "W_mz_bins": int(W),
        "noise_threshold": float(noise_threshold),
        "u_hist_bins": int(u_hist_bins),
        "min_mz_width_ppm": float(min_mz_width_ppm),
        "eps_mass": float(eps_mass),
        "n_files": int(n_files),
        "n_ms1_scans_used": int(n_scans_ms1_used),
        "n_points_used": int(n_points_used),
        "note": "m/z density built on u=log(m) with weights log1p(intensity), without smoothing; RT edges are uniform."
    }
    return edges_mz, meta


def main():
    parser = argparse.ArgumentParser(
        description="Fit adaptive m/z edges from QC mzXML/mzML files; RT is uniform; no smoothing."
    )

    parser.add_argument(
        "--qc-dir", "--qc_dir", dest="qc_dir", type=str, required=True,
        help="QC directory containing mzXML or mzML files"
    )
    parser.add_argument(
        "--out-dir", "--out_dir", dest="out_dir", type=str, required=True,
        help="Directory in which the fitted edge file is saved"
    )


    parser.add_argument("--mz-min", dest="mz_min", type=float, default=220.0)
    parser.add_argument("--mz-max", dest="mz_max", type=float, default=1000.0)
    parser.add_argument("--rt-min", dest="rt_min", type=float, default=50.0,
                        help="RT min in SECONDS (RT edges will be uniform over this range)")
    parser.add_argument("--rt-max", dest="rt_max", type=float, default=900.0,
                        help="RT max in SECONDS (RT edges will be uniform over this range)")


    parser.add_argument("--H", type=int, default=224, help="RT bins (height); RT uses uniform binning")
    parser.add_argument("--W", type=int, default=224, help="m/z bins (width); m/z uses density-equalized adaptive binning")


    parser.add_argument("--noise-threshold", dest="noise_threshold", type=float, default=500.0)


    parser.add_argument("--u-hist-bins", dest="u_hist_bins", type=int, default=20000,
                        help="Fine histogram bins for u=log(m) density (no smoothing)")


    parser.add_argument("--min-mz-width-ppm", dest="min_mz_width_ppm", type=float, default=20.0)


    parser.add_argument("--eps-mass", dest="eps_mass", type=float, default=0)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


    edges_mz, meta = fit_mz_adaptive_edges_from_qc(
        qc_dir=args.qc_dir,
        mz_min=args.mz_min,
        mz_max=args.mz_max,
        W=args.W,
        noise_threshold=args.noise_threshold,
        u_hist_bins=args.u_hist_bins,
        min_mz_width_ppm=args.min_mz_width_ppm,
        eps_mass=args.eps_mass
    )


    edges_rt = np.linspace(float(args.rt_min), float(args.rt_max), int(args.H) + 1, dtype=np.float64)


    out_path = out_dir / f"edges_mz_adaptive_rt_uniform_H{args.H}_W{args.W}.npz"
    np.savez_compressed(
        str(out_path),
        edges_rt=edges_rt.astype(np.float64),
        edges_mz=edges_mz.astype(np.float64),
        meta_json=json.dumps(meta, ensure_ascii=False, indent=2),
    )

    print("\n[OK] Saved edges:")
    print(f"  {out_path}")
    print(f"  edges_rt: len={len(edges_rt)} (uniform), range=({edges_rt[0]:.3f}, {edges_rt[-1]:.3f})")
    print(f"  edges_mz: len={len(edges_mz)} (adaptive), range=({edges_mz[0]:.3f}, {edges_mz[-1]:.3f})")
    print("\nMeta:")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
