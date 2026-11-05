"""Convert centroided LC-MS data into DEPI-MS RT-m/z pseudo-images.

Retention time is uniformly discretized. The m/z axis can use either uniform
bins or density-equalized boundaries loaded from an NPZ file.
"""

import os
import re
import argparse
import random
import numpy as np
from pyteomics import mzxml, mzml
from PIL import Image


def make_uniform_edges(x_min, x_max, nbins):
    """Uniform edges, length nbins+1."""
    return np.linspace(float(x_min), float(x_max), int(nbins) + 1, dtype=np.float64)


def bin_x_uniform(x, x_min, x_max, nbins):
    """Assign values to uniform bins using one-based indices."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.array([], dtype=int)


    edges = make_uniform_edges(x_min, x_max, nbins)
    idx = np.searchsorted(edges, x, side="right") - 1
    idx = np.clip(idx, 0, nbins - 1)
    return idx + 1


def bin_by_edges(x, edges):
    """Assign values to non-uniform bins using one-based indices."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.array([], dtype=int)

    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("edges must be 1D with length >= 2")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("edges must be strictly increasing")

    idx = np.searchsorted(edges, x, side="right") - 1
    idx = np.clip(idx, 0, edges.size - 2)
    return idx + 1


def _validate_edges_mz(edges_mz, mz_range, mz_pixel, strict_endpoints=True):
    """Validate adaptive m/z boundaries against the requested grid."""
    edges_mz = np.asarray(edges_mz, dtype=np.float64)
    if edges_mz.size != int(mz_pixel) + 1:
        raise ValueError(
            f"edges_mz length mismatch: got {edges_mz.size}, expect {int(mz_pixel) + 1} "
            f"(mz_pixel={mz_pixel})"
        )
    if not np.all(np.diff(edges_mz) > 0):
        raise ValueError("edges_mz must be strictly increasing")

    if strict_endpoints:
        a0, a1 = float(edges_mz[0]), float(edges_mz[-1])
        b0, b1 = float(mz_range[0]), float(mz_range[1])
        tol = max(1e-3, 1e-6 * max(abs(b0), abs(b1), 1.0))
        if abs(a0 - b0) > tol or abs(a1 - b1) > tol:
            raise ValueError(
                f"edges_mz endpoints do not match mz_range.\n"
                f"  edges_mz: ({a0:.6f}, {a1:.6f})\n"
                f"  mz_range: ({b0:.6f}, {b1:.6f})\n"
                f"Fit edges with the same mz_min/mz_max used for pseudo-image generation."
            )
    return edges_mz


def get_mz_shift(mz, mean=1.12361, sd=4.076444):
    """Apply ppm-scale Gaussian m/z perturbation."""
    mz = np.asarray(mz, dtype=float)
    if mz.size == 0:
        return mz
    errors = np.random.normal(loc=mean, scale=sd, size=mz.shape[0])
    base = np.where(mz < 400.0, 400.0, mz)
    return mz - (base * errors) / 1e6


def _open_ms_reader(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".mzxml":
        return mzxml.read(file_path)
    if ext == ".mzml":
        return mzml.read(file_path)
    raise ValueError(f"Unsupported file type: {ext} (expect .mzxml or .mzml)")


def _get_rt_from_spectrum(spec):
    """Extract retention time from mzXML or mzML spectrum metadata."""
    rt = None
    unit = None

    scan0 = None
    try:
        scan_list = spec.get("scanList", {}).get("scan", [])
        if scan_list:
            scan0 = scan_list[0]
    except Exception:
        scan0 = None

    for container in (spec, scan0):
        if not container:
            continue
        for k in ("scan start time", "retentionTime", "retention time"):
            if k in container:
                rt = container[k]
                break
        if rt is not None:
            break

    if isinstance(rt, dict):
        unit = rt.get("unitName") or rt.get("unit_name") or rt.get("unit")
        rt = rt.get("value", rt)

    if isinstance(rt, (tuple, list)) and len(rt) > 0:
        if len(rt) > 1 and isinstance(rt[1], str):
            unit = unit or rt[1]
        rt = rt[0]

    if isinstance(rt, str) and rt.startswith("PT"):
        body = rt[2:]
        if body.endswith("S"):
            return float(body[:-1])
        if body.endswith("M"):
            return float(body[:-1]) * 60.0
        if body.endswith("H"):
            return float(body[:-1]) * 3600.0

    try:
        rt_val = float(rt)
    except Exception:
        return 0.0

    if isinstance(unit, str):
        u = unit.lower()
        if u.startswith("min"):
            return rt_val * 60.0
        if u.startswith("sec"):
            return rt_val

    return rt_val


def _read_ms_file(file_name, mz_range, noise_threshold):
    """Read filtered MS1 spectra and retention times from mzXML or mzML."""
    peaks_list = []
    rt_list = []

    with _open_ms_reader(file_name) as reader:
        for spec in reader:
            ms_level = spec.get("msLevel", spec.get("ms level", spec.get("mslevel", 1)))
            try:
                ms_level = int(ms_level)
            except Exception:
                ms_level = 1
            if ms_level != 1:
                continue

            mz_array = spec.get("m/z array", None)
            if mz_array is None:
                mz_array = spec.get("mz array", spec.get("m/zArray", []))
            mz_array = np.asarray(mz_array, dtype=float)

            int_array = spec.get("intensity array", None)
            if int_array is None:
                int_array = spec.get("intensityArray", [])
            int_array = np.asarray(int_array, dtype=float)

            if mz_array.size == 0:
                continue


            mask = (mz_array >= mz_range[0]) & (mz_array <= mz_range[1]) & (int_array > noise_threshold)
            if not np.any(mask):
                peaks_list.append(np.zeros((0, 2), dtype=float))
            else:
                mz_f = mz_array[mask]
                int_f = int_array[mask]
                peaks_list.append(np.stack([mz_f, int_f], axis=1))

            rt_list.append(_get_rt_from_spectrum(spec))

    return peaks_list, np.asarray(rt_list, dtype=float)


def _process_one_spectrum(
    idx,
    peaks_list,
    mz_shift,
    mean,
    sd,
    int_shift,
    int_scale,
    mz_range,
    noise_threshold,
    mz_pixel,
    edges_mz=None,
):
    """
    Process one MS1 spectrum into a length=mz_pixel vector:
      - optional mz jitter
      - optional intensity scaling
      - range + threshold filtering
      - log10 intensity
      - bin by m/z (adaptive edges if provided; else uniform)
      - sum intensities per bin
    """
    x = peaks_list[idx]
    if x.size == 0:
        return np.zeros(mz_pixel, dtype=np.float64)

    mz = x[:, 0].copy()
    inten = x[:, 1].copy()

    if mz_shift:
        mz = get_mz_shift(mz, mean=mean, sd=sd)

    if int_shift:
        inten = inten * float(int_scale)


    mask_mz = (mz >= mz_range[0]) & (mz <= mz_range[1])
    mz = mz[mask_mz]
    inten = inten[mask_mz]
    if mz.size == 0:
        return np.zeros(mz_pixel, dtype=np.float64)


    mask_int = inten > noise_threshold
    mz = mz[mask_int]
    inten = inten[mask_int]
    if mz.size == 0:
        return np.zeros(mz_pixel, dtype=np.float64)

    inten = np.log10(inten)

    if edges_mz is not None:
        cells = bin_by_edges(mz, edges_mz)
    else:
        cells = bin_x_uniform(mz, x_min=mz_range[0], x_max=mz_range[1], nbins=mz_pixel)

    out = np.zeros(mz_pixel, dtype=np.float64)
    np.add.at(out, cells - 1, inten)
    return out


def convert2image(
    file_name,
    mz_range=(70.0, 1000.0),
    rt_range=(50.0, 1000.0),
    mz_pixel=1024,
    rt_pixel=1024,
    noise_threshold=500.0,
    output_path=".",
    mz_shift=False,
    mean=1.12361,
    sd=4.076444,
    rt_shift=False,
    rt_diff=10.0,
    rt_sd=2.0,
    int_shift=False,
    int_times=1.1,
    suffix="",
    peaks_list=None,
    rt=None,
    edges_mz=None,
):
    """Convert one mzXML or mzML file into an RT-m/z pseudo-image.

    Retention time uses uniform bins. The m/z axis uses adaptive bins when
    edges_mz is provided.
    """

    os.makedirs(output_path, exist_ok=True)


    if peaks_list is None or rt is None:
        print("Reading mzXML/mzML and extracting peaks & RT ...")
        peaks_list, rt = _read_ms_file(file_name, mz_range=mz_range, noise_threshold=noise_threshold)
    else:
        rt = np.asarray(rt, dtype=float).copy()
        peaks_list = list(peaks_list)


    if rt.size == 0:
        raise RuntimeError("No RT values read from file.")
    if len(peaks_list) == 0:
        raise RuntimeError("No MS1 spectra found.")


    mz_max_actual = 0.0
    for p in peaks_list:
        if p.size > 0:
            mz_max_actual = max(mz_max_actual, float(p[:, 0].max()))
    print(f"m/z max in file: {mz_max_actual:.2f}")


    rt_max = float(rt.max())
    if rt_max < 100.0:
        print(f"RT appears to be in minutes (max={rt_max:.2f}), converting to seconds.")
        rt = rt * 60.0
    else:
        print(f"RT appears to be in seconds (max={rt_max:.2f}), keep as is.")

    base_name = os.path.basename(file_name)
    base_name = re.sub(r"\.(mzxml|mzml)$", "", base_name, flags=re.IGNORECASE)


    print("RT shift and filtering spectra by RT range ...")
    if rt_shift:
        rt_delta = np.random.normal(loc=float(rt_diff), scale=float(rt_sd))
        rt = rt + rt_delta
        print(f"RT shift value: {rt_delta:.4f}")

    mask_rt = (rt > rt_range[0]) & (rt < rt_range[1])
    rt = rt[mask_rt]
    peaks_list = [p for p, keep in zip(peaks_list, mask_rt) if keep]
    if len(peaks_list) == 0:
        raise RuntimeError("No spectra remain after RT filtering.")

    mz_pixel = int(mz_pixel)
    rt_pixel = int(rt_pixel)


    edges_rt = make_uniform_edges(rt_range[0], rt_range[1], rt_pixel)


    if edges_mz is None:
        edges_mz_use = None
    else:
        edges_mz_use = _validate_edges_mz(edges_mz, mz_range=mz_range, mz_pixel=mz_pixel, strict_endpoints=True)


    low = min(1.05, float(int_times))
    high = max(1.05, float(int_times))
    int_scale = np.random.uniform(low, high)

    print("Binning m/z (and applying mz/intensity shift if enabled) ...")
    processed_cols = []
    for i in range(len(peaks_list)):
        col = _process_one_spectrum(
            idx=i,
            peaks_list=peaks_list,
            mz_shift=mz_shift,
            mean=mean,
            sd=sd,
            int_shift=int_shift,
            int_scale=int_scale,
            mz_range=mz_range,
            noise_threshold=noise_threshold,
            mz_pixel=mz_pixel,
            edges_mz=edges_mz_use,
        )
        processed_cols.append(col)


    peaks = np.column_stack(processed_cols)


    print("Binning RT (uniform) ...")
    cell_rt = bin_by_edges(rt, edges_rt)


    final_peaks = np.zeros((mz_pixel, rt_pixel), dtype=np.float64)

    for j in range(peaks.shape[1]):
        b = int(cell_rt[j]) - 1
        if 0 <= b < rt_pixel:
            final_peaks[:, b] += peaks[:, j]


    print("Generating pseudo-MS image ...")
    img_arr = final_peaks.T

    vmin = float(img_arr.min())
    vmax = float(img_arr.max())
    if vmax > vmin:
        norm = (img_arr - vmin) / (vmax - vmin)
    else:
        norm = np.zeros_like(img_arr)


    img_uint8 = ((1.0 - norm) * 255.0).astype(np.uint8)

    img = Image.fromarray(img_uint8, mode="L")
    png_path = os.path.join(output_path, base_name + suffix + ".png")
    img.save(png_path)
    print(f"Saved pseudo-MS PNG to: {png_path}")


def _load_edges_npz(path):
    """
    Load edges_mz from NPZ.
    Accepts both:
      - edges_ide_bin_*.npz (contains edges_rt + edges_mz)
      - edges_mz_adaptive_rt_uniform_*.npz (contains edges_rt + edges_mz)
    Only edges_mz is used for adaptive m/z binning; RT remains uniform.
    """
    z = np.load(path, allow_pickle=True)
    if "edges_mz" not in z:
        raise ValueError(f"NPZ does not contain 'edges_mz': {path}")
    return z["edges_mz"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LC-MS files using fitted density-equalized m/z bins.")
    parser.add_argument(
        "--file-name", "--file_name", dest="file_name", type=str, required=True,
        help="Input mzXML/mzML file or a directory containing mzXML/mzML files",
    )


    parser.add_argument("--mz-min", type=float, default=220.0)
    parser.add_argument("--mz-max", type=float, default=1000.0)
    parser.add_argument("--rt-min", type=float, default=50.0)
    parser.add_argument("--rt-max", type=float, default=900.0)


    parser.add_argument("--mz-pixel", type=int, default=224)
    parser.add_argument("--rt-pixel", type=int, default=224)
    parser.add_argument("--noise-threshold", type=float, default=500.0)

    parser.add_argument(
        "--output-path", type=str, required=True,
        help="Base output directory",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument(
        "--edges-npz", "--edges_npz", dest="edges_npz", type=str, required=True,
        help="NPZ produced by fit_density_equalized_bins.py; contains adaptive m/z edges.",
    )

    args = parser.parse_args()


    if args.seed is not None:
        print(f"[INFO] Setting random seed to: {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)


    in_path = args.file_name
    base_out = args.output_path


    configs = [
        dict(mz_pixel=int(args.mz_pixel), rt_pixel=int(args.rt_pixel),
             out_dir=os.path.join(base_out, "IDE_img224")),
    ]


    edges_mz_loaded = None
    if args.edges_npz:
        if not os.path.exists(args.edges_npz):
            raise FileNotFoundError(f"--edges_npz not found: {args.edges_npz}")
        edges_mz_loaded = _load_edges_npz(args.edges_npz)
        print(f"[INFO] Loaded edges_mz from: {args.edges_npz} (len={len(edges_mz_loaded)})")


    suffix_plan = [
        (0, dict(mz_shift=False, rt_shift=False, int_shift=False, rt_diff=None, rt_sd=None)),
        (6, dict(mz_shift=False, rt_shift=True,  int_shift=False, rt_diff=60.0, rt_sd=10.0)),

    ]

    if os.path.isdir(in_path):
        input_dir = in_path
        print(f"Input directory: {input_dir}")
        print("Loop over all *.mzXML/*.mzML files...")

        for cfg in configs:
            out_dir = cfg["out_dir"]
            mz_pix = cfg["mz_pixel"]
            rt_pix = cfg["rt_pixel"]
            os.makedirs(out_dir, exist_ok=True)


            edges_mz_for_cfg = _validate_edges_mz(
                edges_mz_loaded,
                mz_range=(args.mz_min, args.mz_max),
                mz_pixel=mz_pix,
                strict_endpoints=True,
            )

            print(f"\n=== Resolution {mz_pix} x {rt_pix} -> {out_dir} ===")

            for fname in os.listdir(input_dir):
                if not fname.lower().endswith((".mzxml", ".mzml")):
                    continue
                full_path = os.path.join(input_dir, fname)
                base_name = os.path.basename(full_path)
                base_name = re.sub(r"\.(mzxml|mzml)$", "", base_name, flags=re.IGNORECASE)

                print(f"\n--- Processing file: {full_path} ---")


                peaks_cache, rt_cache = _read_ms_file(
                    full_path,
                    mz_range=(args.mz_min, args.mz_max),
                    noise_threshold=args.noise_threshold
                )

                for k, plan in suffix_plan:
                    suffix = f"_{k}"
                    png_path = os.path.join(out_dir, base_name + suffix + ".png")
                    if os.path.exists(png_path):
                        print(f"[Skip] {base_name}{suffix} (found {png_path})")
                        continue


                    rt_shift = plan["rt_shift"]
                    rt_diff_val = 10.0 if plan["rt_diff"] is None else plan["rt_diff"]
                    rt_sd_val = 2.0 if plan["rt_sd"] is None else plan["rt_sd"]

                    print(
                        f"[Run] {base_name}{suffix} | "
                        f"mz_shift={plan['mz_shift']}, rt_shift={rt_shift}, int_shift={plan['int_shift']}, "
                        f"rt_diff={rt_diff_val}, rt_sd={rt_sd_val}"
                    )

                    convert2image(
                        file_name=full_path,
                        mz_range=(args.mz_min, args.mz_max),
                        rt_range=(args.rt_min, args.rt_max),
                        mz_pixel=mz_pix,
                        rt_pixel=rt_pix,
                        noise_threshold=args.noise_threshold,
                        output_path=out_dir,
                        mz_shift=plan["mz_shift"],
                        rt_shift=rt_shift,
                        rt_diff=rt_diff_val,
                        rt_sd=rt_sd_val,
                        int_shift=plan["int_shift"],
                        suffix=suffix,
                        peaks_list=peaks_cache,
                        rt=rt_cache,
                        edges_mz=edges_mz_for_cfg,
                    )

        print("\nAll files in directory have been processed.")

    else:

        full_path = in_path
        print(f"Input file: {full_path}")

        base_name = os.path.basename(full_path)
        base_name = re.sub(r"\.(mzxml|mzml)$", "", base_name, flags=re.IGNORECASE)

        for cfg in configs:
            out_dir = cfg["out_dir"]
            mz_pix = cfg["mz_pixel"]
            rt_pix = cfg["rt_pixel"]
            os.makedirs(out_dir, exist_ok=True)

            edges_mz_for_cfg = _validate_edges_mz(
                edges_mz_loaded,
                mz_range=(args.mz_min, args.mz_max),
                mz_pixel=mz_pix,
                strict_endpoints=True,
            )

            print(f"\n=== Resolution {mz_pix} x {rt_pix} -> {out_dir} ===")

            peaks_cache, rt_cache = _read_ms_file(
                full_path,
                mz_range=(args.mz_min, args.mz_max),
                noise_threshold=args.noise_threshold
            )

            for k, plan in suffix_plan:
                suffix = f"_{k}"
                png_path = os.path.join(out_dir, base_name + suffix + ".png")
                if os.path.exists(png_path):
                    print(f"[Skip] {base_name}{suffix} (found {png_path})")
                    continue

                rt_shift = plan["rt_shift"]
                rt_diff_val = 10.0 if plan["rt_diff"] is None else plan["rt_diff"]
                rt_sd_val = 2.0 if plan["rt_sd"] is None else plan["rt_sd"]

                print(
                    f"[Run] {base_name}{suffix} | "
                    f"mz_shift={plan['mz_shift']}, rt_shift={rt_shift}, int_shift={plan['int_shift']}, "
                    f"rt_diff={rt_diff_val}, rt_sd={rt_sd_val}"
                )

                convert2image(
                    file_name=full_path,
                    mz_range=(args.mz_min, args.mz_max),
                    rt_range=(args.rt_min, args.rt_max),
                    mz_pixel=mz_pix,
                    rt_pixel=rt_pix,
                    noise_threshold=args.noise_threshold,
                    output_path=out_dir,
                    mz_shift=plan["mz_shift"],
                    rt_shift=rt_shift,
                    rt_diff=rt_diff_val,
                    rt_sd=rt_sd_val,
                    int_shift=plan["int_shift"],
                    suffix=suffix,
                    peaks_list=peaks_cache,
                    rt=rt_cache,
                    edges_mz=edges_mz_for_cfg,
                )
