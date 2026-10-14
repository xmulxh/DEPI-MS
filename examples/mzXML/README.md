# Example mzXML files

Place a small set of centroided LC-MS MS1 mzXML files in this directory. The pseudo-image converter can process the entire directory with:

```bash
python depi_ms/pseudo_image/generate_pseudo_images.py \
  --file-name ./examples/mzXML \
  --output-path ./outputs/example_pseudo_images \
  --edges-npz ./outputs/binning/edges_mz_adaptive_rt_uniform_H224_W224.npz \
  --mz-min 220 --mz-max 1000 \
  --rt-min 50 --rt-max 900 \
  --mz-pixel 224 --rt-pixel 224 \
  --noise-threshold 500
```

The density-equalized m/z grid is fitted from QC samples, while the files in this directory serve as example conversion inputs. For use with a split workbook, each mzXML filename stem matches the corresponding sample ID, for example `A-1.mzXML` -> `A-1_0.png`.
