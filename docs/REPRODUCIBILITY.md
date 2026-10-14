# Reproducibility notes

## Experimental environment

The public reproduction environment is named depi-ms. The package versions below match the software stack used for the experiments.

| Component | Version |
| --- | --- |
| Python | 3.8.13 |
| NumPy | 1.24.1 |
| pandas | 2.0.3 |
| PyTorch | 1.13.1+cu117 |
| torchvision | 0.14.1+cu117 |
| Pillow | 10.2.0 |
| scikit-learn | 1.3.2 |
| matplotlib | 3.7.5 |
| openpyxl | 3.1.5 |
| pyteomics | 4.7.5 |
| CUDA runtime used by PyTorch | 11.7 |
| cuDNN | 8.5.0 |

Activate the environment before running the training and interpretation scripts:

~~~bash
conda activate depi-ms
~~~

## LC-MS input and pseudo-image coordinates

Pseudo-image construction accepts centroided MS1 data in mzXML or mzML format. The NPC configuration uses m/z 220-1000, RT 50-900 s, an intensity threshold of 500, and 224 x 224 pseudo-images.

The m/z boundaries are fitted from QC runs by density equalization. RT remains uniformly discretized. Pixel-to-RT/m/z mapping uses the same adaptive m/z boundaries as pseudo-image construction.

Small repository examples are stored under `examples/mzXML/`. The pseudo-image converter accepts that directory directly. These files are pseudo-image conversion inputs, while density-equalized m/z boundaries are fitted from QC runs.

## Internal five-fold split workbook

Internal training uses metadata/five_fold_cv_patient.xlsx, which contains Fold1 to Fold5. Each sheet has two columns:

| train_id | val_id |
| --- | --- |
| A-1 | C-7 |
| B-2 | A-8 |

The NPC label mapping is:

- C -> 0: healthy control
- A/B/E/M -> 1: NPC/case

## Image naming

The training code recognizes sample images such as:

~~~text
A-1_0.png
C_7_0.png
B_10_6.png
~~~

Suffix 0 is the unshifted image. Suffix 6 is the large RT-shifted image used by the robustness-evaluation path.

## Main training command

The main NPC training configuration uses 50 epochs and a batch size of 32:

~~~bash
CUDA_VISIBLE_DEVICES=0 python depi_ms/training/train_depi_ms.py \
  --img-dir ./outputs/pseudo_images/IDE_img224 \
  --split-xlsx ./metadata/five_fold_cv_patient.xlsx \
  --epochs 50 \
  --batch-size 32 \
  --out-dir ./outputs/depi_ms
~~~

## Training and inference

Training and inference use separate entry points. depi_ms.training.train_depi_ms fits the five fold-specific models and saves model_last.pth under each fold directory. depi_ms.inference.infer_depi_ms loads these final-epoch checkpoints and performs fold evaluation without updating model weights.

## Visualization

depi_ms.visualization.plot_tsne follows the original DEPI-MS embedding workflow: the global feature returned by model(..., return_feat=True) is optionally reduced by PCA and then projected with t-SNE. The manuscript visualization uses the external pseudo-images and metadata/waibu_five_fold_cv.xlsx.

depi_ms.visualization.plot_violin reads every coordinate from heat_queue_xy.csv and writes complete pixel-value, median, and pairwise-AUC tables. The figure is limited to 20 queue pixels by default; --max-plot-pixels 0 removes the plotting limit.

## RT-warp consistency learning

For each training image, the model constructs a second view by applying a smooth RT-axis warp in tensor space. The implementation optimizes classification losses for both views together with a feature-consistency penalty:

~~~text
L = L_cls(x) + L_cls(T_RT(x)) + lambda_cons * L_cons
~~~

The default lambda_cons is 0.5.

## Grad-CAM consensus

Grad-CAM uses the trained fold checkpoints and the participant-level training IDs from five_fold_cv_patient.xlsx.

~~~bash
CUDA_VISIBLE_DEVICES=0 python -m depi_ms.interpretation.gradcam_consensus \
  --img-dir ./outputs/pseudo_images/IDE_img224 \
  --split-xlsx ./metadata/five_fold_cv_patient.xlsx \
  --checkpoint-dir ./outputs/depi_ms \
  --folds all \
  --cam-quantile 0.90 \
  --cam-freq-thr 0.25 \
  --cam-balance class_weight
~~~

The default target is the predicted class. Per-image CAMs are thresholded at the 0.90 quantile, and the cohort consensus retains pixels with activation frequency at or above 0.25 after class-balanced aggregation.

## External validation

The external cohort uses metadata/waibu_five_fold_cv.xlsx. Its five sheets contain the same 48 external sample IDs. Each trained internal fold model is applied to that full external cohort.

~~~bash
CUDA_VISIBLE_DEVICES=0 python -m depi_ms.inference.infer_depi_ms \
  --img-dir /path/to/external/img224_IDE_mz \
  --split-xlsx ./metadata/waibu_five_fold_cv.xlsx \
  --checkpoint-dir ./outputs/depi_ms \
  --view val0 \
  --out-dir ./outputs/external_validation
~~~

The five external-validation metric sets correspond to five independently trained internal fold models evaluated on the same external cohort.
