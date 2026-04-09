# YOLO11m + U-Net++ Experiment

This folder contains the **code and outputs** for the combined YOLO11m-seg + U-Net++ wound detection, segmentation, and infection classification experiment.

- **Code:** `training_pipeline.ipynb` (interactive), `train_model.py` (CLI + all stages), `pipeline_utils.py` (data handling)
- **Config:** `config.yaml` (all hyperparameters)
- **Outputs:** `checkpoints/`, `results/`, `reports/`
- **Data:** Shared at `../../data` (not copied here)
- **Tools:** `augment_offline.py` (offline augmentation), `build_wound_marker_dataset.py` (wound+marker annotations)
- **Docs:** `DOCUMENTATION.md` (full Arabic documentation with improvement details)

---

## Architecture

Multi-stage pipeline:

1. **YOLO11m-seg** (1024px) — detects wound bounding boxes and produces coarse instance segmentation masks
2. **U-Net++** (encoder & ROI size from `config.yaml`; default **EfficientNet-B1**, **256×256** for faster training) — refines segmentation on cropped wound ROIs with TTA
3. **Mask NMS** — removes duplicate overlapping predictions
4. **Marker Calibration** — uses detected 3x3 cm reference marker for per-image area calculation
5. **Infection Classifier** — lightweight MLP on wound texture/color features

All models are trained independently, then combined at inference.

---

## Training

### Option A: Notebook (interactive)

Open `training_pipeline.ipynb` from this folder, set Kernel cwd to `experiments/YOLO11m_UNetPP`, run cells in order.

### Option B: CLI

```bash
cd experiments/YOLO11m_UNetPP
python train_model.py --stage convert    # COCO -> YOLO label format
python train_model.py --stage yolo       # Train YOLO11m-seg
python train_model.py --stage unet       # Train U-Net++
python train_model.py --stage combined   # Run combined inference + eval (with COCO AP)
python train_model.py --stage infection  # Train infection classifier
python train_model.py --stage all        # All stages sequentially
```

### Optional: Data Expansion

```bash
python augment_offline.py                # 4x dataset expansion (257 -> ~1028 images)
python build_wound_marker_dataset.py     # Build wound+marker annotations
```

---

## Configuration

All hyperparameters are in `config.yaml`. Key settings:

### YOLO11m-seg

| Parameter | Value | Notes |
|-----------|-------|-------|
| `image_size` | 1024 | Higher resolution for better mask boundaries |
| `batch_size` | 4 | Reduced for 1024px |
| `epochs` | 100 | Early stopping patience=20 |
| `lr0` | 0.01 | SGD with momentum=0.937 |
| `mosaic` | 0.5 | Reduced from 1.0 for medical data |
| `mixup` | 0.0 | Disabled (not medically meaningful) |
| `close_mosaic` | 15 | Disable mosaic for last 15 epochs |
| `perspective` | 0.0 | Disabled — preserves marker geometry |

### U-Net++

Defaults favour **shorter training time**. For higher-quality ROI masks, set e.g. `encoder: efficientnet-b3`, `input_size: [384, 384]`, `epochs: 50`, `scheduler_T_max: 50`, `early_stop_patience: 10` (see `DOCUMENTATION.md`).

| Parameter | Default | Notes |
|-----------|---------|-------|
| `encoder` | efficientnet-b1 | Lighter than b3; use **b3** for best detail |
| `input_size` | 256×256 | Fewer pixels per forward pass than 384; use **[384, 384]** for finer boundaries |
| `batch_size` | 16 | |
| `epochs` | 35 | With `early_stop_patience: 6` |
| `lr` | 1e-4 | AdamW, CosineAnnealingLR (`scheduler_T_max` matches `epochs`) |
| `architecture` | unetplusplus | `unetplusplus` baseline, `deeplabv3plus` comparison option |
| `loss_type` | focal_dice | `focal_dice` or `focal_dice_boundary` |
| `roi_padding` | 0.1 | 10% expansion around GT bbox |
| `roi_crop_mode` | gt_only | `gt_only`, `mixed`, `yolo_predicted` |

### Combined Inference

| Parameter | Value | Notes |
|-----------|-------|-------|
| `yolo_conf_thresh` | 0.25 | Lowered from 0.5 for higher recall |
| `unet_mask_thresh` | 0.5 | |
| `pixels_per_cm` | 26.0 | Fallback when marker not detected |
| `marker_real_cm` | 3.0 | Physical size of reference marker |
| `multi_scale_refinement` | false | Enables multi-padding ROI fusion |
| `refinement_postprocess` | none | Optional boundary cleanup after thresholding |

---

## Outputs

```
checkpoints/
  yolo/                 # best.pt, last.pt
  unet/                 # best_model.pth, last_checkpoint.pth
  infection/            # infection_classifier.pth
results/
  yolo/                 # YOLO metrics, curves, predictions/
  unet/                 # U-Net++ metrics, curves (or results/unet/<experiment_name>/)
  combined/             # Combined Dice/IoU + COCO AP + wound areas + predictions/
  roi_cache/            # Cached YOLO ROI matches for train/val splits
  infection/            # Infection classification metrics
  metrics_summary.json  # Global summary (all stages)
reports/
  training_report.md    # Markdown report (all results including infection)
  segmentation_improvement_experiments.md
```

---

## Data Flow

1. **COCO annotations** (`data/wound_focus_clean/*.json`) are converted to **YOLO segmentation format** (one `.txt` label file per image with normalized polygon coordinates)
2. YOLO trains on full images (1024px) using Ultralytics API
3. ROI segmentation trains on configurable crops (256 / 384 / 512) with optional mixed GT + noisy + cached-YOLO ROI sampling
4. At combined inference: YOLO detects wounds -> ROI model refines with optional TTA and multi-scale ROI fusion -> Mask NMS -> marker calibration -> area calculation
5. COCO-style AP evaluation enables fair comparison with Mask R-CNN baseline
6. Infection classifier predicts wound infection status from texture/color features

---

## Improvements Over Baseline

See `DOCUMENTATION.md` for full details. Key changes:

- **+6% Dice** from lowered confidence threshold (0.5 -> 0.25) + TTA
- **Better mask quality** from 1024px YOLO + U-Net++ ROI input (raise `input_size` to 384 in config when you need maximum refinement)
- **Reduced overfitting** from tuned augmentation (mosaic 0.5, no mixup)
- **Focal+Dice loss** for sharper wound boundaries
- **Boundary-aware loss option** for AP75-sensitive contour refinement
- **Experiment-isolated outputs** via `experiment_name`
- **Structured experiment runner** via `scripts/run_segmentation_experiment.py`
- **Marker-based area calibration** (no more hardcoded pixels_per_cm)
- **Infection classification** via wound texture/color analysis
- **COCO AP evaluation** for fair comparison with Mask R-CNN

---

## Limitations

- **Dataset size** — 257 training images (use `augment_offline.py` for 4x expansion)
- **Research use only** — not validated for clinical deployment
- **Infection classifier** — based on simple texture features, not a clinical diagnostic tool
- **Marker detection** — requires retraining with `build_wound_marker_dataset.py` output
