# YOLO11m + U-Net++ Experiment

This folder contains the **code and outputs** for the combined YOLO11m-seg + U-Net++ wound detection and segmentation experiment.

- **Code:** `training_pipeline.ipynb` (interactive), `train_model.py` (CLI + all stages), `pipeline_utils.py` (data handling)
- **Config:** `config.yaml` (all hyperparameters)
- **Outputs:** `checkpoints/`, `results/`, `reports/`
- **Data:** Shared at `../../data` (not copied here)

---

## Architecture

Two-stage pipeline:

1. **YOLO11m-seg** — detects wound bounding boxes and produces coarse instance segmentation masks
2. **U-Net++** (EfficientNet-B3 encoder) — refines segmentation on cropped wound ROIs

Both models are trained independently, then combined at inference.

---

## Training

### Option A: Notebook (interactive)

Open `training_pipeline.ipynb` from this folder, set Kernel cwd to `experiments/YOLO11m_UNetPP`, run cells in order.

### Option B: CLI

```bash
cd experiments/YOLO11m_UNetPP
python train_model.py --stage convert   # COCO -> YOLO label format
python train_model.py --stage yolo      # Train YOLO11m-seg
python train_model.py --stage unet      # Train U-Net++
python train_model.py --stage combined  # Run combined inference + eval
python train_model.py --stage all       # All stages sequentially
```

---

## Configuration

All hyperparameters are in `config.yaml`. Key settings:

### YOLO11m-seg

| Parameter | Value | Notes |
|-----------|-------|-------|
| `image_size` | 640 | YOLO standard |
| `batch_size` | 8 | |
| `epochs` | 100 | Early stopping patience=20 |
| `lr0` | 0.01 | SGD with momentum=0.937 |
| `perspective` | 0.0 | Disabled — preserves 3x3 cm marker |

### U-Net++

| Parameter | Value | Notes |
|-----------|-------|-------|
| `encoder` | efficientnet-b3 | Pretrained on ImageNet |
| `input_size` | 256x256 | ROI crops |
| `batch_size` | 16 | |
| `epochs` | 50 | Early stopping patience=10 |
| `lr` | 1e-4 | AdamW, CosineAnnealingLR |
| `loss` | 0.5 BCE + 0.5 Dice | |
| `roi_padding` | 0.1 | 10% expansion around GT bbox |

---

## Outputs

```
checkpoints/
  yolo/                 # best.pt, last.pt
  unet/                 # best_model.pth, last_checkpoint.pth
results/
  yolo/                 # YOLO metrics, curves, predictions/
  unet/                 # U-Net++ metrics, curves
  combined/             # Combined metrics, predictions/
  metrics_summary.json  # Global summary (all stages)
reports/
  training_report.md    # Markdown report
```

---

## Data Flow

1. **COCO annotations** (`data/wound_focus_clean/*.json`) are converted to **YOLO segmentation format** (one `.txt` label file per image with normalized polygon coordinates)
2. YOLO trains on full images using Ultralytics API
3. U-Net++ trains on **ROI crops** — wound regions extracted from GT bounding boxes (with 10% padding)
4. At combined inference, YOLO provides bounding boxes, and U-Net++ refines each detected ROI

---

## Limitations

- **Dataset annotation quality** — wound-only (single class); subclass annotations are inconsistent
- **Research use only** — not validated for clinical deployment
- **Infection status** — derived from file naming (`-not-` convention); no independent clinical labels
