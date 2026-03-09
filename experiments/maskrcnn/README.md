# Mask R-CNN experiment

This folder contains the **code and outputs** for the Mask R-CNN wound detection experiment.

- **Code:** `training_pipeline.ipynb`, `train_model.py`, `pipeline_utils.py`, `augmentation_strategy.py`
- **Outputs:** `checkpoints/` (models, training_results.json, training_report.md), `results/` (inference JSONs)
- **Data:** Shared at `../../data` (not copied here)

Run from this directory:
- Jupyter: open `training_pipeline.ipynb` (kernel cwd = this folder)
- CLI: `python train_model.py`
- Review: `python train_model.py --review checkpoints`

---

## Training Configuration (current)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `learning_rate` | 0.001 | SGD, linear-scaled for batch_size=2 |
| `epochs` | 80 | Raised from 50 to allow full convergence |
| `batch_size` | 2 | Single-GPU (RTX 4060) |
| `early_stop_patience` | 15 | Raised from 7 to not stop during cosine decay |
| `early_stop_min_delta` | 0.005 | Raised from 0.001 |
| Scheduler | LinearLR warmup (5 ep) → CosineAnnealingLR (75 ep) | Replaced StepLR(step_size=10, gamma=0.1) |
| Validation set | ~57 images (82/18 split of augmented data) | Raised from 16 (too small for COCO AP) |
| `image_size` | 1024×1024 | |
| `num_classes` | 9 | background + 8 wound classes |

## Root Causes Fixed (2026-02-28)

Four root causes were identified that caused 0.0 AP across all epochs and zero detections at inference:

1. **LR too low + StepLR too aggressive**: `lr=0.0005` with `StepLR(step_size=10, gamma=0.1)` dropped LR to `0.00005` at epoch 10, causing val_loss to plateau immediately. Fixed: `lr=0.001`, replaced with LinearLR warmup + CosineAnnealingLR.

2. **Early stopping triggered by LR drop, not overfitting**: Best epoch was 10 (exactly when LR dropped), patience=7 → stopped at epoch 17/50. The model was underfitting, not overfitting. Fixed: `patience=15`, `min_delta=0.005`, `epochs=80`.

3. **Validation set too small (16 images)**: `pycocotools` COCO AP evaluator requires ~50+ images to register non-zero AP; 16 images always yields 0.0. Fixed: split `data/augmented/annotations_augmented.json` 82/18 to get ~259 train / ~57 val images.

4. **Deprecated `pretrained=True` API**: Fixed to `weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT`.
