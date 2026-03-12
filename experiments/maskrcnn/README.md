# Mask R-CNN experiment

This folder contains the **code and outputs** for the Mask R-CNN wound detection experiment.

- **Code:** `training_pipeline.ipynb`, `train_model.py`, `pipeline_utils.py` (augmentation from `scripts/augmentation_strategy.py`)
- **Outputs:** `checkpoints/` (best_model.pth, last_checkpoint.pth, training_results.json, training_report.md), `results/` (inference JSONs)
- **Data:** Shared at `../../data` (not copied here)

## Training (primary: Notebook)

**Recommended:** Use `training_pipeline.ipynb` — open from this folder, set Kernel cwd to `experiments/maskrcnn`, run cells in order. Edit CONFIG (Part 2) for `data_mode`, `batch_size`, etc.

**Alternative CLI:**
- `python train_model.py` (default: clean_online_aug)
- `python train_model.py --data-mode clean_offline_aug`
- `python train_model.py --review checkpoints`

---

## Training Configuration (current)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `learning_rate` | 0.001 | SGD, linear-scaled for batch_size=2 |
| `epochs` | 50 | Max epochs; early stopping can stop sooner |
| `batch_size` | 2 | Single-GPU (RTX 4060) |
| `early_stop_patience` | 12 | Epochs without combined_AP50 improvement before stop |
| `early_stop_min_delta` | 0.003 | Min AP improvement; early stopping is AP-based only |
| Scheduler | LinearLR warmup (5 ep) → CosineAnnealingLR (45 ep) | Cosine decay for remaining epochs after warmup |
| Validation set | ~106 images (82/18 split of annotations_cleaned.json) | Raised from 16 (too small for COCO AP) |
| `image_size` | 1024×1024 | |
| `num_classes` | 9 | background + 8 wound classes (set from dataset; do not use len(coco_json['categories'])+1) |

## Pre-retraining checklist

- **num_classes:** The code uses `train_dataset.num_classes` (filtered/remapped classes). Do not build the model with `len(coco_json['categories'])+1` or you will get head/label mismatch and near-zero AP.
- **Old checkpoints:** If you previously trained with the wrong num_classes (e.g. from raw COCO categories), delete `checkpoints/best_model.pth` and `checkpoints/last_checkpoint.pth` before retraining, or training will fail or behave incorrectly.
- **Startup report:** When you start training, confirm the printed "Model num_classes" and "Dataset num_classes" match, and "Unique labels in sampled batches" are within [1, num_classes-1].
- **Checkpoint strategy:** Best model is selected by `combined_AP50` (not loss). Early stopping monitors combined_AP50 (mode=max). See [docs/MODEL_SELECTION_AND_EARLY_STOPPING.md](../../docs/MODEL_SELECTION_AND_EARLY_STOPPING.md) and [docs/MODEL_SELECTION_AND_CHECKPOINTS.md](../../docs/MODEL_SELECTION_AND_CHECKPOINTS.md).
- **Validation:** If the new label validation fails, fix the dataset or class mapping in `pipeline_utils.py` before training.

## Root Causes Fixed (2026-02-28)

Four root causes were identified that caused 0.0 AP across all epochs and zero detections at inference:

1. **LR too low + StepLR too aggressive**: `lr=0.0005` with `StepLR(step_size=10, gamma=0.1)` dropped LR to `0.00005` at epoch 10, causing val_loss to plateau immediately. Fixed: `lr=0.001`, replaced with LinearLR warmup + CosineAnnealingLR.

2. **Early stopping triggered by LR drop, not overfitting**: Best epoch was 10 (exactly when LR dropped), patience=7 → stopped at epoch 17/50. The model was underfitting, not overfitting. Fixed: Early stopping now monitors `combined_AP50` (mode=max), not loss; `patience=12`, `min_delta=0.003`, `epochs=50`.

3. **Validation set too small (16 images)**: `pycocotools` COCO AP evaluator requires ~50+ images to register non-zero AP; 16 images always yields 0.0. Fixed: split `data/annotations_cleaned.json` 82/18 to get ~424 train / ~106 val images. (Old `data/augmented/` was contaminated - do not use.)

4. **Deprecated `pretrained=True` API**: Fixed to `weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT`.

5. **num_classes from raw COCO categories**: The dataset filters to `TARGET_CLASSES_NAMES` (8 classes) and remaps labels to 1..8; `dataset.num_classes` is 9. The model was built with `len(coco_json['categories'])+1` (e.g. 17), so the classifier/mask heads had more outputs than the label range, leading to untrained logits and near-zero AP. Fixed: use `train_dataset.num_classes` (or `base_dataset.num_classes` when using Subset) for model creation; added label validation and startup report.
