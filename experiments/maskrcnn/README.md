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
| `num_classes` | 9 | background + 8 wound classes (set from dataset; do not use len(coco_json['categories'])+1) |

## Pre-retraining checklist

- **num_classes:** The code uses `train_dataset.num_classes` (filtered/remapped classes). Do not build the model with `len(coco_json['categories'])+1` or you will get head/label mismatch and near-zero AP.
- **Old checkpoints:** If you previously trained with the wrong num_classes (e.g. from raw COCO categories), delete `checkpoints/best_model.pth`, `checkpoints/last.pt`, and `checkpoints/checkpoint_epoch_*.pth` before retraining, or training will fail or behave incorrectly.
- **Startup report:** When you start training, confirm the printed "Model num_classes" and "Dataset num_classes" match, and "Unique labels in sampled batches" are within [1, num_classes-1].
- **Validation:** If the new label validation fails, fix the dataset or class mapping in `pipeline_utils.py` before training.

## Root Causes Fixed (2026-02-28)

Four root causes were identified that caused 0.0 AP across all epochs and zero detections at inference:

1. **LR too low + StepLR too aggressive**: `lr=0.0005` with `StepLR(step_size=10, gamma=0.1)` dropped LR to `0.00005` at epoch 10, causing val_loss to plateau immediately. Fixed: `lr=0.001`, replaced with LinearLR warmup + CosineAnnealingLR.

2. **Early stopping triggered by LR drop, not overfitting**: Best epoch was 10 (exactly when LR dropped), patience=7 → stopped at epoch 17/50. The model was underfitting, not overfitting. Fixed: `patience=15`, `min_delta=0.005`, `epochs=80`.

3. **Validation set too small (16 images)**: `pycocotools` COCO AP evaluator requires ~50+ images to register non-zero AP; 16 images always yields 0.0. Fixed: split `data/augmented/annotations_augmented.json` 82/18 to get ~259 train / ~57 val images.

4. **Deprecated `pretrained=True` API**: Fixed to `weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT`.

5. **num_classes from raw COCO categories**: The dataset filters to `TARGET_CLASSES_NAMES` (8 classes) and remaps labels to 1..8; `dataset.num_classes` is 9. The model was built with `len(coco_json['categories'])+1` (e.g. 17), so the classifier/mask heads had more outputs than the label range, leading to untrained logits and near-zero AP. Fixed: use `train_dataset.num_classes` (or `base_dataset.num_classes` when using Subset) for model creation; added label validation and startup report.
