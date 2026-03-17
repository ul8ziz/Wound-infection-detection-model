# Mask R-CNN Experiment

This folder contains the **code and outputs** for the Mask R-CNN wound detection experiment.

- **Code:** `training_pipeline.ipynb` (main), `training_pipeline_ru.ipynb` (Russian), `train_model.py` (CLI + all helpers + dataset validation), `pipeline_utils.py`
- **Outputs:** `checkpoints/`, `results/`, `reports/`
- **Data:** Shared at `../../data` (not copied here)

---

## Revised Project Scope

The project focuses on:

1. **Wound isolation / wound segmentation** — segmenting the wound region from surrounding tissue
2. **Infected vs. non-infected wound analysis** — distinguishing infected from non-infected cases (file names with `-not-` indicate no infection)
3. **Wound area computation** — using the 3×3 cm marker as scale reference

It is **not currently reliable** for robust fine-grained multi-class segmentation of infection subclasses (fibrin, granulation, edema, hyperemia, necrosis). Manual dataset inspection suggests that annotations for these subclasses are inconsistent or imprecise. Training runs complete successfully, but segmentation metrics for detailed subclasses are near-zero; dataset annotation quality is a major limiting factor.

---

## Training

### Wound-only baseline (recommended)

After building the wound-only dataset (`scripts/build_wound_only_dataset.py`):

```bash
python train_model.py --validate-only   # Pre-training validation only
python train_model.py              # Full training (runs validation first)
python train_model.py --epochs 1   # Quick sanity check
```

Outputs: `checkpoints/`, `results/`, `reports/`

### Notebook (interactive)

Use `training_pipeline.ipynb` — open from this folder, set Kernel cwd to `experiments/maskrcnn`, run cells in order. Edit CONFIG for `batch_size`, `epochs`, etc.

---

## Training Configuration (Current)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `learning_rate` | 0.001 | SGD |
| `epochs` | 50 | Max epochs; early stopping can stop sooner |
| `batch_size` | 2 | Single-GPU |
| `early_stop_patience` | 12 | Epochs without combined_AP50 improvement before stop |
| `early_stop_min_delta` | 0.003 | Min AP improvement |
| Scheduler | StepLR (step=5, gamma=0.1) | |
| Dataset | `data/wound_focus_clean/` | train_wound_only.json, val_wound_only.json, test_wound_only.json |
| `image_size` | 512×512 | |
| `num_classes` | 2 | background + wound |

---

## Experimental Findings

| Aspect | Status |
|--------|--------|
| Training | Runs successfully; loss converges |
| Detection (bbox) | Weak; below expectations |
| Segmentation (subclasses) | Near-zero; annotation quality limits performance |
| Inference | Pipeline outputs wound area, infection presence, and confidence |

---

## Pre-retraining Checklist

- **num_classes:** The code uses `train_dataset.num_classes` (filtered/remapped classes). Do not build the model with `len(coco_json['categories'])+1` or you will get head/label mismatch and near-zero AP.
- **Old checkpoints:** If you previously trained with the wrong num_classes (e.g. from raw COCO categories), delete `checkpoints/best_model.pth` and `checkpoints/last_checkpoint.pth` before retraining, or training will fail or behave incorrectly.
- **Startup report:** When you start training, confirm the printed "Model num_classes" and "Dataset num_classes" match, and "Unique labels in sampled batches" are within [1, num_classes-1].
- **Checkpoint strategy:** Best model is selected by `combined_AP50` (not loss). Early stopping monitors combined_AP50 (mode=max). See [docs/MODEL_SELECTION_AND_EARLY_STOPPING.md](../../docs/MODEL_SELECTION_AND_EARLY_STOPPING.md) and [docs/MODEL_SELECTION_AND_CHECKPOINTS.md](../../docs/MODEL_SELECTION_AND_CHECKPOINTS.md).
- **Validation:** If the new label validation fails, fix the dataset or class mapping in `pipeline_utils.py` before training.
- **Data paths:** Ensure `data/original_data/` contains task folders and annotations. `file_name` in COCO annotations should use `original_data/task_N/data/...` paths.

---

## Root Causes Fixed (2026-02-28)

Four root causes were identified that caused 0.0 AP across all epochs and zero detections at inference:

1. **LR too low + StepLR too aggressive**: `lr=0.0005` with `StepLR(step_size=10, gamma=0.1)` dropped LR to `0.00005` at epoch 10, causing val_loss to plateau immediately. Fixed: `lr=0.001`, replaced with LinearLR warmup + CosineAnnealingLR.

2. **Early stopping triggered by LR drop, not overfitting**: Best epoch was 10 (exactly when LR dropped), patience=7 → stopped at epoch 17/50. The model was underfitting, not overfitting. Fixed: Early stopping now monitors `combined_AP50` (mode=max), not loss; `patience=12`, `min_delta=0.003`, `epochs=50`.

3. **Validation set too small (16 images)**: `pycocotools` COCO AP evaluator requires ~50+ images to register non-zero AP; 16 images always yields 0.0. Fixed: split `data/original_data/annotations_cleaned.json` 82/18 to get ~424 train / ~106 val images. (Old `data/augmented/` was contaminated - do not use.)

4. **Deprecated `pretrained=True` API**: Fixed to `weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT`.

5. **num_classes from raw COCO categories**: The dataset filters to `TARGET_CLASSES_NAMES` (8 classes) and remaps labels to 1..8; `dataset.num_classes` is 9. The model was built with `len(coco_json['categories'])+1` (e.g. 17), so the classifier/mask heads had more outputs than the label range, leading to untrained logits and near-zero AP. Fixed: use `train_dataset.num_classes` (or `base_dataset.num_classes` when using Subset) for model creation; added label validation and startup report.

---

## Limitations

- **Dataset annotation quality** — subclasses (edema, hyperemia, necrosis, granulation, fibrin, pus) are inconsistent or imprecise; not suitable for reliable fine-grained segmentation.
- **Model performance** — detection and segmentation metrics are weak; near-zero segmentation AP for subclasses.
- **Research use only** — not validated for clinical deployment.
- **Infection status** — derived from file naming (`-not-` convention); no independent clinical labels.

---

## Future Work

- Dataset cleaning and annotation verification
- Simplification to wound-only segmentation
- Infection vs. non-infection classification as primary task
- Detailed subclass segmentation only if better annotations become available
