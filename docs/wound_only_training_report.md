# Wound-Only Segmentation Training Report

**Generated:** (Run `python train_model.py` to populate)

This report is auto-generated when wound-only baseline training completes. See `train_model.py` for the training script.

## Purpose of This Stage

Establish a clean baseline for wound-only segmentation using the standardized wound_focus_clean dataset. Single class: wound. No infection subclass segmentation.

## Dataset Files Used

- Train: `data/wound_focus_clean/train_wound_only.json`
- Val: `data/wound_focus_clean/val_wound_only.json`
- Test: `data/wound_focus_clean/test_wound_only.json`
- Root: `data/wound_focus_clean`

## Model/Config Used

- Model: Mask R-CNN ResNet-50-FPN
- num_classes: 2 (background + wound)
- batch_size: 2
- epochs: 50
- lr: 0.001
- image_size: (512, 512)
- use_medical_augmentation: True

## Train/Val/Test Sizes

- Train: 257
- Val: 57
- Test: 55

## Training Behavior Summary

(Run training to populate)

## Best Metrics Achieved (Validation)

(Run training to populate)

## Test Set Metrics

(Run training to populate)

## Comparison with Previous Multi-Class Attempt

Previous multi-class (8 classes) had near-zero segm_AP for subclasses due to annotation quality. This wound-only baseline focuses on the single well-annotated class (wound).

## Qualitative Prediction Observations

See `results/predictions/` for example predictions.

## Issues Found

None noted.

## Recommended Next Step

1. If segm_AP50 improved vs multi-class: proceed with wound-only + infection classification pipeline.
2. If still weak: consider data augmentation, longer training, or architecture tuning.
3. Add infected vs non-infected image-level classification using labels_infection.json.
