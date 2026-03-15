# Wound-Only Segmentation Training Report

**Generated:** 2026-03-15 18:06:41

## Purpose

Single class: wound. Dataset: wound_focus_clean.

## Dataset

Train: E:\GitHub\Wound-infection-detection-model\data\wound_focus_clean\train_wound_only.json
Val: E:\GitHub\Wound-infection-detection-model\data\wound_focus_clean\val_wound_only.json
Test: E:\GitHub\Wound-infection-detection-model\data\wound_focus_clean\test_wound_only.json

## Model

Mask R-CNN ResNet-50-FPN, num_classes=2, image_size=(512, 512)

## Sizes

Train: 257, Val: 57, Test: 55

## Best Metrics

Best epoch: 11
combined_AP50: 0.4437
bbox_AP50: 0.5252
segm_AP50: 0.3623

Training time: 1711.66s

## Test Metrics

- bbox_AP: 0.1496
- bbox_AP50: 0.3996
- bbox_AP75: 0.0703
- segm_AP: 0.0651
- segm_AP50: 0.2358
- segm_AP75: 0.0160
- combined_AP50: 0.3177

See results/predictions/ for qualitative outputs.

