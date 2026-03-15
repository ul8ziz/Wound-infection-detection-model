# Wound-Only Segmentation Training Report

**Generated:** 2026-03-15 07:44:17

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

Best epoch: 13
combined_AP50: 0.4243
bbox_AP50: 0.5117
segm_AP50: 0.3369

Training time: 1831.99s

## Test Metrics

- bbox_AP: 0.1658
- bbox_AP50: 0.3987
- bbox_AP75: 0.0822
- segm_AP: 0.0639
- segm_AP50: 0.2399
- segm_AP75: 0.0032
- combined_AP50: 0.3193

See results/predictions/ for qualitative outputs.

