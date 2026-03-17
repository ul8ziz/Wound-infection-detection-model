# Wound-Only Segmentation Training Report

**Generated:** 2026-03-17 09:57:54

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
combined_AP50: 0.4171
bbox_AP50: 0.5171
segm_AP50: 0.3170

Training time: 3206.89s

## Test Metrics

- bbox_AP: 0.1521
- bbox_AP50: 0.3981
- bbox_AP75: 0.0625
- segm_AP: 0.0575
- segm_AP50: 0.2170
- segm_AP75: 0.0076
- combined_AP50: 0.3076

See results/predictions/ for qualitative outputs.

