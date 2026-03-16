# Wound-Only Segmentation Training Report

**Generated:** 2026-03-16 06:43:48

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

Best epoch: 10
combined_AP50: 0.4319
bbox_AP50: 0.5153
segm_AP50: 0.3486

Training time: 2256.45s

## Test Metrics

- bbox_AP: 0.1515
- bbox_AP50: 0.4120
- bbox_AP75: 0.0605
- segm_AP: 0.0636
- segm_AP50: 0.2568
- segm_AP75: 0.0041
- combined_AP50: 0.3344

See results/predictions/ for qualitative outputs.

