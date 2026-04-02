# YOLO11m + U-Net++ Training Report

Generated: 2026-03-30 00:06:49

---

## Configuration

### YOLO

| Parameter | Value |
|-----------|-------|
| model | yolo11m-seg.pt |
| image_size | 640 |
| batch_size | 8 |
| epochs | 100 |
| lr0 | 0.01 |
| lrf | 0.01 |
| optimizer | SGD |
| momentum | 0.937 |
| weight_decay | 0.0005 |
| patience | 20 |
| degrees | 10 |
| perspective | 0.0 |
| flipud | 0.5 |
| fliplr | 0.5 |
| mosaic | 1.0 |
| mixup | 0.1 |
| hsv_h | 0.015 |
| hsv_s | 0.7 |
| hsv_v | 0.4 |

### UNET

| Parameter | Value |
|-----------|-------|
| encoder | efficientnet-b3 |
| encoder_weights | imagenet |
| input_size | [256, 256] |
| in_channels | 3 |
| classes | 1 |
| batch_size | 16 |
| epochs | 50 |
| lr | 0.0001 |
| weight_decay | 0.0001 |
| optimizer | AdamW |
| scheduler | CosineAnnealingLR |
| scheduler_T_max | 50 |
| scheduler_eta_min | 1e-06 |
| early_stop_patience | 10 |
| loss_bce_weight | 0.5 |
| loss_dice_weight | 0.5 |
| roi_padding | 0.1 |

### COMBINED

| Parameter | Value |
|-----------|-------|
| yolo_conf_thresh | 0.5 |
| unet_mask_thresh | 0.5 |
| roi_padding | 0.1 |
| pixels_per_cm | 26.0 |
| num_qualitative_samples | 8 |


---

## YOLO11m-seg Results

| Metric | Value |
|--------|-------|
| epoch | 75.0000 |
| time | 1444.2700 |
| train/box_loss | 1.1418 |
| train/seg_loss | 2.4741 |
| train/cls_loss | 0.6723 |
| train/dfl_loss | 1.0943 |
| train/sem_loss | 0.0000 |
| metrics/precision(B) | 0.9262 |
| metrics/recall(B) | 0.7765 |
| metrics/mAP50(B) | 0.8685 |
| metrics/mAP50-95(B) | 0.5193 |
| metrics/precision(M) | 0.8442 |
| metrics/recall(M) | 0.6941 |
| metrics/mAP50(M) | 0.7270 |
| metrics/mAP50-95(M) | 0.2148 |
| val/box_loss | 1.4622 |
| val/seg_loss | 2.8784 |
| val/cls_loss | 0.8736 |
| val/dfl_loss | 1.2819 |
| val/sem_loss | 0.0000 |
| lr/pg0 | 0.0027 |
| lr/pg1 | 0.0027 |
| lr/pg2 | 0.0027 |
| training_completed | True |
| bbox_mAP50 | 0.7636 |
| bbox_mAP50_95 | 0.4940 |
| segm_mAP50 | 0.6230 |
| segm_mAP50_95 | 0.2348 |
| combined_AP50 | 0.6933 |

---

## U-Net++ Results

- **Best Dice (val):** 0.7737 at epoch 13
- **Training time:** 496s

### Test Metrics

| Metric | Value |
|--------|-------|
| dice | 0.7679 |
| iou | 0.6371 |
| pixel_accuracy | 0.8751 |

---

## Combined Pipeline Results

| Metric | Value |
|--------|-------|
| mean_dice | 0.6800 |
| mean_iou | 0.5508 |
| n_images_evaluated | 50 |
| n_predictions_saved | 8 |