# YOLO11m + U-Net++ Training Report

Generated: 2026-04-05 20:35:18

---

## Configuration

### YOLO

| Parameter | Value |
|-----------|-------|
| model | yolo11m-seg.pt |
| image_size | 1024 |
| batch_size | 4 |
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
| mosaic | 0.5 |
| mixup | 0.0 |
| close_mosaic | 15 |
| hsv_h | 0.015 |
| hsv_s | 0.7 |
| hsv_v | 0.4 |

### UNET

| Parameter | Value |
|-----------|-------|
| encoder | efficientnet-b1 |
| encoder_weights | imagenet |
| input_size | [256, 256] |
| in_channels | 3 |
| classes | 1 |
| batch_size | 16 |
| epochs | 35 |
| lr | 0.0001 |
| weight_decay | 0.0001 |
| optimizer | AdamW |
| scheduler | CosineAnnealingLR |
| scheduler_T_max | 35 |
| scheduler_eta_min | 1e-06 |
| early_stop_patience | 6 |
| loss_type | focal_dice |
| loss_bce_weight | 0.5 |
| loss_dice_weight | 0.5 |
| focal_alpha | 0.25 |
| focal_gamma | 2.0 |
| roi_padding | 0.1 |

### COMBINED

| Parameter | Value |
|-----------|-------|
| yolo_conf_thresh | 0.25 |
| unet_mask_thresh | 0.5 |
| roi_padding | 0.1 |
| pixels_per_cm | 26.0 |
| marker_real_cm | 3.0 |
| num_qualitative_samples | 8 |


---

## YOLO11m-seg Results

| Metric | Value |
|--------|-------|
| epoch | 56.0000 |
| time | 8043.9600 |
| train/box_loss | 1.2005 |
| train/seg_loss | 2.4889 |
| train/cls_loss | 0.7703 |
| train/dfl_loss | 1.2847 |
| train/sem_loss | 0.0000 |
| metrics/precision(B) | 0.8279 |
| metrics/recall(B) | 0.8118 |
| metrics/mAP50(B) | 0.8237 |
| metrics/mAP50-95(B) | 0.4912 |
| metrics/precision(M) | 0.7536 |
| metrics/recall(M) | 0.6353 |
| metrics/mAP50(M) | 0.6407 |
| metrics/mAP50-95(M) | 0.2289 |
| val/box_loss | 1.3983 |
| val/seg_loss | 2.8013 |
| val/cls_loss | 0.9343 |
| val/dfl_loss | 1.4564 |
| val/sem_loss | 0.0000 |
| lr/pg0 | 0.0046 |
| lr/pg1 | 0.0046 |
| lr/pg2 | 0.0046 |
| training_completed | True |
| bbox_mAP50 | 0.7858 |
| bbox_mAP50_95 | 0.4726 |
| segm_mAP50 | 0.6772 |
| segm_mAP50_95 | 0.2365 |
| combined_AP50 | 0.7315 |

---

## U-Net++ Results

- **Best Dice (val):** 0.7676 at epoch 10
- **Training time:** 734s

### Test Metrics

| Metric | Value |
|--------|-------|
| dice | 0.7817 |
| iou | 0.6552 |
| pixel_accuracy | 0.8775 |

---

## Combined Pipeline Results

| Metric | Value |
|--------|-------|
| mean_dice | 0.7076 |
| mean_iou | 0.5780 |
| n_images_evaluated | 51 |
| n_predictions_saved | 8 |
| coco_bbox_AP | 0.1707 |
| coco_bbox_AP50 | 0.5981 |
| coco_bbox_AP75 | 0.0124 |
| coco_segm_AP | 0.1888 |
| coco_segm_AP50 | 0.5794 |
| coco_segm_AP75 | 0.0422 |
| coco_combined_AP50 | 0.5888 |

---

## Infection Classification Results

*Not available.*
