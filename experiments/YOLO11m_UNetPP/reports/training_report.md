# YOLO11m + U-Net++ Training Report

Generated: 2026-04-08 22:27:07

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
| yolo_conf_thresh | 0.15 |
| yolo_min_conf_inference | 0.001 |
| unet_mask_thresh | 0.25 |
| roi_padding | 0.1 |
| mask_upscale | linear_probs |
| bbox_selection_strategy | all_above_thresh |
| merge_iou_thresh | 0.5 |
| coco_bbox_mode | yolo_xyxy |
| min_mask_area | 100 |
| postprocess_preset | none |
| postprocess | [] |
| enable_tta | True |
| debug_save_intermediates | False |
| debug_output_dir | results/combined/debug |
| debug_max_images | 16 |
| balanced_score_weights | {'bbox_AP50': 0.25, 'segm_AP50': 0.2, 'combined_AP50': 0.2, 'bbox_AP75': 0.15, 'segm_AP75': 0.15, 'mean_dice': 0.05} |
| pixels_per_cm | 26.0 |
| marker_real_cm | 3.0 |
| num_qualitative_samples | 8 |


---

## YOLO11m-seg Results

*Not available — YOLO was not trained or evaluated.*


---

## U-Net++ Results

*Not available.*


---

## Combined Pipeline Results

| Metric | Value |
|--------|-------|
| mean_dice | 0.6494 |
| mean_iou | 0.5271 |
| mean_dice_conditional | 0.6868 |
| mean_iou_conditional | 0.5575 |
| n_images_total | 55 |
| n_images_evaluated | 52 |
| n_images_missed | 3 |
| n_predictions_saved | 8 |
| coco_bbox_AP | 0.4301 |
| coco_bbox_AP50 | 0.7333 |
| coco_bbox_AP75 | 0.4567 |
| coco_segm_AP | 0.1744 |
| coco_segm_AP50 | 0.5279 |
| coco_segm_AP75 | 0.0578 |
| coco_combined_AP50 | 0.6306 |

---

## Infection Classification Results

*Not available.*
