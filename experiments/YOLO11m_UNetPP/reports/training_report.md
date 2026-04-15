# YOLO11m + U-Net++ Training Report

Generated: 2026-04-15 04:06:04

---

## Configuration

### YOLO

| Parameter | Value |
|-----------|-------|
| model | yolo11m-seg.pt |
| image_size | 768 |
| batch_size | 4 |
| epochs | 60 |
| lr0 | 0.01 |
| lrf | 0.01 |
| optimizer | SGD |
| momentum | 0.937 |
| weight_decay | 0.0005 |
| patience | 15 |
| degrees | 7 |
| perspective | 0.0 |
| flipud | 0.0 |
| fliplr | 0.5 |
| mosaic | 0.3 |
| mixup | 0.0 |
| close_mosaic | 15 |
| hsv_h | 0.01 |
| hsv_s | 0.5 |
| hsv_v | 0.3 |

### UNET

| Parameter | Value |
|-----------|-------|
| architecture | unetplusplus |
| encoder | efficientnet-b1 |
| encoder_weights | imagenet |
| input_size | [384, 384] |
| in_channels | 3 |
| classes | 1 |
| batch_size | 8 |
| epochs | 50 |
| lr | 5e-05 |
| weight_decay | 0.0001 |
| optimizer | AdamW |
| scheduler | CosineAnnealingLR |
| scheduler_T_max | 45 |
| scheduler_eta_min | 1e-06 |
| early_stop_patience | 12 |
| loss_type | focal_dice |
| loss_bce_weight | 0.4 |
| loss_dice_weight | 0.6 |
| loss_boundary_weight | 0.15 |
| boundary_kernel_size | 5 |
| focal_alpha | 0.25 |
| focal_gamma | 2.0 |
| roi_padding | 0.12 |
| roi_crop_mode | mixed |
| eval_roi_crop_mode | yolo_predicted |
| roi_mix_weights | {'gt': 0.45, 'jitter': 0.3, 'yolo_cached': 0.25} |
| roi_jitter | {'scale_min': 0.85, 'scale_max': 1.15, 'shift_frac': 0.1} |
| yolo_roi_cache_path | experiments/YOLO11m_UNetPP/results/roi_cache/train_yolo_rois.json |
| eval_yolo_roi_cache_path | experiments/YOLO11m_UNetPP/results/roi_cache/val_yolo_rois.json |
| test_yolo_roi_cache_path | experiments/YOLO11m_UNetPP/results/roi_cache/test_yolo_rois.json |
| yolo_match_iou_min | 0.05 |
| resume_checkpoint | None |
| freeze_encoder | False |

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
| refinement_postprocess | none |
| enable_tta | True |
| multi_scale_refinement | False |
| multi_scale_roi_paddings | [0.0, 0.12, 0.2] |
| multi_scale_fusion | mean |
| multi_scale_weights | [] |
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
| mean_dice | 0.6949 |
| mean_iou | 0.5670 |
| mean_dice_conditional | 0.7078 |
| mean_iou_conditional | 0.5775 |
| n_images_total | 55 |
| n_images_evaluated | 54 |
| n_images_missed | 1 |
| n_predictions_saved | 8 |
| coco_bbox_AP | 0.5061 |
| coco_bbox_AP50 | 0.7737 |
| coco_bbox_AP75 | 0.5903 |
| coco_segm_AP | 0.2214 |
| coco_segm_AP50 | 0.6448 |
| coco_segm_AP75 | 0.0936 |
| coco_combined_AP50 | 0.7093 |
| coco_combined_AP75 | 0.3419 |

---

## Infection Classification Results

*Not available.*
