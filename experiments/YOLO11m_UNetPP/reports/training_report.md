# YOLO11m + U-Net++ Training Report

Generated: 2026-07-16 04:00:02

---

## Configuration

### YOLO

| Parameter | Value |
|-----------|-------|
| model | yolo11m-seg.pt |
| image_size | 768 |
| batch_size | 2 |
| epochs | 50 |
| lr0 | 0.005 |
| lrf | 0.01 |
| optimizer | SGD |
| momentum | 0.937 |
| weight_decay | 0.001 |
| patience | 12 |
| degrees | 7 |
| perspective | 0.0 |
| flipud | 0.3 |
| fliplr | 0.5 |
| mosaic | 0.5 |
| mixup | 0.0 |
| close_mosaic | 15 |
| hsv_h | 0.01 |
| hsv_s | 0.5 |
| hsv_v | 0.4 |
| dropout | 0.1 |
| label_smoothing | 0.1 |
| cos_lr | True |
| warmup_epochs | 5 |

### UNET

| Parameter | Value |
|-----------|-------|
| architecture | unetplusplus |
| encoder | efficientnet-b1 |
| encoder_weights | imagenet |
| input_size | [512, 512] |
| in_channels | 3 |
| classes | 1 |
| batch_size | 4 |
| epochs | 60 |
| lr | 3e-05 |
| weight_decay | 0.0005 |
| optimizer | AdamW |
| scheduler | CosineAnnealingLR |
| scheduler_T_max | 55 |
| scheduler_eta_min | 1e-06 |
| early_stop_patience | 10 |
| loss_type | focal_dice_boundary |
| loss_bce_weight | 0.4 |
| loss_dice_weight | 0.6 |
| loss_boundary_weight | 0.35 |
| boundary_kernel_size | 3 |
| focal_alpha | 0.25 |
| focal_gamma | 2.0 |
| roi_padding | 0.2 |
| roi_crop_mode | mixed |
| eval_roi_crop_mode | yolo_predicted |
| roi_mix_weights | {'gt': 0.45, 'jitter': 0.3, 'yolo_cached': 0.25} |
| roi_jitter | {'scale_min': 0.85, 'scale_max': 1.15, 'shift_frac': 0.1} |
| yolo_roi_cache_path | experiments/YOLO11m_UNetPP/results/roi_cache/train_yolo_rois.json |
| eval_yolo_roi_cache_path | experiments/YOLO11m_UNetPP/results/roi_cache/val_yolo_rois.json |
| test_yolo_roi_cache_path | experiments/YOLO11m_UNetPP/results/roi_cache/test_yolo_rois.json |
| yolo_match_iou_min | 0.05 |
| resume_checkpoint | experiments/YOLO11m_UNetPP/checkpoints/unet/best_phase7_finetune_roi20/best_model.pth |
| freeze_encoder | False |

### COMBINED

| Parameter | Value |
|-----------|-------|
| yolo_conf_thresh | 0.15 |
| yolo_min_conf_inference | 0.001 |
| unet_mask_thresh | 0.4 |
| roi_padding | 0.2 |
| mask_upscale | nearest_binary |
| bbox_selection_strategy | merge_overlapping |
| merge_iou_thresh | 0.3 |
| coco_bbox_mode | yolo_xyxy |
| min_mask_area | 0 |
| postprocess_preset | largest_then_fill |
| postprocess | [] |
| refinement_postprocess | boundary_refine |
| enable_tta | False |
| multi_scale_refinement | False |
| multi_scale_roi_paddings | [0.1, 0.2, 0.3] |
| multi_scale_fusion | mean |
| multi_scale_weights | [] |
| score_fusion_yolo_weight | 0.7 |
| debug_save_intermediates | False |
| debug_output_dir | results/combined/debug |
| debug_max_images | 16 |
| balanced_score_weights | {'bbox_AP50': 0.25, 'segm_AP50': 0.2, 'combined_AP50': 0.2, 'bbox_AP75': 0.15, 'segm_AP75': 0.15, 'mean_dice': 0.05} |
| marker_class_id | 1 |
| marker_real_cm | 3.0 |
| num_qualitative_samples | 8 |


---

## YOLO11m-seg Results

| Metric | Value |
|--------|-------|
| bbox_mAP50 | 0.4428 |
| bbox_mAP50_95 | 0.2008 |
| segm_mAP50 | 0.2825 |
| segm_mAP50_95 | 0.0712 |
| combined_AP50 | 0.3626 |

---

## U-Net++ Results

- **Best Dice (val):** 0.0000 at epoch 0
- **Training time:** 0s

---

## Combined Pipeline Results

| Metric | Value |
|--------|-------|
| mean_dice | 0.4824 |
| mean_iou | 0.3547 |
| mean_dice_conditional | 0.4824 |
| mean_iou_conditional | 0.3547 |
| n_images_total | 55 |
| n_images_evaluated | 55 |
| n_images_missed | 0 |
| n_predictions_saved | 8 |
| coco_bbox_AP | 0.1327 |
| coco_bbox_AP50 | 0.3084 |
| coco_bbox_AP75 | 0.1045 |
| coco_segm_AP | 0.0334 |
| coco_segm_AP50 | 0.1717 |
| coco_segm_AP75 | 0.0000 |
| coco_combined_AP50 | 0.2401 |
| coco_combined_AP75 | 0.0522 |

---

## Infection Classification Results

*Not available.*
