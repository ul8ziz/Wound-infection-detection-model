# YOLO11m + U-Net++ Training Report

Generated: 2026-07-16 20:59:39

---

## Configuration

### YOLO

| Parameter | Value |
|-----------|-------|
| model | yolo11m-seg.pt |
| image_size | 1024 |
| batch_size | 2 |
| epochs | 80 |
| lr0 | 0.01 |
| lrf | 0.01 |
| optimizer | SGD |
| momentum | 0.937 |
| weight_decay | 0.0005 |
| patience | 20 |
| degrees | 7 |
| perspective | 0.0 |
| flipud | 0.3 |
| fliplr | 0.5 |
| mosaic | 0.3 |
| mixup | 0.0 |
| close_mosaic | 15 |
| hsv_h | 0.01 |
| hsv_s | 0.5 |
| hsv_v | 0.3 |
| dropout | 0.0 |
| label_smoothing | 0.0 |
| cos_lr | True |
| warmup_epochs | 3 |

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
| epochs | 80 |
| lr | 0.0001 |
| weight_decay | 0.0001 |
| optimizer | AdamW |
| scheduler | CosineAnnealingLR |
| scheduler_T_max | 75 |
| scheduler_eta_min | 1e-06 |
| early_stop_patience | 15 |
| loss_type | focal_dice_boundary |
| loss_bce_weight | 0.4 |
| loss_dice_weight | 0.6 |
| loss_boundary_weight | 0.2 |
| boundary_kernel_size | 5 |
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
| resume_checkpoint | None |
| freeze_encoder | False |

### COMBINED

| Parameter | Value |
|-----------|-------|
| yolo_conf_thresh | 0.25 |
| yolo_min_conf_inference | 0.001 |
| unet_mask_thresh | 0.4 |
| roi_padding | 0.2 |
| mask_upscale | linear_probs |
| bbox_selection_strategy | all_above_thresh |
| merge_iou_thresh | 0.5 |
| coco_bbox_mode | yolo_xyxy |
| min_mask_area | 0 |
| postprocess_preset | largest_then_fill |
| postprocess | [] |
| refinement_postprocess | none |
| enable_tta | True |
| multi_scale_refinement | True |
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
| bbox_mAP50 | 0.8989 |
| bbox_mAP50_95 | 0.6793 |
| segm_mAP50 | 0.8770 |
| segm_mAP50_95 | 0.5976 |
| combined_AP50 | 0.8879 |

---

## U-Net++ Results

- **Best Dice (val):** 0.6656 at epoch 5
- **Training time:** 24801s

### Test Metrics

| Metric | Value |
|--------|-------|
| dice | 0.7483 |
| iou | 0.6084 |
| pixel_accuracy | 0.8793 |

---

## Combined Pipeline Results

| Metric | Value |
|--------|-------|
| mean_dice | 0.5897 |
| mean_iou | 0.4420 |
| median_dice | 0.6616 |
| median_iou | 0.4943 |
| mean_dice_conditional | 0.5897 |
| mean_iou_conditional | 0.4420 |
| n_images_total | 57 |
| n_images_evaluated | 57 |
| n_images_missed | 0 |
| n_marker_detected | 51 |
| marker_detection_rate | 0.8947 |
| n_dice_below_0_5 | 11 |
| n_predictions_saved | 8 |
| coco_bbox_AP | 0.2168 |
| coco_bbox_AP50 | 0.3888 |
| coco_bbox_AP75 | 0.2146 |
| coco_segm_AP | 0.0891 |
| coco_segm_AP50 | 0.3313 |
| coco_segm_AP75 | 0.0164 |
| coco_combined_AP50 | 0.3600 |
| coco_combined_AP75 | 0.1155 |

---

## Infection Classification Results

| Metric | Value |
|--------|-------|
| canonical_seed | 43 |
| best_epoch | 35 |
| decision_threshold | 0.5400 |
| train_n_samples | 253 |
| train_n_infected | 124 |
| train_n_non_infected | 129 |
| train_accuracy | 0.6285 |
| train_precision | 0.6364 |
| train_recall | 0.5645 |
| train_specificity | 0.6899 |
| train_f1_score | 0.5983 |
| train_threshold | 0.5400 |
| train_tp | 70 |
| train_fp | 40 |
| train_fn | 54 |
| train_tn | 89 |
| val_n_samples | 59 |
| val_n_infected | 16 |
| val_n_non_infected | 43 |
| val_accuracy | 0.7966 |
| val_precision | 0.7500 |
| val_recall | 0.3750 |
| val_specificity | 0.9535 |
| val_f1_score | 0.5000 |
| val_threshold | 0.5400 |
| val_tp | 6 |
| val_fp | 2 |
| val_fn | 10 |
| val_tn | 41 |
| test_n_samples | 57 |
| test_n_infected | 13 |
| test_n_non_infected | 44 |
| test_accuracy | 0.6491 |
| test_precision | 0.3333 |
| test_recall | 0.5385 |
| test_specificity | 0.6818 |
| test_f1_score | 0.4118 |
| test_threshold | 0.5400 |
| test_tp | 7 |
| test_fp | 14 |
| test_fn | 6 |
| test_tn | 30 |