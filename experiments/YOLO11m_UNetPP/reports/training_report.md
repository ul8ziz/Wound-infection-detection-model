# YOLO11m + U-Net++ Training Report

Generated: 2026-04-17 21:26:24

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
| epoch | 50.0000 |
| time | 5613.8500 |
| train/box_loss | 0.5776 |
| train/seg_loss | 1.2368 |
| train/cls_loss | 0.3384 |
| train/dfl_loss | 0.9105 |
| train/sem_loss | 0.0000 |
| metrics/precision(B) | 0.9824 |
| metrics/recall(B) | 0.9294 |
| metrics/mAP50(B) | 0.9522 |
| metrics/mAP50-95(B) | 0.7409 |
| metrics/precision(M) | 0.8980 |
| metrics/recall(M) | 0.8471 |
| metrics/mAP50(M) | 0.8426 |
| metrics/mAP50-95(M) | 0.4140 |
| val/box_loss | 0.8468 |
| val/seg_loss | 2.4340 |
| val/cls_loss | 1.5366 |
| val/dfl_loss | 1.0554 |
| val/sem_loss | 0.0000 |
| lr/pg0 | 0.0001 |
| lr/pg1 | 0.0001 |
| lr/pg2 | 0.0001 |
| training_completed | True |
| bbox_mAP50 | 0.8620 |
| bbox_mAP50_95 | 0.6249 |
| segm_mAP50 | 0.6751 |
| segm_mAP50_95 | 0.3197 |
| combined_AP50 | 0.7685 |

---

## U-Net++ Results

- **Best Dice (val):** 0.8270 at epoch 28
- **Training time:** 4032s

### Test Metrics

| Metric | Value |
|--------|-------|
| dice | 0.7989 |
| iou | 0.6867 |
| pixel_accuracy | 0.9183 |

---

## Combined Pipeline Results

| Metric | Value |
|--------|-------|
| mean_dice | 0.7647 |
| mean_iou | 0.6604 |
| mean_dice_conditional | 0.7647 |
| mean_iou_conditional | 0.6604 |
| n_images_total | 55 |
| n_images_evaluated | 55 |
| n_images_missed | 0 |
| n_predictions_saved | 8 |
| coco_bbox_AP | 0.5797 |
| coco_bbox_AP50 | 0.7913 |
| coco_bbox_AP75 | 0.6502 |
| coco_segm_AP | 0.3214 |
| coco_segm_AP50 | 0.6513 |
| coco_segm_AP75 | 0.3073 |
| coco_combined_AP50 | 0.7213 |
| coco_combined_AP75 | 0.4788 |

---

## Infection Classification Results

*Not available.*
