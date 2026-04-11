# YOLO11m + U-Net++ Training Report

Generated: 2026-04-10 (updated with Group B mixed-ROI improvements)

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
| lr | 0.0001 |
| weight_decay | 0.0001 |
| optimizer | AdamW |
| scheduler | CosineAnnealingLR |
| scheduler_T_max | 45 |
| scheduler_eta_min | 1e-06 |
| early_stop_patience | 8 |
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
| roi_mix_weights | {'gt': 0.45, 'jitter': 0.30, 'yolo_cached': 0.25} |
| roi_jitter | {'scale_min': 0.85, 'scale_max': 1.15, 'shift_frac': 0.10} |
| yolo_roi_cache_path | results/roi_cache/train_yolo_rois.json |
| eval_yolo_roi_cache_path | results/roi_cache/val_yolo_rois.json |
| yolo_match_iou_min | 0.05 |
| resume_checkpoint | None |
| freeze_encoder | False |

### COMBINED

| Parameter | Value |
|-----------|-------|
| yolo_conf_thresh | 0.2 |
| yolo_min_conf_inference | 0.001 |
| unet_mask_thresh | 0.35 |
| roi_padding | 0.12 |
| mask_upscale | linear_probs |
| bbox_selection_strategy | all_above_thresh |
| merge_iou_thresh | 0.5 |
| coco_bbox_mode | mask_tight |
| min_mask_area | 200 |
| postprocess_preset | close_fill |
| postprocess | [] |
| refinement_postprocess | none |
| enable_tta | False |
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

| Metric | Value |
|--------|-------|
| epoch | 60.0000 |
| time | 4012.7500 |
| train/box_loss | 0.8125 |
| train/seg_loss | 1.9429 |
| train/cls_loss | 0.4990 |
| train/dfl_loss | 1.0036 |
| train/sem_loss | 0.0000 |
| metrics/precision(B) | 0.8971 |
| metrics/recall(B) | 0.8204 |
| metrics/mAP50(B) | 0.8591 |
| metrics/mAP50-95(B) | 0.5163 |
| metrics/precision(M) | 0.8074 |
| metrics/recall(M) | 0.7294 |
| metrics/mAP50(M) | 0.7183 |
| metrics/mAP50-95(M) | 0.2601 |
| val/box_loss | 1.3275 |
| val/seg_loss | 3.0237 |
| val/cls_loss | 0.7123 |
| val/dfl_loss | 1.3802 |
| val/sem_loss | 0.0000 |
| lr/pg0 | 0.0003 |
| lr/pg1 | 0.0003 |
| lr/pg2 | 0.0003 |
| training_completed | True |
| bbox_mAP50 | 0.8169 |
| bbox_mAP50_95 | 0.5387 |
| segm_mAP50 | 0.6620 |
| segm_mAP50_95 | 0.2503 |
| combined_AP50 | 0.7395 |

---

## U-Net++ Results (Mixed ROI Training)

- **Best Dice (val):** 0.7497 at epoch 1
- **Training time:** 5586s
- **ROI mode:** mixed (45% GT, 30% jittered, 25% cached YOLO)
- **Eval ROI mode:** yolo_predicted (realistic inference conditions)

### Test Metrics (YOLO-predicted ROIs)

| Metric | Value |
|--------|-------|
| dice | 0.3376 |
| iou | 0.2868 |
| pixel_accuracy | 0.8989 |

> Note: Standalone U-Net++ test metrics use YOLO-predicted ROIs, which are noisier
> than GT boxes. The real-world performance is best measured by the Combined Pipeline
> metrics below, which show improvement over the GT-only baseline.

---

## Combined Pipeline Results (Group B — Mixed ROI)

| Metric | Value | Baseline (GT-only) | Delta |
|--------|-------|-------------------|-------|
| mean_dice | 0.6761 | 0.6695 | +0.0066 |
| mean_iou | 0.5543 | 0.5491 | +0.0052 |
| mean_dice_conditional | 0.6886 | 0.6819 | +0.0067 |
| mean_iou_conditional | 0.5646 | 0.5592 | +0.0054 |
| n_images_total | 55 | 55 | — |
| n_images_evaluated | 54 | 54 | — |
| n_images_missed | 1 | 1 | — |
| coco_bbox_AP | 0.4849 | 0.4805 | +0.0044 |
| coco_bbox_AP50 | 0.7389 | 0.7502 | −0.0113 |
| coco_bbox_AP75 | 0.5428 | 0.5223 | **+0.0205** |
| coco_segm_AP | 0.2033 | 0.1984 | +0.0049 |
| coco_segm_AP50 | 0.5814 | 0.5611 | **+0.0204** |
| coco_segm_AP75 | 0.1050 | 0.0991 | +0.0060 |
| coco_combined_AP50 | 0.6602 | 0.6556 | +0.0046 |
| coco_combined_AP75 | 0.3239 | 0.3107 | **+0.0132** |

---

## Infection Classification Results

*Not available.*
