# Postoperative Wound Detection & Segmentation — Progress Report

---

## 0. Introduction

This project develops a deep-learning pipeline for detecting and segmenting postoperative wounds from clinical photographs, as part of a Master's thesis on wound infection analysis.

**Two main experiments** were conducted on a curated wound-only dataset (380 images, 532 wound annotations):

| Experiment | Architecture | Approach |
|------------|-------------|----------|
| **Experiment 1** | Mask R-CNN (ResNet-50-FPN) | Single-stage instance segmentation |
| **Experiment 2** | YOLO11m-seg + U-Net++ | Two-stage: detection then ROI refinement |

**Best overall results (test set):**

| Metric | Mask R-CNN | YOLO11m-seg (standalone) | Combined pipeline (tuned) |
|--------|----------:|-------------------------:|-------------------------:|
| bbox AP50 | 0.398 | **0.817** | 0.750 |
| bbox AP75 | 0.063 | — | **0.522** |
| segm AP50 | 0.217 | **0.662** | 0.561 |
| Dice | — | — | 0.670 |

The YOLO11m-seg + U-Net++ pipeline outperformed Mask R-CNN on every metric. The combined pipeline's optimization cycle improved bbox precision at IoU 0.75 by **42x** (from 0.012 to 0.522).

---

## 1. Problems Encountered and Solutions

### 1.1 Dataset Problems

| Problem | Impact | Solution |
|---------|--------|----------|
| **Inconsistent filenames** across 241 CVAT task folders | Could not merge data reliably | Built a standardization pipeline: `task_{id}_img_{global}_{label}.jpg` with full mapping CSV |
| **No explicit infection labels** | Needed for classification task | Inferred from filename heuristic (`-not-` = not infected); excluded 150 ambiguous cases |
| **Weak annotation quality** for fine-grained classes (edema, necrosis, fibrin, etc.) | Multi-class segmentation was unreliable | Narrowed scope to **wound-only** segmentation — the only class with consistent annotations |
| **Small dataset** (369 wound images) | Risk of overfitting | Applied offline 4x augmentation (→ ~1028 training images) + online medical augmentation |

### 1.2 Technical Problems

| Problem | Impact | Solution |
|---------|--------|----------|
| **ROI padding mismatch** — U-Net++ trained with `roi_padding=0.12` but inference used 0.10 | Distribution shift: degraded mask quality at inference | Aligned to 0.12 everywhere |
| **Bounding box mode** — `padded_roi` generated boxes from the padded ROI crop, not the actual mask | Extremely loose bboxes → `bbox_AP75 = 0.012` | Switched to `mask_tight` mode: bbox derived from predicted mask contour |
| **Mask holes and noise** in U-Net++ output | Lower Dice/IoU and poor visual quality | Added `close_fill` morphological postprocessing |
| **Missed detections at high confidence thresholds** | Some wounds scored 0.2–0.5, missed by default YOLO conf=0.5 | Combined pipeline uses two-stage filtering: YOLO at conf=0.001, then Python filter at 0.20 |
| **Evaluation excluded missed images** — pre-optimization Dice was computed on 51/55 images only | Metrics were misleadingly high | Fixed evaluation to include all 55 images (missed = Dice 0) |

### 1.3 Impact Summary

After solving these problems, the combined pipeline achieved:
- `bbox_AP75`: 0.012 → **0.522** (42x improvement)
- `segm_AP75`: 0.042 → **0.099** (2.3x improvement)
- Missed images: 4 → **1**

---

## 2. Technologies Used

### 2.1 Why These Architectures

| Technology | Role | Why chosen |
|------------|------|------------|
| **Mask R-CNN** (ResNet-50-FPN) | Experiment 1: baseline | Industry-standard instance segmentation model; well-documented; establishes a reliable comparison baseline |
| **YOLO11m-seg** | Experiment 2, Stage 1: detection + coarse segmentation | State-of-the-art real-time detector with built-in segmentation head; significantly faster and more accurate than Mask R-CNN for wound localization |
| **U-Net++** (EfficientNet-B1) | Experiment 2, Stage 2: mask refinement | U-shaped architecture excels at fine boundary segmentation; operates on cropped ROI for higher effective resolution; EfficientNet encoder provides strong features with moderate size |

### 2.2 Why a Two-Stage Pipeline

A single model must balance between localizing the wound in the full image and producing precise pixel-level boundaries. The two-stage design separates these concerns:

1. **YOLO11m-seg** processes the full image (768px) to detect wound regions — optimized for localization.
2. **U-Net++** processes only the cropped wound ROI (384×384) — optimized for boundary precision.

This division allows each model to focus on its strength.

### 2.3 Supporting Technologies

| Technology | Purpose |
|------------|---------|
| **PyTorch** | Deep learning framework |
| **Ultralytics** | YOLO11m-seg training and inference |
| **Segmentation Models PyTorch (SMP)** | U-Net++ implementation |
| **Albumentations** | Medical-safe augmentation (avoids destroying marker geometry) |
| **COCO API (pycocotools)** | Standardized AP evaluation |
| **Focal-Dice Loss** | Handles class imbalance (wound vs background) better than BCE alone |
| **CosineAnnealingLR** | Smooth learning rate decay for U-Net++ training |
| **Morphological postprocessing** | Fills holes, smooths mask boundaries |

---

## 3. Comparison Between Experiments

### 3.1 Configuration Comparison

| Parameter | Mask R-CNN | YOLO11m-seg + U-Net++ |
|-----------|----------:|----------------------:|
| Detection model | Mask R-CNN (ResNet-50-FPN) | YOLO11m-seg |
| Refinement model | — | U-Net++ (EfficientNet-B1) |
| Input size (detection) | 512 × 512 | 768 |
| Input size (refinement) | — | 384 × 384 |
| Batch size | 2 | 4 (YOLO) / 8 (U-Net++) |
| Epochs | 50 | 60 (YOLO) / 50 (U-Net++) |
| Optimizer | SGD (lr=0.001) | SGD (YOLO) / AdamW (U-Net++) |
| Loss | default | default (YOLO) / focal-dice (U-Net++) |
| Training time | ~53 min | ~67 min + ~28 min = ~95 min |
| Total parameters | ~44M | ~20M + ~8M = ~28M |

### 3.2 Test Metrics Comparison

| Metric | Mask R-CNN | YOLO standalone | Combined (tuned) | Winner |
|--------|----------:|-----------:|-----------:|--------|
| bbox AP | 0.152 | — | **0.481** | Combined |
| bbox AP50 | 0.398 | **0.817** | 0.750 | YOLO |
| bbox AP75 | 0.063 | — | **0.522** | Combined |
| segm AP | 0.058 | — | **0.198** | Combined |
| segm AP50 | 0.217 | **0.662** | 0.561 | YOLO |
| segm AP75 | 0.008 | — | **0.099** | Combined |
| combined AP50 | 0.308 | 0.739 | **0.656** | YOLO |
| Dice | — | — | **0.670** | Combined |
| IoU | — | — | **0.549** | Combined |

### 3.3 Key Findings

1. **YOLO11m-seg more than doubled Mask R-CNN's detection performance** — bbox AP50 jumped from 0.398 to 0.817 (+105%).
2. **Segmentation tripled** — segm AP50 from 0.217 to 0.662 (+205%).
3. **The combined pipeline added precision at high IoU** — bbox AP75 of 0.522 is the strongest metric in the project (Mask R-CNN achieved only 0.063).
4. **Mask R-CNN's main weakness:** low-resolution input (512px) and no dedicated refinement stage led to poor boundary quality (`segm_AP75 = 0.008`).
5. **Combined pipeline's remaining weakness:** `segm_AP75 = 0.099` is still low, primarily caused by ROI-to-mask alignment errors in 33% of test images.

---

## 4. Detailed Experiment Results

### 4.1 Experiment 1: Mask R-CNN

**Architecture:** Mask R-CNN, ResNet-50-FPN, `num_classes=2` (background + wound).
**Input:** 512 × 512, batch size 2, 50 epochs, SGD lr=0.001, moderate medical augmentation.
**Best validation epoch:** 13 (`combined_AP50 = 0.417`).

**Test results:**

| Metric | Value |
|------|------:|
| bbox_AP50 | 0.398 |
| bbox_AP75 | 0.063 |
| segm_AP50 | 0.217 |
| segm_AP75 | 0.008 |
| combined_AP50 | 0.308 |
| Training time | 3207 s (~53 min) |

This model served as a functional baseline. It proved the dataset and pipeline work end-to-end, but its low segmentation metrics confirmed that a stronger architecture was needed.

### 4.2 Experiment 2: YOLO11m-seg + U-Net++

#### Stage 1 — YOLO11m-seg (detection + coarse segmentation)

**Input:** 768px, batch 4, 60 epochs, SGD, 4x offline augmentation (~1028 images).

| Metric | Validation | Test |
|------|------:|------:|
| bbox precision | 0.897 | — |
| bbox recall | 0.820 | — |
| bbox mAP50 | 0.859 | **0.817** |
| bbox mAP50-95 | 0.516 | **0.539** |
| segm mAP50 | 0.718 | **0.662** |
| segm mAP50-95 | 0.260 | **0.250** |

#### Stage 2 — U-Net++ (ROI mask refinement)

**Input:** 384 × 384 ROI crops, EfficientNet-B1, AdamW, focal-dice loss, 50 epochs.

| Metric | Value |
|------|------:|
| Best val Dice | 0.776 (epoch 19) |
| Test Dice | **0.784** |
| Test IoU | **0.661** |
| Test pixel accuracy | **0.888** |
| Training time | 1657 s (~28 min) |

#### Combined Pipeline — Final Tuned Results

**Config:** `yolo_conf_thresh=0.20`, `unet_mask_thresh=0.35`, `roi_padding=0.12`, `close_fill` postprocessing, `mask_tight` bbox mode.

| Metric | Before optimization | After optimization | Improvement |
|------|------:|------:|------|
| coco_bbox_AP | 0.171 | **0.481** | ×2.8 |
| coco_bbox_AP50 | 0.598 | **0.750** | +25% |
| coco_bbox_AP75 | 0.012 | **0.522** | **×42** |
| coco_segm_AP | 0.189 | **0.198** | +5% |
| coco_segm_AP50 | 0.579 | **0.561** | −3% |
| coco_segm_AP75 | 0.042 | **0.099** | ×2.3 |
| combined_AP50 | 0.589 | **0.656** | +11% |
| mean_dice | 0.708 | **0.670** | corrected eval |
| Images evaluated | 51/55 | **54/55** | +3 recovered |

### 4.3 Error Analysis (Test Split, Combined Pipeline)

| Error category | Count | % |
|------|------:|------:|
| ok_or_minor | 27 | 49% |
| shifted_roi_or_mask | 18 | 33% |
| over_segmentation | 9 | 16% |
| fragmented_mask | 8 | 15% |
| poor_bbox_localization | 8 | 15% |
| boundary_or_alignment | 6 | 11% |
| missed_detection | 3 | 5% |

---

## 5. Training Plots and Figures

### 5.1 Combined Training Overview

**Figure 1 — YOLO + U-Net++ training curves (4-panel).**
Top: YOLO mAP curves and loss over 60 epochs. Bottom: U-Net++ loss and Dice/IoU over training.

![Combined training overview.](../experiments/YOLO11m_UNetPP/results/training_curves_combined.png)

### 5.2 YOLO11m-seg Metrics

**Figure 2 — YOLO precision, recall, validation losses, and mAP.**
Shows box and mask precision/recall converging, validation losses plateauing, and mAP50 peaking around epoch 50–60.

![YOLO detailed metrics.](../experiments/YOLO11m_UNetPP/results/yolo/yolo_metrics_combined.png)

**Figure 3 — YOLO confusion matrix (normalized).**
Shows wound class detection accuracy versus background false positives.

![YOLO confusion matrix.](../experiments/YOLO11m_UNetPP/results/yolo/confusion_matrix_normalized.png)

**Figure 4 — YOLO results strip (Ultralytics).**
Standard Ultralytics summary of all training metrics in a single plot.

![YOLO results strip.](../experiments/YOLO11m_UNetPP/results/yolo/results.png)

### 5.3 U-Net++ Metrics

**Figure 5 — U-Net++ loss, Dice, IoU curves.**
Shows train/val loss convergence, validation Dice peaking at 0.776 (epoch 19), and IoU tracking.

![U-Net++ metrics.](../experiments/YOLO11m_UNetPP/results/unet/unet_metrics_combined.png)

### 5.4 Optimization Impact

**Figure 6 — Combined pipeline: before vs after optimization.**
Bar chart showing the dramatic improvement in bbox AP75 (0.012 → 0.522) after fixing ROI padding, bbox mode, and postprocessing.

![Optimization comparison.](../experiments/YOLO11m_UNetPP/results/combined/optimization_comparison.png)

### 5.5 Mask R-CNN Baseline Curves

**Figure 7 — Mask R-CNN training and validation loss.**

![Mask R-CNN loss curves.](../experiments/maskrcnn/results/training_curves.png)

**Figure 8 — Mask R-CNN AP curves.**

![Mask R-CNN AP.](../experiments/maskrcnn/results/ap_curves.png)

### 5.6 YOLO Precision-Recall and F1 Curves

**Figure 9 — Box Precision-Recall curve.**

![Box PR curve.](../experiments/YOLO11m_UNetPP/results/yolo/BoxPR_curve.png)

**Figure 10 — Mask Precision-Recall curve.**

![Mask PR curve.](../experiments/YOLO11m_UNetPP/results/yolo/MaskPR_curve.png)

**Figure 11 — Box F1 vs confidence threshold.**

![Box F1 curve.](../experiments/YOLO11m_UNetPP/results/yolo/BoxF1_curve.png)

**Figure 12 — Mask F1 vs confidence threshold.**

![Mask F1 curve.](../experiments/YOLO11m_UNetPP/results/yolo/MaskF1_curve.png)

### 5.7 Qualitative Predictions

**Figure 13 — YOLO prediction (infected wound).**

![YOLO prediction.](../experiments/YOLO11m_UNetPP/results/yolo/predictions/pred_task_145_img_000151_infected.png)

**Figure 14 — YOLO prediction (not infected).**

![YOLO prediction.](../experiments/YOLO11m_UNetPP/results/yolo/predictions/pred_task_115_img_000041_not_infected.png)

**Figure 15 — Combined pipeline prediction (infected).**

![Combined prediction.](../experiments/YOLO11m_UNetPP/results/combined/predictions/combined_task_145_img_000151_infected.png)

**Figure 16 — Combined pipeline prediction (not infected).**

![Combined prediction.](../experiments/YOLO11m_UNetPP/results/combined/predictions/combined_task_115_img_000041_not_infected.png)

**Figure 17 — Mask R-CNN prediction (infected).**

![Mask R-CNN prediction.](../experiments/maskrcnn/results/predictions/pred_task_163_img_000186_infected_conf_0.90.png)

**Figure 18 — Mask R-CNN prediction (not infected).**

![Mask R-CNN prediction.](../experiments/maskrcnn/results/predictions/pred_task_223_img_000317_not_infected_conf_0.90.png)

### 5.8 Error Analysis

Error analysis was conducted on all 55 test images. Each image was categorized by dominant failure mode. Qualitative error analysis reports are available in `experiments/YOLO11m_UNetPP/results/combined/error_analysis/error_report_test.md`. The main finding is that **shifted ROI or mask** accounts for 33% of test images, making it the primary target for future improvement through better ROI-to-mask coordinate alignment.
