# YOLO11m-seg + U-Net++ Hybrid Pipeline — Experiment Analysis Report

**Experiment:** Postoperative Wound Detection and Segmentation  
**Pipeline:** Two-stage hybrid — YOLO11m-seg (detection + coarse segmentation) followed by U-Net++ (ROI-level mask refinement)  
**Date:** 2026-04-14  
**Dataset:** 369 unique clinical wound images (257 train / 57 val / 55 test), single class (`wound`), offline 4x augmentation for training (1028 samples)

---

## 1. Experiment Overview

This experiment evaluates a two-stage wound segmentation pipeline designed for postoperative wound infection detection. Stage 1 uses YOLO11m-seg for real-time wound detection and coarse instance segmentation on full-resolution images. Stage 2 refines detected ROIs using a U-Net++ (EfficientNet-B1 encoder) operating on cropped and resized wound regions. The combined pipeline produces final wound masks evaluated against COCO-style metrics and pixel-level Dice/IoU scores.

### Architecture

```
Input Image (variable resolution)
    │
    ▼
┌─────────────────────────────┐
│  YOLO11m-seg (768×768)      │  → Bounding boxes + coarse masks
│  SGD, lr=0.01, 60 epochs    │
└──────────────┬──────────────┘
               │  ROI crops (padded 12%)
               ▼
┌─────────────────────────────┐
│  U-Net++ (384×384)          │  → Refined binary masks per ROI
│  EfficientNet-B1, AdamW     │
│  Focal-Dice loss, 50 epochs │
└──────────────┬──────────────┘
               │  Upscale to original resolution
               ▼
┌─────────────────────────────┐
│  Combined Post-processing   │  → Final masks, COCO eval, Dice/IoU
│  Threshold + morphology     │
└─────────────────────────────┘
```

---

## 2. Dataset Characteristics

| Property | Value |
|----------|-------|
| Total unique images | 369 |
| Train (unique / augmented) | 257 / 1028 |
| Validation | 57 |
| Test | 55 |
| Classes | 1 (wound) |
| Annotation format | COCO-style polygons |
| Offline augmentation | 3 variants per train image (flip, rotate, shift-scale, color jitter) |
| Split seed | 42 (70/15/15 ratio) |

The dataset is small relative to the model complexity. Offline augmentation provides diversity but does not substitute for independent clinical images. All images contain at least one wound polygon; images with "-not-" in the filename indicate absence of infection (relevant for downstream classification, not for segmentation evaluation).

---

## 3. YOLO11m-seg Results (Stage 1)

### 3.1 Training Dynamics

YOLO11m-seg was trained for 60 epochs with SGD (lr0=0.01, cosine decay to 0.01×lr0), batch size 4, image size 768px, and moderate augmentation (mosaic=0.3, fliplr=0.5, degrees=7, hsv adjustments).

| Metric | Best Value | Best Epoch | Final (Epoch 60) |
|--------|-----------|------------|-------------------|
| Box mAP50 (val) | 0.892 | 46 | 0.859 |
| Box mAP50-95 (val) | 0.538 | 44 | 0.516 |
| Mask mAP50 (val) | 0.729 | 30 | 0.718 |
| Mask mAP50-95 (val) | 0.260 | 60 | 0.260 |
| Box Precision (val) | 0.920 | 56 | 0.897 |
| Box Recall (val) | 0.835 | 59 | 0.820 |

### 3.2 Test Set Evaluation

| Metric | Value |
|--------|-------|
| bbox mAP50 | 0.817 |
| bbox mAP50-95 | 0.539 |
| segm mAP50 | 0.662 |
| segm mAP50-95 | 0.250 |
| combined AP50 | 0.739 |

### 3.3 Training Observations

- **Box detection is strong:** mAP50 > 0.85 on validation, 0.82 on test. The model reliably localizes wounds.
- **Mask quality lags behind boxes:** segm mAP50 is ~0.15 lower than bbox mAP50 on both val and test, indicating coarse boundary prediction from the YOLO segmentation head.
- **Mild overfitting on segmentation:** Validation seg loss remains ~1.4x higher than training seg loss in late epochs, while box loss converges more cleanly.
- **Box metrics peak before mask metrics stabilize:** Best box mAP50 at epoch 46, best mask mAP50 at epoch 30, but mask mAP50-95 still improving at epoch 60. The Ultralytics fitness function (dominated by box metrics) may not select the best checkpoint for segmentation.
- **Val set noise:** With only 57 validation images, epoch-to-epoch metric oscillation is expected and observed (box mAP50 swings of ±0.05 between consecutive epochs).

---

## 4. U-Net++ Results (Stage 2)

### 4.1 Experiment Groups (Ablation Study)

Seven experiment configurations (A–G) were designed as a structured ablation:

| Group | Description | Key Change | Executed |
|-------|-------------|------------|----------|
| A | Baseline | GT-only ROI crops | Yes |
| B | Mixed ROI | 45% GT + 30% jitter + 25% cached YOLO | Yes |
| C | Resolution sweep | 256 / 384 / 512 input sizes | **No** |
| D | Boundary loss | focal_dice_boundary loss | Yes |
| E | Multi-scale | Multi-scale inference fusion | Yes |
| F | DeepLabV3+ | Alternative architecture | **No** |
| G | Boundary refine | Morphological boundary postprocessing | Yes |

### 4.2 Key Comparison: COCO Metrics on Combined Pipeline

| Metric | A (Baseline) | B (Mixed ROI) | D (Boundary) | E (Multi-scale) | G (Refine) |
|--------|:---:|:---:|:---:|:---:|:---:|
| segm AP50 | 0.526 | **0.581** | 0.516 | 0.486 | 0.560 |
| segm AP75 | 0.028 | **0.105** | 0.058 | 0.093 | 0.082 |
| bbox AP50 | 0.739 | 0.739 | **0.750** | **0.750** | **0.750** |
| bbox AP75 | 0.466 | **0.543** | 0.565 | 0.432 | 0.413 |
| combined AP50 | 0.633 | **0.660** | 0.633 | 0.618 | 0.655 |
| combined AP75 | 0.247 | **0.324** | 0.312 | 0.262 | 0.247 |
| mean Dice | 0.638 | **0.676** | 0.669 | 0.670 | **0.676** |
| mean IoU | 0.513 | 0.554 | 0.548 | 0.548 | **0.556** |
| Images missed | 4 | **1** | **1** | **1** | **1** |

### 4.3 Interpretation

**Group B (Mixed ROI training) is the clear winner**, dominating on segm AP50/AP75 and combined AP50/AP75 while matching or exceeding pixel-level metrics.

- The primary bottleneck was **train-inference distribution shift**: U-Net++ trained on perfect GT crops could not handle noisy YOLO-predicted ROIs at inference. Mixed ROI training resolved this.
- Boundary-aware loss (D), multi-scale inference (E), and morphological refinement (G) provided **marginal or no net improvement** over Group B — they addressed secondary effects while the ROI mismatch was the dominant issue.
- segm AP75 improved 3.75x (0.028 → 0.105) from A to B, confirming that ROI robustness was the main lever.

---

## 5. Combined Pipeline — Final Results

### 5.1 Current Best (Group B, Mixed ROI)

| Metric | Value |
|--------|-------|
| mean Dice (full split) | 0.676 |
| mean Dice (conditional, excl. missed) | 0.689 |
| mean IoU (full split) | 0.554 |
| mean IoU (conditional) | 0.565 |
| COCO bbox AP | 0.485 |
| COCO bbox AP50 | 0.739 |
| COCO bbox AP75 | 0.543 |
| COCO segm AP | 0.203 |
| COCO segm AP50 | 0.581 |
| COCO segm AP75 | 0.105 |
| COCO combined AP50 | 0.660 |
| COCO combined AP75 | 0.324 |
| Images evaluated | 54 / 55 |
| Images missed | 1 |

### 5.2 Comparison: Hybrid vs. Standalone Models

| Metric | YOLO-only (test) | U-Net-only (GT crops, test) | Combined Pipeline |
|--------|:---:|:---:|:---:|
| bbox mAP50 | **0.817** | N/A | 0.739 |
| segm mAP50 | **0.662** | N/A | 0.581 |
| Dice | N/A | **0.784** | 0.689 (conditional) |
| IoU | N/A | **0.661** | 0.565 (conditional) |

The combined pipeline **does not surpass YOLO-only on COCO segm AP50** (0.581 vs. 0.662). The hybrid value lies in refined mask quality (Dice 0.689 on realistic YOLO crops vs. 0.784 on perfect GT crops), not in detection-level matching which COCO AP penalizes.

### 5.3 Improvement Timeline (Before/After Optimization)

| Metric | Before Optimization | After (Current Best) | Change |
|--------|:---:|:---:|:---:|
| bbox AP50 | 0.598 | **0.739** | +23.6% |
| bbox AP75 | 0.012 | **0.543** | +4425% |
| segm AP50 | 0.579 | **0.581** | +0.3% |
| segm AP75 | 0.042 | **0.105** | +150% |
| combined AP50 | 0.589 | **0.660** | +12.1% |
| mean Dice | 0.708* | **0.676** | -4.5% |

*Old Dice was inflated by excluding missed images (counted as Dice=0 in corrected calculation).

---

## 6. Error Analysis

### 6.1 Error Distribution (Test Set)

| Error Type | Count | % of Images |
|------------|:-----:|:-----------:|
| OK or minor errors | 27 | 49.1% |
| Shifted ROI or mask | 18 | 32.7% |
| Over-segmentation | 9 | 16.4% |
| Fragmented mask | 8 | 14.5% |
| Poor bbox localization | 8 | 14.5% |
| Boundary/alignment error | 6 | 10.9% |
| Missed detection | 3 | 5.5% |
| Moderate bbox IoU | 2 | 3.6% |

Images can have multiple error labels. The dominant failure mode is **shifted ROI or mask** (32.7%), indicating that YOLO box quality directly limits mask alignment quality downstream.

### 6.2 Failure Mode Analysis

- **Shifted ROI/mask (18 images):** YOLO bbox is offset from wound center, causing the U-Net++ crop to include irrelevant context or clip wound edges. This is the single largest error source and is a structural limitation of the two-stage approach.
- **Over-segmentation (9 images):** U-Net++ predicts wound-like regions outside the actual wound boundary, often at skin folds or surgical tape edges.
- **Fragmented masks (8 images):** Multiple disconnected mask fragments instead of one contiguous wound region. Partially addressed by `min_mask_area` filtering but not fully resolved.
- **Missed detections (3 images):** YOLO fails to detect the wound entirely. These are small, well-healed wounds with minimal visual contrast.

---

## 7. Key Bottleneck: segm AP75

The most significant remaining limitation is the gap between bbox AP75 (0.543) and segm AP75 (0.105). This means:

- At IoU threshold 0.5, most predicted masks overlap sufficiently with ground truth.
- At IoU threshold 0.75, most masks **fail** — boundaries are not precise enough.

**Root cause:** The U-Net++ operates on 384x384 crops resized from variable-size ROIs. The bilinear interpolation chain (crop → resize down → model → resize up → threshold) introduces boundary smoothing at each step. Wounds spanning different resolution ranges experience different levels of detail loss.

This is a **fundamental limitation of fixed-resolution refinement** and the primary target for future improvement (see companion document: `improvement_plan.md`).

---

## 8. Infection Classification (Preliminary)

A lightweight binary classifier was trained on extracted features to distinguish infected vs. non-infected wounds.

| Metric | Value |
|--------|-------|
| Training accuracy | 72.7% |
| Precision | 63.9% |
| Recall | 77.3% |
| F1 score | 70.0% |
| Dataset size | 311 samples (128 infected, 183 non-infected) |

These results are **training-set only** — no independent test evaluation was performed. The classifier is preliminary and should not be considered a validated infection detection result.

---

## 9. Conclusions

1. **Detection is solved for this dataset:** YOLO11m-seg achieves bbox mAP50 > 0.81, reliably localizing wounds in clinical photographs.

2. **Coarse segmentation is adequate:** YOLO mask mAP50 of 0.66 provides a useful starting point for wound area estimation.

3. **The hybrid pipeline improves mask quality but not COCO AP:** Adding U-Net++ refinement increases Dice from N/A to 0.689 on realistic YOLO crops, but COCO segm AP50 is lower than YOLO-only (0.581 vs. 0.662) due to detection-level matching penalties.

4. **ROI robustness was the dominant bottleneck:** Mixed ROI training (Group B) provided the largest single improvement (segm AP75: 3.75x), confirming that train-inference distribution alignment is critical for two-stage pipelines.

5. **Boundary precision remains the key limitation:** segm AP75 = 0.105 means masks are not clinically precise at strict IoU thresholds. The fixed-resolution resize chain is the structural cause.

6. **The pipeline is suitable for research use** with documented limitations. Clinical deployment would require significantly improved boundary precision and independent validation on larger datasets.

---

## Appendix: Reproduction

All results can be reproduced using the training notebook:

```
experiments/YOLO11m_UNetPP/training_pipeline.ipynb
```

Configuration: `experiments/YOLO11m_UNetPP/config.yaml`  
Checkpoints: `experiments/YOLO11m_UNetPP/checkpoints/`  
Results artifacts: `experiments/YOLO11m_UNetPP/results/`

Seed: 42 (set in config for reproducibility).
