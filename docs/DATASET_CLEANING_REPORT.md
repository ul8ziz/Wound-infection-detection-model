# Dataset Cleaning Report

## Overview

This report documents the dataset cleaning pipeline implemented for the Wound Infection Detection project. The pipeline fixes noisy polygons, invalid masks, class mismatches, and other annotation issues that cause near-zero AP in Mask R-CNN training.

---

## What Was Broken

| Issue | Description |
|-------|-------------|
| **Extra classes** | Raw annotations contained 15 classes; only 8 target classes are used for training |
| **Non-target annotations** | 4,724 annotations from classes not in the training set (e.g. Зона шва, Металлоконструкция, Вторичная пигментация) |
| **Outside-image polygons** | 278 annotations with coordinates outside image bounds that could not be repaired |
| **Noisy polygons** | Some polygons had 100+ points without simplification |
| **No validation** | No checks for empty segmentations, zero-area masks, or invalid bboxes |

---

## What Was Cleaned

| Action | Result |
|--------|--------|
| **Class filtering** | Kept only 8 target classes; removed all others |
| **Class remapping** | Remapped to contiguous IDs 1..8 |
| **Polygon simplification** | Douglas-Peucker simplification when polygon has > 500 points |
| **Invalid removal** | Dropped empty segmentation, zero-area masks, invalid bbox, outside-image |
| **Bbox/area recompute** | Recomputed from cleaned polygons using `cv2.contourArea` |

---

## Target Classes (8)

1. ВсяРана (Whole Wound)
2. Метка для размерности (Size Marker)
3. Зона отека вокруг раны (Edema Zone)
4. Зона гиперемии вокруг (Hyperemia Zone)
5. Зона некроза (Necrosis Zone)
6. Зона грануляций (Granulation Zone)
7. Фибрин (Fibrin)
8. Гнойное отделяемое (Pus)

---

## Statistics

| Metric | Before | After | Removed |
|--------|--------|-------|---------|
| Images | 530 | 530 | 0 |
| Annotations | 11,338 | 6,336 | 5,002 |
| Classes | 15 | 8 | 7 |

### Removal Breakdown

- Non-target classes: 4,724 annotations
- Outside image (unrepairable): 278 annotations
- Empty segmentation: 0
- Invalid polygon: 0
- Zero-area mask: 0
- Invalid bbox: 0
- Tiny object (area ratio): 0

---

## Files Created

| File | Purpose |
|------|---------|
| `scripts/clean_dataset.py` | Main cleaning script (CVAT/COCO input, filter, simplify, validate) |
| `scripts/validate_cleaned_dataset.py` | Validates cleaned annotations |
| `scripts/visualize_cleaned_dataset.py` | Visualizes samples for manual inspection |
| `data/annotations_cleaned.json` | Cleaned COCO annotations (output) |
| `data/cleaning_report.txt` | Text report (regenerated each run) |
| `data/cleaned_visualizations/` | Sample visualization images |

---

## Files Modified

| File | Change |
|------|--------|
| `experiments/maskrcnn/train_model.py` | Added `ann_file_cleaned`; uses it when splits are missing |
| `scripts/apply_augmentation_only.py` | Fallback to `annotations_cleaned.json` when splits missing |
| `README.md` | Documented dataset cleaning workflow |

---

## Usage

### Clean from CVAT tasks

```bash
cd scripts
python clean_dataset.py --input-mode cvat --data-root ../data
```

### Clean from existing COCO JSON

```bash
python clean_dataset.py --input-mode coco --input-file ../data/annotations.json
```

### Regenerate splits after cleaning

```bash
python clean_dataset.py --input-mode cvat --data-root ../data --split
```

### Validate and visualize

```bash
python validate_cleaned_dataset.py
python visualize_cleaned_dataset.py --num-samples 8
```

---

## Training on Cleaned Data

- **If splits exist** (from `clean_dataset.py --split`): Use `data/splits/train.json` (derived from cleaned data).
- **If splits are missing**: `train_model.py` automatically uses `data/annotations_cleaned.json`.

---

## Configurable Thresholds (in `clean_dataset.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_POLYGON_POINTS` | 3 | Minimum points for valid polygon |
| `MAX_POLYGON_POINTS` | 500 | Simplify if exceeded |
| `SIMPLIFY_EPSILON` | 0.5 | Douglas-Peucker tolerance |
| `MIN_MASK_AREA_PX` | 4 | Minimum mask area in pixels |
| `MIN_BBOX_SIDE_PX` | 2 | Minimum bbox width/height |
| `MIN_OBJECT_AREA_RATIO` | 1e-6 | Minimum object area / image area |

---

## Date

Report generated: March 2025
