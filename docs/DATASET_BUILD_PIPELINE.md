# Dataset Build Pipeline Documentation

**Document Version:** 1.0  
**Date:** 2025-03-12  
**Related:** `scripts/build_wound_focus_dataset.py`, `scripts/build_wound_only_dataset.py`

---

## 1. Overview

The dataset build pipeline transforms raw CVAT task data into a standardized, wound-focused dataset ready for training. It consists of **two sequential stages**:

| Stage | Script | Purpose |
|-------|--------|---------|
| **1. Standardization** | `build_wound_focus_dataset.py` | Rename images, infer infection status, copy to `wound_focus_clean/images/` |
| **2. Wound-Only & Splits** | `build_wound_only_dataset.py` | Filter wound annotations, create infection labels, build train/val/test splits |

**Key principle:** Raw data in `data/original_data/` is never modified. All outputs go to `data/wound_focus_clean/`.

---

## 2. Pipeline Flow

```
data/original_data/
├── task_0/ ... task_240/     (raw images, manifest.jsonl)
├── annotations_cleaned.json  (full COCO, categories 1–8)
└── project.json

        │
        ▼  Stage 1: build_wound_focus_dataset.py
        │
data/wound_focus_clean/
├── images/                   (380 standardized .jpg)
├── mappings/
│   ├── image_mapping.json
│   ├── image_mapping.csv
│   ├── skipped_images.csv
│   └── ambiguous_cases.csv
└── reports/RENAMING_REPORT.md

        │
        ▼  Stage 2: build_wound_only_dataset.py
        │
data/wound_focus_clean/
├── annotations_wound_only.json
├── labels_infection.json
├── labels_infection.csv
├── train_images.txt, val_images.txt, test_images.txt
├── train_wound_only.json, val_wound_only.json, test_wound_only.json
├── mappings/original_to_standardized.json
└── reports/
    ├── dataset_build_report.md
    ├── validation_report.txt
    └── review_summary_for_chatgpt.md
```

---

## 3. Stage 1: Wound Focus Standardization

**Script:** `scripts/build_wound_focus_dataset.py`

### What It Does

1. Scans `data/original_data/task_*/data/` and reads `manifest.jsonl`
2. Infers infection status from manifest `name` (e.g. `-not-` → non-infected)
3. Assigns `global_id`, generates standardized filenames
4. Optionally copies valid images to `wound_focus_clean/images/`

### Command

```bash
cd scripts

# Mapping only (no copy)
python build_wound_focus_dataset.py --data-root ../data --output-dir ../data/wound_focus_clean

# With image copy
python build_wound_focus_dataset.py --data-root ../data --output-dir ../data/wound_focus_clean --copy
```

### Output Summary (Stage 1)

| Category | Count |
|----------|-------|
| Valid (mapped & copied) | 380 |
| Skipped | 1 |
| Ambiguous | 150 |
| Total processed | 531 |

**Filename format:** `task_{task_id:03d}_img_{global_id:06d}_{infection_label}.jpg`

**Full documentation:** See `docs/WOUND_FOCUS_DATASET_DOCUMENTATION.md`

---

## 4. Stage 2: Wound-Only Annotations and Splits

**Script:** `scripts/build_wound_only_dataset.py`

### Prerequisites

- Stage 1 must be run first (with `--copy`)
- `data/wound_focus_clean/images/` must contain 380 images
- `data/wound_focus_clean/mappings/image_mapping.json` must exist
- `data/original_data/annotations_cleaned.json` must exist

### What It Does

1. Loads `annotations_cleaned.json` and `image_mapping.json`
2. Filters annotations to **category 1 only** (ВсяРана / whole wound) → remapped to `wound`
3. Builds `annotations_wound_only.json` — only images with at least one valid wound polygon
4. Builds `labels_infection.json` — all 380 images with infection label from mapping
5. Creates train/val/test splits (70% / 15% / 15%, seed=42)
6. Writes wound-only COCO splits: `train_wound_only.json`, `val_wound_only.json`, `test_wound_only.json`
7. Validates and generates reports

### Command

```bash
cd scripts
python build_wound_only_dataset.py --data-root ../data
```

### Class Filtering

| Category | Original | Action |
|----------|----------|--------|
| 1 | ВсяРана (whole wound) | **Kept** → `wound` |
| 2–8 | Marker, edema, hyperemia, necrosis, granulation, fibrin, pus, suture | **Removed** |

### Output Summary (Stage 2)

| Metric | Count |
|--------|-------|
| Total standardized images | 380 |
| Images with wound annotations | 369 |
| Images skipped (no wound ann) | 11 |
| Wound annotations (total) | 532 |
| Infected | 158 |
| Non-infected | 222 |
| Train images | 266 |
| Val images | 57 |
| Test images | 57 |
| Train wound-only images | 257 |
| Val wound-only images | 57 |
| Test wound-only images | 55 |

### Consistency Check (Verified)

| Equation | Result |
|----------|--------|
| infected + non_infected = total standardized images | 158 + 222 = 380 ✓ |
| train + val + test = total standardized images | 266 + 57 + 57 = 380 ✓ |
| train_wound_only + val_wound_only + test_wound_only = images with wound ann | 257 + 57 + 55 = 369 ✓ |
| images with wound ann + images without wound ann = total standardized images | 369 + 11 = 380 ✓ |

---

## 5. Full Run (Both Stages)

```bash
cd scripts

# Stage 1: Standardize and copy images
python build_wound_focus_dataset.py --data-root ../data --output-dir ../data/wound_focus_clean --copy

# Stage 2: Build wound-only annotations and splits
python build_wound_only_dataset.py --data-root ../data
```

---

## 6. Output Files Reference

### Stage 1 Outputs

| File | Purpose |
|------|---------|
| `wound_focus_clean/images/*.jpg` | 380 standardized images |
| `mappings/image_mapping.json` | Full mapping (valid, skipped, ambiguous) |
| `mappings/image_mapping.csv` | Valid images only |
| `mappings/skipped_images.csv` | Skipped records |
| `mappings/ambiguous_cases.csv` | Ambiguous infection status |
| `reports/RENAMING_REPORT.md` | Stage 1 summary |

### Stage 2 Outputs

| File | Purpose |
|------|---------|
| `annotations_wound_only.json` | Wound-only COCO (369 images, 532 annotations) |
| `labels_infection.json` | Image filename → infection label (380 images) |
| `labels_infection.csv` | Same, CSV format |
| `train_images.txt` | Train split filenames |
| `val_images.txt` | Val split filenames |
| `test_images.txt` | Test split filenames |
| `train_wound_only.json` | Train COCO (wound-only) |
| `val_wound_only.json` | Val COCO (wound-only) |
| `test_wound_only.json` | Test COCO (wound-only) |
| `mappings/original_to_standardized.json` | Traceability mapping |
| `reports/dataset_build_report.md` | Stage 2 summary |
| `reports/validation_report.txt` | Validation results |
| `reports/review_summary_for_chatgpt.md` | Review-ready summary |

---

## 7. Usage After Build

### Wound-Only Segmentation Training

- **Annotation file:** `data/wound_focus_clean/train_wound_only.json`
- **Root:** `data/wound_focus_clean`
- **file_name format:** e.g. `images/task_105_img_000001_infected.jpg`

### Infected vs. Non-Infected Classification

- **Labels:** `data/wound_focus_clean/labels_infection.json`
- **Splits:** `train_images.txt`, `val_images.txt`, `test_images.txt`
- All 380 images have labels (including 11 without wound annotations)

---

## 8. Assumptions and Constraints

1. **Raw data immutable** — No modifications to `data/original_data/`
2. **Manifest authoritative** — Infection status inferred from manifest `name`
3. **Annotations source** — `annotations_cleaned.json` uses original paths (e.g. `task_105/data/...`)
4. **Split reproducibility** — Seed 42 for deterministic train/val/test
5. **Medical constraints** — No augmentations that distort marker geometry (3×3 cm reference)

---

## 9. Related Documentation

- `docs/WOUND_FOCUS_DATASET_DOCUMENTATION.md` — Detailed Stage 1 documentation
- `data/wound_focus_clean/reports/dataset_build_report.md` — Stage 2 build report
- `docs/DATA_AUGMENTATION_GUIDE.md` — Augmentation strategy

---

*End of documentation*
