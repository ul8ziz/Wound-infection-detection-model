# Wound Focus Clean Dataset - Documentation

**Document Version:** 1.0  
**Date:** 2025-03-12  
**Pipeline Script:** `scripts/build_wound_focus_dataset.py`

---

## 1. Overview

This document describes the **Safe Image Renaming and Mapping Pipeline** for the wound infection detection project. The pipeline scans the raw dataset (`data/original_data/task_0/` through `data/original_data/task_240/`), infers infection status from manifest metadata, and produces a clean dataset with standardized filenames and full traceability.

**Key principle:** The raw dataset is **never modified**. All outputs are written to `data/wound_focus_clean/`.

**Data structure:** Raw data lives under `data/original_data/` (task folders, project.json, annotations).

---

## 2. What Was Executed

### 2.1 Command

```bash
cd scripts
python build_wound_focus_dataset.py --data-root ../data --output-dir ../data/wound_focus_clean --copy
```

### 2.2 Pipeline Stages

| Stage | Description | Result |
|-------|-------------|--------|
| 1. Inventory | Scan `data/original_data/task_*/data/`, read `manifest.jsonl`, infer infection status | 531 images processed |
| 2. Mapping | Assign `global_id`, generate new filenames, write CSV/JSON | Mappings written |
| 3. Validation | Duplicate check, missing file check, report generation | No errors |
| 4. Copy | Copy valid images to `wound_focus_clean/images/` | 380 images copied |

---

## 3. Output Structure

```
data/wound_focus_clean/
├── images/                    # 380 copied images with new filenames
│   ├── task_105_img_000001_infected.jpg
│   ├── task_106_img_000002_not_infected.jpg
│   └── ... (380 files total)
├── mappings/
│   ├── image_mapping.csv     # Valid images only (380 rows)
│   ├── image_mapping.json    # Full structure: valid, skipped, ambiguous
│   ├── skipped_images.csv    # 1 row (non-image file)
│   └── ambiguous_cases.csv    # 150 rows (ambiguous infection status)
└── reports/
    └── RENAMING_REPORT.md    # Summary report
```

---

## 4. Exact Statistics

### 4.1 Processing Summary

| Category | Count | Percentage |
|----------|-------|------------|
| **Valid (mapped & copied)** | 380 | 71.6% |
| **Skipped** | 1 | 0.2% |
| **Ambiguous** | 150 | 28.2% |
| **Total processed** | 531 | 100% |

### 4.2 Infection Status (Valid Images Only)

| Status | Count | Percentage |
|--------|-------|------------|
| **infected** (1) | 158 | 41.6% |
| **not_infected** (0) | 222 | 58.4% |

### 4.3 Task Statistics

| Metric | Value |
|--------|-------|
| Tasks with at least one valid image | 139 |
| Tasks with multiple valid images | 89 |
| Total raw task folders scanned | 241 |

### 4.4 Ambiguity Reasons (150 cases)

| Reason | Count | Description |
|--------|-------|-------------|
| purely_numeric | 57 | Manifest name is digits only (e.g. "1", "2", "1739189416119") |
| generic_pattern | 52 | WhatsApp Image, IMG_, phone numbers (+1747...) |
| no_clinical_markers | 41 | Name has no MK/МК, -day-, -inf-, -not- pattern |

### 4.5 Skipped (1 case)

| task_id | original_local_filename | skip_reason |
|---------|-------------------------|-------------|
| 13 | 547b0215-049d-4ab7-980d-91af1aaab287.jfif | non_image (extension .jfif not supported) |

---

## 5. Filename Convention

**Format:** `task_{task_id:03d}_img_{global_id:06d}_{infection_label}.jpg`

| Component | Example | Description |
|-----------|---------|-------------|
| task_id | 014 | Zero-padded task folder number |
| global_id | 000001 | Sequential 1-based index across valid images |
| infection_label | not_infected / infected | From manifest `name` |

**Examples:**
- `task_105_img_000001_infected.jpg`
- `task_106_img_000002_not_infected.jpg`
- `task_014_img_000050_not_infected.jpg`

---

## 6. Infection Status Inference Rules

### 6.1 Determinate (assigned status)

| Condition | Result |
|-----------|--------|
| `-not-` in manifest `name` | `not_infected` (0) |
| Clinical pattern (MK/МК, -day-, -inf-, -hosp-) and no `-not-` | `infected` (1) |

### 6.2 Ambiguous (not assigned, not copied)

| Condition | Reason |
|-----------|--------|
| Empty or missing name | empty_name |
| Purely numeric (e.g. "1", "2", "1739189416119") | purely_numeric |
| WhatsApp Image, IMG_, +phone, long timestamp | generic_pattern |
| No clinical markers in name | no_clinical_markers |

---

## 7. Mapping File Schemas

### 7.1 image_mapping.csv (valid images)

| Field | Type | Description |
|-------|------|-------------|
| global_id | int | Sequential ID (1–380) |
| task_id | int | Source task folder |
| original_local_path | str | e.g. task_105/data/ц-MK540557-inf-day-3-2025-02-28.jpg |
| original_local_filename | str | Original filename |
| manifest_path | str | Path to manifest.jsonl |
| source_name_from_manifest | str | Manifest `name` (source identity) |
| new_filename | str | e.g. task_105_img_000001_infected.jpg |
| infection_status | 0/1 | 0=not_infected, 1=infected |
| annotation_available | bool | Present in annotations_cleaned.json |
| status | str | ok |
| notes | str | Optional |

### 7.2 skipped_images.csv

Same fields + `skip_reason` (e.g. non_image, missing_file, missing_manifest).

### 7.3 ambiguous_cases.csv

Same fields + `ambiguity_reason`. `global_id` and `new_filename` are empty.

---

## 8. Validation Checks Performed

| Check | Result |
|-------|--------|
| Duplicate new filenames | None (all unique) |
| Missing image file | 0 (all valid records have existing files) |
| Missing manifest | 0 (all tasks with images have manifest) |
| Non-image files | 1 (.jfif excluded) |
| Ambiguous infection status | 150 (reported in ambiguous_cases.csv) |

---

## 9. Assumptions

1. **Raw dataset immutable:** No files in `data/original_data/task_*/` are renamed or deleted.
2. **Manifest authoritative:** The `name` field in `manifest.jsonl` is the original source filename.
3. **Extension normalization:** All copied images use `.jpg` extension.
4. **annotation_available:** Based on presence in `data/original_data/annotations_cleaned.json` (if it exists).

---

## 10. How to Re-run

```bash
cd scripts

# Mapping only (no copy). Uses data/original_data/ when present.
python build_wound_focus_dataset.py --data-root ../data --output-dir ../data/wound_focus_clean

# With image copy
python build_wound_focus_dataset.py --data-root ../data --output-dir ../data/wound_focus_clean --copy
```

**Note:** The script auto-detects `data/original_data/` and scans `original_data/task_*/` when that folder exists. Paths in mappings use `original_data/task_N/data/...` format.

---

## 11. File Counts (Verification)

| Path | Count |
|------|-------|
| `data/wound_focus_clean/images/*.jpg` | 380 |
| `data/wound_focus_clean/mappings/image_mapping.csv` | 381 lines (1 header + 380 data) |
| `data/wound_focus_clean/mappings/skipped_images.csv` | 2 lines (1 header + 1 data) |
| `data/wound_focus_clean/mappings/ambiguous_cases.csv` | 151 lines (1 header + 150 data) |

---

## 12. Traceability

Every copied image can be traced back to its source:

1. **new_filename** → **original_local_path** (via `image_mapping.csv` or `image_mapping.json`)
2. **original_local_path** = `task_{id}/data/{original_local_filename}`
3. **source_name_from_manifest** = original source identity from CVAT/manifest
4. **infection_status** = inferred from `source_name_from_manifest`

---

*End of documentation*
