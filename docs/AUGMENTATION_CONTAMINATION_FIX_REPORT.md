# Augmentation Contamination Fix — Final Report

**Date:** 2026-03-11  
**Task:** Fix augmentation pipeline contamination and ensure training uses only cleaned data.

---

## 1. Was the old augmented dataset contaminated?

**Yes.**

---

## 2. Evidence from the repository

| Evidence | Location |
|----------|----------|
| Offline augmentation script originally used `data/splits/train.json` (from pre-cleaning pipeline) | `scripts/apply_augmentation_only.py` — CONFIG default was splits/train.json before fix |
| Training notebook previously loaded `data/augmented/annotations_augmented.json` and `_AUG_ROOT = '../../data/augmented'` | `experiments/maskrcnn/training_pipeline.ipynb` (before fix) |
| Dataset cleaning produces `data/annotations_cleaned.json`; offline augmented data was generated before this step | `scripts/clean_dataset.py` output vs. `data/augmented/` creation timeline |
| Project rules and docs referenced `data/augmented/` and `annotations_augmented.json` as the offline augmented source | `.cursor/rules/projact-roles.mdc`, `docs/PROJECT_OVERVIEW.md` |

**Conclusion:** The offline augmented dataset in `data/augmented/` was built from pre-cleaning annotations (splits or raw COCO), so it contains invalid or inconsistent annotations and is contaminated.

---

## 3. Files that used the old augmented dataset

| File | Usage (before fix) |
|------|--------------------|
| `experiments/maskrcnn/training_pipeline.ipynb` | Loaded `_AUG_ANN_FILE` and `_AUG_ROOT` pointing to `data/augmented/` |
| `scripts/apply_augmentation_only.py` | Output to `data/augmented/`; input from splits/train.json |
| `experiments/maskrcnn/train_model.py` | CONFIG referenced `ann_file_offline_aug` (could point to old path) |
| `.cursor/rules/projact-roles.mdc` | Documented `data/augmented/` and `annotations_augmented.json` |
| `docs/PROJECT_OVERVIEW.md` | Documented `data/augmented/` as optional augmented path |
| `TECHNICAL_REPORT_R-CNN ResNet-50.md` | Referenced `data/augmented/` for offline augmentation |
| `README.md` | Referenced `data/augmented/` in Arabic section |

---

## 4. What changes were made

### Phase 1 — Inspection
- Confirmed offline augmentation was generated from pre-cleaning annotations.
- Identified all references to `data/augmented/` and `annotations_augmented.json`.

### Phase 2 — Option A (implemented)
- Stopped using the old offline augmented dataset.
- Training now uses **cleaned original data** + **online augmentation** only.

### Phase 3 — Training pipeline
- **`training_pipeline.ipynb`**: Switched to `_CLEAN_ANN_FILE` and `_DATA_ROOT`; 82/18 split of `annotations_cleaned.json`; `DATA_MODE` = `clean_online_aug`.
- **`train_model.py`**: Added `data_mode`, `ann_file_cleaned`, `ann_file_offline_aug`; data source order: splits (from cleaned) → `annotations_cleaned` → `ann_file_full`.

### Phase 4 — Safety checks
- **`scripts/apply_augmentation_only.py`**: Default input `annotations_cleaned.json`; output `data/augmented_clean/` and `annotations_augmented_clean.json`.
- **`.cursor/rules/projact-roles.mdc`**: Updated to describe `augmented_clean` and warn against `data/augmented/`.
- **`docs/PROJECT_OVERVIEW.md`**: Updated augmented data section and execution steps.
- **`TECHNICAL_REPORT_R-CNN ResNet-50.md`**: Updated offline augmentation path.
- **`experiments/maskrcnn/README.md`**: Updated validation set description and root cause #3.
- **`README.md`**: Updated Arabic section to reference `annotations_cleaned.json` and `augmented_clean`.

---

## 5. Current data usage

The project now uses:

- **Cleaned dataset only** (Mode 1: `clean_only`): `data/annotations_cleaned.json` or `data/splits/` (from cleaned), no online augmentation.
- **Cleaned dataset + online augmentation** (Mode 2: `clean_online_aug`, **recommended**): Same source + `get_medical_augmentation_pipeline()` from `augmentation_strategy.py`.
- **Cleaned dataset + regenerated offline augmentation** (Mode 3: `clean_offline_aug`, optional): `data/augmented_clean/annotations_augmented_clean.json` when regenerated from cleaned data.

---

## 6. Folders to delete

You may safely delete the old contaminated augmented data:

```
data/augmented/
```

This includes:
- `data/augmented/images/`
- `data/augmented/annotations_augmented.json`

**Note:** Only delete after confirming training works with the new setup. Keep a backup if needed.

---

## 7. Command to regenerate augmentation safely

From the project root:

```bash
cd scripts
python apply_augmentation_only.py
```

The script uses CONFIG defaults: input `data/annotations_cleaned.json`, output `data/augmented_clean/`. To change paths, edit the CONFIG dict in `scripts/apply_augmentation_only.py`.

This produces:
- `data/augmented_clean/images/`
- `data/augmented_clean/annotations_augmented_clean.json`

---

## 8. Dataset file for final training

**Primary (recommended):** `data/annotations_cleaned.json` with 82/18 in-memory split and online augmentation (`data_mode: 'clean_online_aug'`).

**Optional:** `data/augmented_clean/annotations_augmented_clean.json` if you regenerate offline augmentation and set `data_mode: 'clean_offline_aug'`.

**Do not use:** `data/augmented/annotations_augmented.json` — contaminated.
