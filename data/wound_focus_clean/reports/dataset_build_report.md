# Wound-Only Dataset Build Report

## Purpose

This report describes the wound-only dataset-building stage for the standardized `wound_focus_clean` dataset. The goal is to prepare data for:
1. Wound-only segmentation (single class: wound region)
2. Infected vs. non-infected image-level classification

## Input Files Used

- `data/original_data/annotations_cleaned.json` — full COCO annotations (categories 1–8)
- `data/wound_focus_clean/mappings/image_mapping.json` — original path to standardized filename mapping
- `data/wound_focus_clean/images/` — 380 standardized images

## Output Files Created

| File | Purpose |
|------|---------|
| `annotations_wound_only.json` | Wound-only COCO (single class: wound) |
| `labels_infection.json` | Image filename → infection label |
| `labels_infection.csv` | Same, CSV format |
| `train_images.txt` | Train split filenames |
| `val_images.txt` | Val split filenames |
| `test_images.txt` | Test split filenames |
| `train_wound_only.json` | Train COCO (wound-only) |
| `val_wound_only.json` | Val COCO (wound-only) |
| `test_wound_only.json` | Test COCO (wound-only) |
| `mappings/original_to_standardized.json` | Traceability mapping |
| `reports/validation_report.txt` | Validation results |
| `reports/dataset_build_report.md` | This report |
| `reports/review_summary_for_chatgpt.md` | Review-ready summary |

## Class Filtering Strategy

- **Kept:** Category 1 (ВсяРана / whole wound) — remapped to `wound`
- **Removed:** Categories 2–8 (marker, edema, hyperemia, necrosis, granulation, fibrin, pus, suture zone)

Only images present in `wound_focus_clean/images/` and with at least one wound annotation are included in `annotations_wound_only.json`. Images without wound annotations are excluded from the segmentation set but remain in `labels_infection.json` for classification.

## Infection Labeling Strategy

- **Rule:** If original filename (from manifest) contains `-not-` → `non_infected`; otherwise → `infected`
- **Source:** `infection_label` from `image_mapping.json` valid entries
- **Coverage:** All 380 standardized images have determinate labels (no ambiguous cases in wound_focus_clean)

## Split Rebuilding Summary

- **Ratios:** 70% train, 15% val, 15% test
- **Seed:** 42 (reproducible)
- **Split lists:** All 380 images split into train/val/test
- **Wound-only COCO splits:** Each split contains only images with wound annotations; annotations filtered accordingly

## Statistics

| Metric | Count |
|--------|-------|
| Total standardized images | 380 |
| Images with wound annotations | 369 |
| Wound annotations (total) | 532 |
| Images skipped (no wound ann) | 11 |
| Infected | 158 |
| Non-infected | 222 |
| Train images | 266 |
| Val images | 57 |
| Test images | 57 |
| Train wound-only images | 257 |
| Val wound-only images | 57 |
| Test wound-only images | 55 |

## Consistency Check

All equations verified from actual files:

| Equation | Result |
|----------|--------|
| infected + non_infected = total standardized images | 158 + 222 = 380 ✓ |
| train + val + test = total standardized images | 266 + 57 + 57 = 380 ✓ |
| train_wound_only + val_wound_only + test_wound_only = images with wound ann | 257 + 57 + 55 = 369 ✓ |
| images with wound ann + images without wound ann = total standardized images | 369 + 11 = 380 ✓ |

## Validation Results

PASSED

All checks passed.

## Detected Issues

None. (Previously the report stated "Images skipped (no wound ann) = 10"; the correct count is 11, verified by comparing `image_mapping.json` valid entries vs `annotations_wound_only.json` image IDs.)

## Recommendations for Next Stage

1. Use `train_wound_only.json` with root `data/wound_focus_clean` for wound-only segmentation training.
2. Use `labels_infection.json` with split files for infected vs. non-infected classification.
3. Ensure the dataset loader uses `file_name` relative to `data/wound_focus_clean` (e.g. `images/task_105_img_000001_infected.jpg`).
