# Wound Focus Clean Dataset - Renaming Report

## How Filenames Are Derived

New filenames follow: `task_{task_id:03d}_img_{global_id:06d}_{infection_label}.jpg`
- `task_id`: from folder name (e.g. task_14 → 014)
- `global_id`: sequential 1-based index across valid images
- `infection_label`: `not_infected` or `infected` (only for determinate cases)

## How Infection Status Is Inferred

- **not_infected (0)**: manifest `name` contains `-not-`
- **infected (1)**: manifest `name` has clinical pattern (MK/МК, -day-, -inf-, -hosp-) and no `-not-`
- **ambiguous**: name empty, purely numeric, generic (WhatsApp, IMG_, etc.), or no clinical markers

## Assumptions

- Raw dataset is immutable; only copies are created in wound_focus_clean/images/
- Manifest `name` is the original source filename (authoritative for infection)
- Extension normalized to .jpg on copy
- annotation_available = whether image appears in annotations_cleaned.json

## Summary

- **Valid (mapped)**: 380
- **Skipped**: 1
- **Ambiguous**: 150
- **Tasks with images**: 139
- **Tasks with multiple images**: 89
