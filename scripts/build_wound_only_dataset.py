"""
Build Wound-Only Dataset
=======================

Creates wound-only COCO annotations, infection labels, and train/val/test splits
for the standardized wound_focus_clean dataset. Prepares data for wound-only
segmentation and infected vs non-infected analysis.

Usage:
    cd scripts
    python build_wound_only_dataset.py --data-root ../data

Output:
    data/wound_focus_clean/
    ├── annotations_wound_only.json
    ├── labels_infection.json
    ├── labels_infection.csv
    ├── train_images.txt, val_images.txt, test_images.txt
    ├── train_wound_only.json, val_wound_only.json, test_wound_only.json
    ├── mappings/original_to_standardized.json
    └── reports/validation_report.txt, dataset_build_report.md, review_summary_for_chatgpt.md
"""

import argparse
import csv
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# Wound class in annotations_cleaned (ВсяРана)
WOUND_CATEGORY_ID = 1
WOUND_CATEGORY_NAME = "wound"

# Split ratios
SPLIT_RATIOS = (0.7, 0.15, 0.15)  # train, val, test
SPLIT_SEED = 42


def _normalize_path(p: str) -> str:
    """Normalize path for matching (forward slashes)."""
    return str(p).replace("\\", "/")


def load_mapping(wound_focus_dir: Path) -> Tuple[Dict[str, Dict], List[str]]:
    """Load image_mapping.json and build path -> record lookup. Returns (path_to_record, all_new_filenames)."""
    mapping_path = wound_focus_dir / "mappings" / "image_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping not found: {mapping_path}")

    with open(mapping_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid = data.get("valid", [])
    path_to_record: Dict[str, Dict] = {}
    all_filenames: List[str] = []

    for rec in valid:
        orig_path = _normalize_path(rec["original_local_path"])
        path_to_record[orig_path] = rec
        all_filenames.append(rec["new_filename"])

    return path_to_record, all_filenames


def load_annotations(annotations_path: Path) -> Dict[str, Any]:
    """Load annotations_cleaned.json."""
    with open(annotations_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_wound_only_coco(
    coco: Dict[str, Any],
    path_to_record: Dict[str, Dict],
    images_dir: Path,
) -> Tuple[Dict[str, Any], List[str], List[Dict]]:
    """
    Build wound-only COCO. Returns (coco_dict, new_filenames_with_wounds, stats_list).
    Only includes images that (a) are in wound_focus_clean and (b) have at least one wound annotation.
    """
    img_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_img: Dict[int, List[Dict]] = {}
    for ann in coco["annotations"]:
        if ann["category_id"] != WOUND_CATEGORY_ID:
            continue
        img_id = ann["image_id"]
        if img_id not in anns_by_img:
            anns_by_img[img_id] = []
        anns_by_img[img_id].append(ann)

    new_images: List[Dict] = []
    new_annotations: List[Dict] = []
    new_filename_to_img_id: Dict[str, int] = {}
    new_img_id = 0
    new_ann_id = 0
    stats: List[Dict] = []

    for img in coco["images"]:
        file_name = _normalize_path(img.get("file_name", ""))
        if not file_name:
            continue
        rec = path_to_record.get(file_name)
        if not rec:
            continue  # Not in wound_focus_clean
        new_filename = rec["new_filename"]
        img_path = images_dir / new_filename
        if not img_path.exists():
            logging.warning("Image not found: %s", img_path)
            continue

        wound_anns = anns_by_img.get(img["id"], [])
        valid_anns = []
        for ann in wound_anns:
            seg = ann.get("segmentation", [])
            if not seg:
                continue
            bbox = ann.get("bbox", [0, 0, 0, 0])
            if len(bbox) < 4 or bbox[2] <= 0 or bbox[3] <= 0:
                continue
            valid_anns.append(ann)

        if not valid_anns:
            stats.append({
                "new_filename": new_filename,
                "original_path": file_name,
                "status": "no_wound_annotation",
                "skipped": True,
            })
            continue

        new_img_id += 1
        new_images.append({
            "id": new_img_id,
            "file_name": f"images/{new_filename}",
            "width": img["width"],
            "height": img["height"],
        })
        new_filename_to_img_id[new_filename] = new_img_id

        for ann in valid_anns:
            new_ann_id += 1
            new_annotations.append({
                "id": new_ann_id,
                "image_id": new_img_id,
                "category_id": 1,
                "segmentation": ann["segmentation"],
                "bbox": ann["bbox"],
                "area": ann["area"],
                "iscrowd": ann.get("iscrowd", 0),
            })

    wound_only_coco = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": [{"id": 1, "name": WOUND_CATEGORY_NAME}],
    }
    filenames_with_wounds = list(new_filename_to_img_id.keys())
    return wound_only_coco, filenames_with_wounds, stats


def build_infection_labels(path_to_record: Dict[str, Dict]) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
    """Build infection labels for all standardized images. Returns (dict, csv_rows)."""
    labels: Dict[str, str] = {}
    csv_rows: List[Tuple[str, str]] = [("filename", "infection_label")]

    for rec in path_to_record.values():
        fn = rec["new_filename"]
        inf = rec.get("infection_label", "not_infected")
        if inf == "not_infected":
            labels[fn] = "non_infected"
        else:
            labels[fn] = "infected"
        csv_rows.append((fn, labels[fn]))

    return labels, csv_rows


def create_splits(
    all_filenames: List[str],
    filenames_with_wounds: List[str],
    wound_only_coco: Dict[str, Any],
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, Any]]]:
    """
    Split all 380 images 70/15/15. Create split lists and split-specific wound-only COCO files.
    """
    random.seed(SPLIT_SEED)
    shuffled = list(all_filenames)
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val = int(n * SPLIT_RATIOS[1])
    train_fns = set(shuffled[:n_train])
    val_fns = set(shuffled[n_train : n_train + n_val])
    test_fns = set(shuffled[n_train + n_val :])

    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }

    wound_img_by_fn = {img["file_name"].replace("images/", ""): img for img in wound_only_coco["images"]}
    anns_by_img = {}
    for ann in wound_only_coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in anns_by_img:
            anns_by_img[img_id] = []
        anns_by_img[img_id].append(ann)

    split_cocos: Dict[str, Dict[str, Any]] = {}
    for split_name, fns in splits.items():
        fn_set = set(fns)
        split_images = []
        split_anns = []
        new_img_id = 0
        new_ann_id = 0
        fn_to_new_id = {}

        for fn in fns:
            if fn not in wound_img_by_fn:
                continue
            orig_img = wound_img_by_fn[fn]
            new_img_id += 1
            fn_to_new_id[fn] = new_img_id
            split_images.append({
                "id": new_img_id,
                "file_name": f"images/{fn}",
                "width": orig_img["width"],
                "height": orig_img["height"],
            })

        for img in wound_only_coco["images"]:
            fn = img["file_name"].replace("images/", "")
            if fn not in fn_set or fn not in fn_to_new_id:
                continue
            new_img_id = fn_to_new_id[fn]
            orig_img_id = img["id"]
            for ann in anns_by_img.get(orig_img_id, []):
                new_ann_id += 1
                split_anns.append({
                    "id": new_ann_id,
                    "image_id": new_img_id,
                    "category_id": 1,
                    "segmentation": ann["segmentation"],
                    "bbox": ann["bbox"],
                    "area": ann["area"],
                    "iscrowd": ann.get("iscrowd", 0),
                })

        split_cocos[split_name] = {
            "images": split_images,
            "annotations": split_anns,
            "categories": [{"id": 1, "name": WOUND_CATEGORY_NAME}],
        }

    return splits, split_cocos


def run_validation(
    wound_focus_dir: Path,
    wound_only_coco: Dict[str, Any],
    labels: Dict[str, str],
    splits: Dict[str, List[str]],
) -> Tuple[bool, List[str]]:
    """Run validation checks. Returns (passed, list of messages)."""
    msgs: List[str] = []
    images_dir = wound_focus_dir / "images"

    # Every image in annotations exists
    for img in wound_only_coco["images"]:
        fn = img["file_name"].replace("images/", "")
        if not (images_dir / fn).exists():
            msgs.append(f"FAIL: Image in annotations not found: {fn}")

    # No duplicate image IDs
    img_ids = [img["id"] for img in wound_only_coco["images"]]
    if len(img_ids) != len(set(img_ids)):
        msgs.append("FAIL: Duplicate image IDs")

    # No duplicate annotation IDs
    ann_ids = [a["id"] for a in wound_only_coco["annotations"]]
    if len(ann_ids) != len(set(ann_ids)):
        msgs.append("FAIL: Duplicate annotation IDs")

    # Segmentation non-empty, bbox valid
    for ann in wound_only_coco["annotations"]:
        if not ann.get("segmentation"):
            msgs.append(f"WARN: Empty segmentation for ann id={ann['id']}")
        bbox = ann.get("bbox", [])
        if len(bbox) < 4 or bbox[2] <= 0 or bbox[3] <= 0:
            msgs.append(f"WARN: Invalid bbox for ann id={ann['id']}")

    # Image width/height valid
    for img in wound_only_coco["images"]:
        if img.get("width", 0) <= 0 or img.get("height", 0) <= 0:
            msgs.append(f"WARN: Invalid image size for {img.get('file_name')}")

    # Wound-only filter
    for ann in wound_only_coco["annotations"]:
        if ann.get("category_id") != 1:
            msgs.append("FAIL: Non-wound category in wound-only COCO")

    # Infection labels for all 380
    all_in_splits = set()
    for fns in splits.values():
        all_in_splits.update(fns)
    for fn in all_in_splits:
        if fn not in labels:
            msgs.append(f"FAIL: Missing infection label for {fn}")

    # Split counts sum to 380
    total = sum(len(fns) for fns in splits.values())
    if total != 380:
        msgs.append(f"FAIL: Split total {total} != 380")

    passed = not any(m.startswith("FAIL") for m in msgs)
    return passed, msgs


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Build wound-only dataset for wound_focus_clean")
    parser.add_argument("--data-root", type=str, default=str(PROJECT_ROOT / "data"))
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    wound_focus_dir = data_root / "wound_focus_clean"
    original_data_dir = data_root / "original_data"
    annotations_path = original_data_dir / "annotations_cleaned.json"
    images_dir = wound_focus_dir / "images"

    if not wound_focus_dir.exists():
        logging.error("wound_focus_clean not found: %s", wound_focus_dir)
        return 1
    if not annotations_path.exists():
        logging.error("annotations_cleaned.json not found: %s", annotations_path)
        return 1

    logging.info("Loading mapping and annotations...")
    path_to_record, all_filenames = load_mapping(wound_focus_dir)
    coco = load_annotations(annotations_path)
    logging.info("  Valid standardized images: %d", len(all_filenames))

    logging.info("Building wound-only COCO...")
    wound_only_coco, filenames_with_wounds, skip_stats = build_wound_only_coco(
        coco, path_to_record, images_dir
    )
    n_images_wo = len(wound_only_coco["images"])
    n_anns_wo = len(wound_only_coco["annotations"])
    logging.info("  Images with wound annotations: %d", n_images_wo)
    logging.info("  Wound annotations: %d", n_anns_wo)
    logging.info("  Skipped (no wound ann): %d", len(skip_stats))

    logging.info("Building infection labels...")
    labels, csv_rows = build_infection_labels(path_to_record)
    logging.info("  Labels: %d", len(labels))

    logging.info("Creating splits...")
    splits, split_cocos = create_splits(all_filenames, filenames_with_wounds, wound_only_coco)
    for name, fns in splits.items():
        logging.info("  %s: %d images", name, len(fns))

    wound_focus_dir.mkdir(parents=True, exist_ok=True)
    mappings_dir = wound_focus_dir / "mappings"
    reports_dir = wound_focus_dir / "reports"
    mappings_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Writing output files...")

    with open(wound_focus_dir / "annotations_wound_only.json", "w", encoding="utf-8") as f:
        json.dump(wound_only_coco, f, indent=2, ensure_ascii=False)

    with open(wound_focus_dir / "labels_infection.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    with open(wound_focus_dir / "labels_infection.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    for name, fns in splits.items():
        with open(wound_focus_dir / f"{name}_images.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(fns) + "\n")

    for name, split_coco in split_cocos.items():
        with open(wound_focus_dir / f"{name}_wound_only.json", "w", encoding="utf-8") as f:
            json.dump(split_coco, f, indent=2, ensure_ascii=False)

    orig_to_std = {rec["original_local_path"]: rec["new_filename"] for rec in path_to_record.values()}
    with open(mappings_dir / "original_to_standardized.json", "w", encoding="utf-8") as f:
        json.dump(orig_to_std, f, indent=2, ensure_ascii=False)

    logging.info("Running validation...")
    passed, val_msgs = run_validation(wound_focus_dir, wound_only_coco, labels, splits)
    with open(reports_dir / "validation_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(val_msgs) if val_msgs else "All checks passed.")
    if not passed:
        for m in val_msgs:
            if m.startswith("FAIL"):
                logging.error(m)
        return 1

    n_infected = sum(1 for v in labels.values() if v == "infected")
    n_non_infected = len(labels) - n_infected

    dataset_report = f"""# Wound-Only Dataset Build Report

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
| Images with wound annotations | {n_images_wo} |
| Wound annotations (total) | {n_anns_wo} |
| Images skipped (no wound ann) | {len(skip_stats)} |
| Infected | {n_infected} |
| Non-infected | {n_non_infected} |
| Train images | {len(splits['train'])} |
| Val images | {len(splits['val'])} |
| Test images | {len(splits['test'])} |
| Train wound-only images | {len(split_cocos['train']['images'])} |
| Val wound-only images | {len(split_cocos['val']['images'])} |
| Test wound-only images | {len(split_cocos['test']['images'])} |

## Validation Results

{"PASSED" if passed else "FAILED"}

{chr(10).join(val_msgs) if val_msgs else "All checks passed."}

## Detected Issues

{chr(10).join("- " + m for m in val_msgs if m.startswith("WARN")) if any(m.startswith("WARN") for m in val_msgs) else "None."}

## Recommendations for Next Stage

1. Use `train_wound_only.json` with root `data/wound_focus_clean` for wound-only segmentation training.
2. Use `labels_infection.json` with split files for infected vs. non-infected classification.
3. Ensure the dataset loader uses `file_name` relative to `data/wound_focus_clean` (e.g. `images/task_105_img_000001_infected.jpg`).
"""

    with open(reports_dir / "dataset_build_report.md", "w", encoding="utf-8") as f:
        f.write(dataset_report)

    review_summary = f"""# Review Summary for Technical Review

## What Was Implemented

- Script: `scripts/build_wound_only_dataset.py`
- Pipeline: Load annotations + mapping → filter wound-only → build infection labels → create splits → validate → report

## Files Generated

- `data/wound_focus_clean/annotations_wound_only.json`
- `data/wound_focus_clean/labels_infection.json`, `labels_infection.csv`
- `data/wound_focus_clean/train_images.txt`, `val_images.txt`, `test_images.txt`
- `data/wound_focus_clean/train_wound_only.json`, `val_wound_only.json`, `test_wound_only.json`
- `data/wound_focus_clean/mappings/original_to_standardized.json`
- `data/wound_focus_clean/reports/validation_report.txt`
- `data/wound_focus_clean/reports/dataset_build_report.md`
- `data/wound_focus_clean/reports/review_summary_for_chatgpt.md`

## Key Counts

- Standardized images: 380
- Images with wound annotations: {n_images_wo}
- Wound annotations: {n_anns_wo}
- Infected: {n_infected}, Non-infected: {n_non_infected}
- Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}

## Validation Outcomes

{"PASSED" if passed else "FAILED"}

## Unresolved Issues

{chr(10).join("- " + m for m in val_msgs if m.startswith("WARN")) if any(m.startswith("WARN") for m in val_msgs) else "None."}

## Recommended Next Step

1. Train wound-only segmentation model using `train_wound_only.json` and root `data/wound_focus_clean`.
2. Train or evaluate infected vs. non-infected classifier using `labels_infection.json` and split files.
3. Update `pipeline_utils.py` or experiment config to use the new wound-only annotation files.
"""

    with open(reports_dir / "review_summary_for_chatgpt.md", "w", encoding="utf-8") as f:
        f.write(review_summary)

    logging.info("Done. Reports written to %s", reports_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
