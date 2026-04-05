"""
Build Wound + Marker Dataset
=============================

Rebuilds COCO annotations for wound_focus_clean images to include both
the wound (ВсяРана, id=1) and marker (Метка для размерности, id=2) classes.

The output JSONs are compatible with the existing pipeline — just update
config.yaml to point ``ann_train/val/test`` at the new files.

Usage:
    python build_wound_marker_dataset.py
    python build_wound_marker_dataset.py --data-root ../../data
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent

WOUND_CAT_ID_ORIG = 1   # ВсяРана in annotations_cleaned.json
MARKER_CAT_ID_ORIG = 2  # Метка для размерности in annotations_cleaned.json

OUTPUT_CATEGORIES = [
    {"id": 1, "name": "wound"},
    {"id": 2, "name": "marker"},
]

CAT_ID_MAP = {
    WOUND_CAT_ID_ORIG: 1,
    MARKER_CAT_ID_ORIG: 2,
}

SPLIT_SEED = 42
SPLIT_RATIOS = (0.7, 0.15, 0.15)


def _normalize_path(p: str) -> str:
    return str(p).replace("\\", "/")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build wound+marker COCO JSONs")
    parser.add_argument("--data-root", type=str, default=str(PROJECT_ROOT / "data"))
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    wound_focus_dir = data_root / "wound_focus_clean"
    original_data_dir = data_root / "original_data"
    annotations_path = original_data_dir / "annotations_cleaned.json"
    images_dir = wound_focus_dir / "images"
    mapping_path = wound_focus_dir / "mappings" / "image_mapping.json"

    for p, label in [
        (wound_focus_dir, "wound_focus_clean"),
        (annotations_path, "annotations_cleaned.json"),
        (mapping_path, "image_mapping.json"),
    ]:
        if not p.exists():
            logging.error("%s not found: %s", label, p)
            return 1

    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    path_to_record = {}
    for rec in mapping.get("valid", []):
        orig = _normalize_path(rec["original_local_path"])
        path_to_record[orig] = rec

    with open(annotations_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    keep_cats: Set[int] = set(CAT_ID_MAP.keys())

    img_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_img: Dict[int, List[Dict]] = {}
    for ann in coco["annotations"]:
        if ann["category_id"] in keep_cats:
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

    new_images: List[Dict] = []
    new_annotations: List[Dict] = []
    new_img_id = 0
    new_ann_id = 0
    wound_count = 0
    marker_count = 0

    for img in coco["images"]:
        file_name = _normalize_path(img.get("file_name", ""))
        if not file_name:
            continue
        rec = path_to_record.get(file_name)
        if not rec:
            continue
        new_filename = rec["new_filename"]
        img_path = images_dir / new_filename
        if not img_path.exists():
            continue

        orig_anns = anns_by_img.get(img["id"], [])
        has_wound = any(a["category_id"] == WOUND_CAT_ID_ORIG for a in orig_anns)
        if not has_wound:
            continue

        valid_anns = []
        for ann in orig_anns:
            seg = ann.get("segmentation", [])
            if not seg:
                continue
            bbox = ann.get("bbox", [0, 0, 0, 0])
            if len(bbox) < 4 or bbox[2] <= 0 or bbox[3] <= 0:
                continue
            valid_anns.append(ann)

        if not valid_anns:
            continue

        new_img_id += 1
        new_images.append({
            "id": new_img_id,
            "file_name": f"images/{new_filename}",
            "width": img["width"],
            "height": img["height"],
        })

        for ann in valid_anns:
            new_ann_id += 1
            mapped_cat = CAT_ID_MAP[ann["category_id"]]
            new_annotations.append({
                "id": new_ann_id,
                "image_id": new_img_id,
                "category_id": mapped_cat,
                "segmentation": ann["segmentation"],
                "bbox": ann["bbox"],
                "area": ann["area"],
                "iscrowd": ann.get("iscrowd", 0),
            })
            if mapped_cat == 1:
                wound_count += 1
            else:
                marker_count += 1

    full_coco = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": OUTPUT_CATEGORIES,
    }

    logging.info("Images: %d, Wound anns: %d, Marker anns: %d",
                 len(new_images), wound_count, marker_count)

    random.seed(SPLIT_SEED)
    img_ids = [img["id"] for img in new_images]
    random.shuffle(img_ids)
    n = len(img_ids)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val = int(n * SPLIT_RATIOS[1])
    train_ids = set(img_ids[:n_train])
    val_ids = set(img_ids[n_train:n_train + n_val])
    test_ids = set(img_ids[n_train + n_val:])

    id_to_img = {img["id"]: img for img in new_images}

    for split_name, split_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        split_images = [id_to_img[i] for i in sorted(split_ids)]
        split_anns = [a for a in new_annotations if a["image_id"] in split_ids]
        split_coco = {
            "images": split_images,
            "annotations": split_anns,
            "categories": OUTPUT_CATEGORIES,
        }
        out_path = wound_focus_dir / f"{split_name}_wound_marker.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(split_coco, f, indent=2, ensure_ascii=False)
        logging.info("  %s: %d images, %d annotations -> %s",
                     split_name, len(split_images), len(split_anns), out_path)

    full_path = wound_focus_dir / "annotations_wound_marker.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(full_coco, f, indent=2, ensure_ascii=False)
    logging.info("Full annotations: %s", full_path)

    print("\nTo use wound+marker data, update config.yaml:")
    print('  ann_train: "data/wound_focus_clean/train_wound_marker.json"')
    print('  ann_val:   "data/wound_focus_clean/val_wound_marker.json"')
    print('  ann_test:  "data/wound_focus_clean/test_wound_marker.json"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
