"""
Validate Cleaned Dataset
========================

Validates annotations_cleaned.json for Mask R-CNN training:
- All category names in TARGET_CLASSES
- Category IDs 1..8
- No empty segmentations
- Bbox w, h > 0
- Area > 0
- Polygons inside image bounds

Usage:
    cd scripts
    python validate_cleaned_dataset.py
    python validate_cleaned_dataset.py --input ../data/annotations_cleaned.json
"""

import argparse
import json
import sys
from pathlib import Path

# Resolve project root
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

TARGET_CLASSES = [
    "ВсяРана",
    "Метка для размерности",
    "Зона отека вокруг раны",
    "Зона гиперемии вокруг",
    "Зона некроза",
    "Зона грануляций",
    "Фибрин",
    "Гнойное отделяемое",
]


def main():
    parser = argparse.ArgumentParser(description="Validate cleaned wound dataset")
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "data" / "annotations_cleaned.json"),
        help="Path to cleaned COCO JSON",
    )
    parser.add_argument(
        "--sample-masks",
        type=int,
        default=0,
        help="Sample N images and verify masks render (0 = skip)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = (PROJECT_ROOT / input_path).resolve()

    if not input_path.exists():
        print(f"[ERROR] File not found: {input_path}")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    categories = {c["id"]: c["name"] for c in data["categories"]}
    annotations = data["annotations"]

    passed = 0
    failed = 0

    # Check 1: Categories are exactly TARGET_CLASSES with IDs 1..8
    expected_cats = {i + 1: name for i, name in enumerate(TARGET_CLASSES)}
    if set(categories.items()) == set(expected_cats.items()):
        print("[PASS] Categories: 8 classes with IDs 1..8 matching TARGET_CLASSES")
        passed += 1
    else:
        print(f"[FAIL] Categories: expected {expected_cats}, got {categories}")
        failed += 1

    # Check 2: No empty segmentations
    empty_seg = [a for a in annotations if not a.get("segmentation") or len(a["segmentation"]) == 0]
    if not empty_seg:
        print("[PASS] No empty segmentations")
        passed += 1
    else:
        print(f"[FAIL] {len(empty_seg)} annotations with empty segmentation")
        failed += 1

    # Check 3: Bbox w, h > 0
    bad_bbox = []
    for a in annotations:
        bbox = a.get("bbox", [])
        if len(bbox) != 4:
            bad_bbox.append(a["id"])
            continue
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            bad_bbox.append(a["id"])
    if not bad_bbox:
        print("[PASS] All bboxes have w > 0 and h > 0")
        passed += 1
    else:
        print(f"[FAIL] {len(bad_bbox)} annotations with invalid bbox: {bad_bbox[:5]}...")
        failed += 1

    # Check 4: Area > 0
    bad_area = [a["id"] for a in annotations if a.get("area", 0) <= 0]
    if not bad_area:
        print("[PASS] All annotations have area > 0")
        passed += 1
    else:
        print(f"[FAIL] {len(bad_area)} annotations with area <= 0")
        failed += 1

    # Check 5: Polygons inside image bounds
    out_of_bounds = []
    for a in annotations:
        img_id = a["image_id"]
        img = images.get(img_id)
        if not img:
            out_of_bounds.append(a["id"])
            continue
        w, h = img["width"], img["height"]
        for seg in a.get("segmentation", []):
            if not isinstance(seg, list) or len(seg) < 6:
                continue
            for i in range(0, len(seg), 2):
                x, y = seg[i], seg[i + 1]
                if x < 0 or x >= w or y < 0 or y >= h:
                    out_of_bounds.append(a["id"])
                    break
    if not out_of_bounds:
        print("[PASS] All polygon coordinates inside image bounds")
        passed += 1
    else:
        print(f"[FAIL] {len(set(out_of_bounds))} annotations with coordinates outside image")
        failed += 1

    # Check 6: All category_id in 1..8
    bad_cat = [a["id"] for a in annotations if a.get("category_id") not in range(1, 9)]
    if not bad_cat:
        print("[PASS] All category_id in [1, 8]")
        passed += 1
    else:
        print(f"[FAIL] {len(bad_cat)} annotations with category_id outside 1..8")
        failed += 1

    # Summary
    print()
    print(f"Validation: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)

    # Optional: sample mask rendering
    if args.sample_masks > 0:
        try:
            import cv2
            import numpy as np
            data_root = input_path.parent  # annotations in data/; file_name relative to data/
            for i, ann in enumerate(annotations[: args.sample_masks]):
                img_id = ann["image_id"]
                img_info = images.get(img_id)
                if not img_info:
                    continue
                img_path = data_root / img_info["file_name"]
                if not img_path.exists():
                    img_path = input_path.parent / img_info["file_name"]
                if not img_path.exists():
                    print(f"[WARN] Image not found: {img_info['file_name']}")
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"[WARN] Could not load: {img_path}")
                    continue
                h, w = img.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                for seg in ann.get("segmentation", []):
                    poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [poly], 1)
                if mask.sum() == 0:
                    print(f"[WARN] Annotation {ann['id']}: mask is empty after fillPoly")
                else:
                    print(f"[OK] Annotation {ann['id']}: mask area = {mask.sum()} px")
        except ImportError:
            print("[WARN] cv2/numpy not available; skip mask sampling")


if __name__ == "__main__":
    main()
