"""
Dataset Cleaning Script
=======================

Cleans wound annotations for Mask R-CNN training:
- Filters to 8 target classes
- Remaps to contiguous IDs 1..8
- Simplifies noisy polygons (Douglas-Peucker)
- Removes invalid annotations (empty, zero-area, bad bbox, outside image)
- Recomputes bbox and area from cleaned polygons

Usage:
    cd scripts
    python clean_dataset.py --input-mode cvat --data-root ../data
    python clean_dataset.py --input-mode coco --input-file ../data/annotations.json

Output:
    - data/annotations_cleaned.json
    - data/cleaning_report.txt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np

# Resolve project root (parent of scripts/)
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# ============================================================================
# Configurable thresholds
# ============================================================================

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

MIN_POLYGON_POINTS = 3
MAX_POLYGON_POINTS = 500
SIMPLIFY_EPSILON = 0.5  # Douglas-Peucker tolerance (multiplier of arc length)
MIN_MASK_AREA_PX = 4
MIN_BBOX_SIDE_PX = 2
MIN_OBJECT_AREA_RATIO = 1e-6  # object area / image area


# ============================================================================
# Load from CVAT
# ============================================================================

def _parse_cvat_points(points: Any) -> Optional[List[List[float]]]:
    """Parse polygon points from CVAT format (flat list or list of pairs)."""
    if not isinstance(points, list) or len(points) == 0:
        return None
    if isinstance(points[0], (int, float)):
        if len(points) % 2 != 0:
            return None
        return [[float(points[i]), float(points[i + 1])] for i in range(0, len(points), 2)]
    if all(isinstance(p, (list, tuple)) and len(p) == 2 for p in points):
        return [[float(p[0]), float(p[1])] for p in points]
    return None


def load_from_cvat(data_root: Path) -> Tuple[List[Dict], List[Dict], Dict[int, str]]:
    """
    Load images and annotations from CVAT task folders.
    Returns: (images, annotations, category_id_to_name)
    """
    project_file = data_root / "project.json"
    if not project_file.exists():
        raise FileNotFoundError(f"project.json not found at {project_file}")

    with open(project_file, "r", encoding="utf-8") as f:
        project_info = json.load(f)
    label_map = {label["name"]: idx for idx, label in enumerate(project_info["labels"])}

    images: List[Dict] = []
    annotations: List[Dict] = []
    image_id = 0
    annotation_id = 0

    task_folders = sorted([f for f in data_root.iterdir() if f.is_dir() and f.name.startswith("task_")])
    print(f"[DEBUG] Processing {len(task_folders)} CVAT tasks...")

    for task_folder in task_folders:
        try:
            ann_file = task_folder / "annotations.json"
            if not ann_file.exists():
                continue
            with open(ann_file, "r", encoding="utf-8") as f:
                cvat_anns = json.load(f)

            data_dir = task_folder / "data"
            image_files = list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png"))

            for img_file in image_files:
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                h, w = img.shape[:2]
                is_infected = "-not-" not in img_file.name.lower()
                images.append({
                    "id": image_id,
                    "file_name": str(img_file.relative_to(data_root)),
                    "width": w,
                    "height": h,
                    "infection_status": is_infected,
                })

                if len(cvat_anns) > 0 and "shapes" in cvat_anns[0]:
                    for shape in cvat_anns[0]["shapes"]:
                        if shape["type"] != "polygon" or shape["label"] not in label_map:
                            continue
                        points = _parse_cvat_points(shape["points"])
                        if points is None or len(points) < MIN_POLYGON_POINTS:
                            continue
                        polygon = [c for p in points for c in p]
                        x_coords = [p[0] for p in points]
                        y_coords = [p[1] for p in points]
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)
                        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                        area = (x_max - x_min) * (y_max - y_min)
                        annotations.append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": label_map[shape["label"]],
                            "category_name": shape["label"],
                            "segmentation": [polygon],
                            "area": area,
                            "bbox": bbox,
                            "iscrowd": 0,
                        })
                        annotation_id += 1

                image_id += 1
        except Exception as e:
            print(f"[WARNING] Error processing {task_folder.name}: {e}")

    # Build category_id_to_name from project
    cat_id_to_name = {idx: name for name, idx in label_map.items()}
    return images, annotations, cat_id_to_name


# ============================================================================
# Load from COCO
# ============================================================================

def load_from_coco(input_file: Path) -> Tuple[List[Dict], List[Dict], Dict[int, str]]:
    """Load images and annotations from COCO JSON."""
    with open(input_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}

    # Add category_name to each annotation for filtering
    for ann in annotations:
        ann["category_name"] = cat_id_to_name.get(ann["category_id"], "unknown")

    print(f"[DEBUG] Loaded {len(images)} images, {len(annotations)} annotations from COCO")
    return images, annotations, cat_id_to_name


# ============================================================================
# Polygon simplification
# ============================================================================

def simplify_polygon(points: List[List[float]], max_points: int, epsilon: float) -> List[List[float]]:
    """
    Simplify polygon using Douglas-Peucker if it has too many points.
    Returns cleaned list of [x, y] points.
    """
    if len(points) <= max_points:
        return points
    contour = np.array(points, dtype=np.float32)
    arc_len = cv2.arcLength(contour, True)
    eps = epsilon * arc_len
    approx = cv2.approxPolyDP(contour, eps, True)
    result = approx.reshape(-1, 2).tolist()
    if len(result) < MIN_POLYGON_POINTS:
        return points  # Keep original if simplification collapses
    return result


# ============================================================================
# Validation and cleaning
# ============================================================================

def _polygon_area(points: List[List[float]]) -> float:
    """Compute polygon area using cv2.contourArea."""
    if len(points) < 3:
        return 0.0
    contour = np.array(points, dtype=np.float32)
    return float(cv2.contourArea(contour))


def _clamp_polygon_to_bounds(
    points: List[List[float]], img_w: int, img_h: int
) -> Tuple[List[List[float]], bool]:
    """
    Clamp polygon coordinates to [0, W) x [0, H).
    Returns (clamped_points, valid). valid=False if polygon is fully outside.
    """
    clamped = []
    for x, y in points:
        x = max(0.0, min(float(x), img_w - 1e-6))
        y = max(0.0, min(float(y), img_h - 1e-6))
        clamped.append([x, y])
    # Check if polygon has any area after clamping
    area = _polygon_area(clamped)
    return clamped, area >= MIN_MASK_AREA_PX


def clean_annotation(
    ann: Dict,
    img_w: int,
    img_h: int,
    name_to_new_id: Dict[str, int],
    stats: Dict[str, int],
) -> Optional[Dict]:
    """
    Clean a single annotation. Returns None if invalid.
    """
    # 1. Filter by target class
    cat_name = ann.get("category_name") or ""
    if cat_name not in name_to_new_id:
        stats["non_target_class"] += 1
        return None

    segs = ann.get("segmentation", [])
    if not segs:
        stats["empty_segmentation"] += 1
        return None

    # Process first polygon (COCO can have multiple RLE/polygons per object; we use first)
    poly_flat = segs[0] if isinstance(segs[0], list) else segs
    if len(poly_flat) < 6:  # 3 points = 6 coords
        stats["invalid_polygon"] += 1
        return None

    points = [[float(poly_flat[i]), float(poly_flat[i + 1])] for i in range(0, len(poly_flat), 2)]
    if len(points) < MIN_POLYGON_POINTS:
        stats["invalid_polygon"] += 1
        return None

    # 2. Simplify if needed
    points = simplify_polygon(points, MAX_POLYGON_POINTS, SIMPLIFY_EPSILON)
    if len(points) < MIN_POLYGON_POINTS:
        stats["invalid_polygon"] += 1
        return None

    # 3. Clamp to image bounds
    points, valid = _clamp_polygon_to_bounds(points, img_w, img_h)
    if not valid:
        stats["outside_image"] += 1
        return None

    # 4. Recompute area from polygon
    area = _polygon_area(points)
    if area < MIN_MASK_AREA_PX:
        stats["zero_area_mask"] += 1
        return None

    # 5. Recompute bbox
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    x_min = max(0, min(x_coords))
    y_min = max(0, min(y_coords))
    x_max = min(img_w, max(x_coords))
    y_max = min(img_h, max(y_coords))
    w = x_max - x_min
    h = y_max - y_min

    if w < MIN_BBOX_SIDE_PX or h < MIN_BBOX_SIDE_PX:
        stats["invalid_bbox"] += 1
        return None

    # 6. Optional: filter very small objects relative to image
    img_area = img_w * img_h
    if img_area > 0 and (area / img_area) < MIN_OBJECT_AREA_RATIO:
        stats["tiny_object"] += 1
        return None

    new_polygon = [c for p in points for c in p]
    return {
        "id": ann["id"],
        "image_id": ann["image_id"],
        "category_id": name_to_new_id[cat_name],
        "segmentation": [new_polygon],
        "area": area,
        "bbox": [x_min, y_min, w, h],
        "iscrowd": 0,
    }


# ============================================================================
# Main cleaning pipeline
# ============================================================================

def run_cleaning(
    images: List[Dict],
    annotations: List[Dict],
    cat_id_to_name: Dict[int, str],
) -> Tuple[List[Dict], List[Dict], Dict[str, Any]]:
    """
    Run full cleaning pipeline. Returns (cleaned_images, cleaned_annotations, report_dict).
    """
    name_to_new_id = {name: i + 1 for i, name in enumerate(TARGET_CLASSES)}
    img_by_id = {img["id"]: img for img in images}
    anns_by_img = {}
    for ann in annotations:
        img_id = ann["image_id"]
        if img_id not in anns_by_img:
            anns_by_img[img_id] = []
        anns_by_img[img_id].append(ann)

    stats = {
        "non_target_class": 0,
        "empty_segmentation": 0,
        "invalid_polygon": 0,
        "zero_area_mask": 0,
        "invalid_bbox": 0,
        "outside_image": 0,
        "tiny_object": 0,
    }

    # List all classes in raw data
    raw_classes = set()
    for ann in annotations:
        raw_classes.add(ann.get("category_name", "unknown"))

    print(f"[DEBUG] Raw classes in data: {len(raw_classes)} classes")
    print(f"[DEBUG] Target classes (keeping): {len(TARGET_CLASSES)} classes")
    print(f"[DEBUG] Filtering and cleaning annotations...")

    cleaned_annotations: List[Dict] = []
    new_ann_id = 0
    for ann in annotations:
        img_id = ann["image_id"]
        img = img_by_id.get(img_id)
        if not img:
            continue
        img_w, img_h = img["width"], img["height"]
        cleaned = clean_annotation(ann, img_w, img_h, name_to_new_id, stats)
        if cleaned is not None:
            cleaned["id"] = new_ann_id
            new_ann_id += 1
            cleaned_annotations.append(cleaned)

    # Keep only images that have at least one valid annotation
    img_ids_with_anns = {a["image_id"] for a in cleaned_annotations}
    cleaned_images = [img for img in images if img["id"] in img_ids_with_anns]
    dropped_images = len(images) - len(cleaned_images)

    # Remap image_id if we drop images? No - we keep image ids, just drop images with zero anns
    # Actually we should keep all images that had annotations before - we only drop images
    # that have ZERO valid annotations after cleaning. So cleaned_images = images that have >=1 cleaned ann.
    # But wait - if we drop an image, we need to remove its annotations too. Actually we already
    # only have annotations for images that exist. The question is: do we keep images with 0 annotations?
    # For training, images with no annotations are typically skipped or used as negative samples.
    # The plan says "Images removed (if any have zero valid annotations)". So we drop images with 0 valid anns.
    # We already did that: cleaned_images = [img for img in images if img["id"] in img_ids_with_anns]
    # But then we need to filter cleaned_annotations to only those in cleaned_images - we already did that
    # since we only add to cleaned_annotations when we have a valid ann, and img_id is from images.
    # So we're good.

    report = {
        "input_images": len(images),
        "input_annotations": len(annotations),
        "raw_classes": sorted(raw_classes),
        "target_classes": TARGET_CLASSES,
        "stats": stats,
        "output_images": len(cleaned_images),
        "output_annotations": len(cleaned_annotations),
        "dropped_images": dropped_images,
    }
    return cleaned_images, cleaned_annotations, report


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Clean wound dataset for Mask R-CNN")
    parser.add_argument(
        "--input-mode",
        choices=["cvat", "coco"],
        default="cvat",
        help="Input format: cvat (task folders) or coco (single JSON)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(PROJECT_ROOT / "data"),
        help="Data root (for cvat mode)",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="",
        help="Path to COCO JSON (for coco mode)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "data" / "annotations_cleaned.json"),
        help="Output cleaned COCO JSON path",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=str(PROJECT_ROOT / "data" / "cleaning_report.txt"),
        help="Output report path",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Regenerate train/val/test splits after cleaning",
    )
    parser.add_argument(
        "--split-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "splits"),
        help="Directory for split files",
    )
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        default=[0.7, 0.15, 0.15],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train/val/test ratios",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        # Resolve relative to project root
        rel = str(data_root).replace("\\", "/").lstrip("/")
        while rel.startswith("../"):
            rel = rel[3:]
        data_root = (PROJECT_ROOT / (rel or "data")).resolve()

    if args.input_mode == "cvat":
        if not data_root.exists():
            print(f"[ERROR] Data root not found: {data_root}")
            sys.exit(1)
        images, annotations, cat_id_to_name = load_from_cvat(data_root)
    else:
        input_path = Path(args.input_file) if args.input_file else data_root / "annotations.json"
        if not input_path.is_absolute():
            input_path = (PROJECT_ROOT / input_path).resolve()
        if not input_path.exists():
            print(f"[ERROR] Input file not found: {input_path}")
            sys.exit(1)
        images, annotations, cat_id_to_name = load_from_coco(input_path)

    if not images or not annotations:
        print("[ERROR] No images or annotations loaded")
        sys.exit(1)

    cleaned_images, cleaned_annotations, report = run_cleaning(images, annotations, cat_id_to_name)

    categories = [{"id": i + 1, "name": name} for i, name in enumerate(TARGET_CLASSES)]
    output_data = {
        "images": cleaned_images,
        "annotations": cleaned_annotations,
        "categories": categories,
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (PROJECT_ROOT / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved cleaned annotations to {output_path}")

    # Write report
    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = (PROJECT_ROOT / report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_lines = [
        "=== Dataset Cleaning Report ===",
        f"Input: {args.data_root if args.input_mode == 'cvat' else args.input_file}",
        f"Target classes: {len(TARGET_CLASSES)}",
        "",
        "Before cleaning:",
        f"  Images: {report['input_images']}",
        f"  Annotations: {report['input_annotations']}",
        f"  Classes in data: {report['raw_classes']}",
        "",
        "Removed:",
        f"  - Non-target classes: {report['stats']['non_target_class']} annotations",
        f"  - Empty segmentation: {report['stats']['empty_segmentation']}",
        f"  - Invalid polygon: {report['stats']['invalid_polygon']}",
        f"  - Zero-area mask: {report['stats']['zero_area_mask']}",
        f"  - Invalid bbox: {report['stats']['invalid_bbox']}",
        f"  - Outside image (unrepairable): {report['stats']['outside_image']}",
        f"  - Tiny object (area ratio): {report['stats']['tiny_object']}",
        "",
        "After cleaning:",
        f"  Images: {report['output_images']}",
        f"  Annotations: {report['output_annotations']}",
        f"  Images dropped (zero valid anns): {report['dropped_images']}",
        f"  Classes: 8 (IDs 1-8)",
        "",
        f"Output: {output_path}",
    ]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"[OK] Saved report to {report_path}")

    # Print report to stdout (avoid Unicode errors on Windows console)
    print()
    for line in report_lines:
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode("ascii"))

    # Regenerate splits if requested
    if args.split and cleaned_images:
        import random
        random.seed(42)
        shuffled = list(cleaned_images)
        random.shuffle(shuffled)
        n = len(shuffled)
        tr, va, te = args.split_ratios
        n_train = int(n * tr)
        n_val = int(n * va)
        splits = {
            "train": shuffled[:n_train],
            "val": shuffled[n_train : n_train + n_val],
            "test": shuffled[n_train + n_val :],
        }
        split_dir = Path(args.split_dir)
        if not split_dir.is_absolute():
            split_dir = (PROJECT_ROOT / split_dir).resolve()
        split_dir.mkdir(parents=True, exist_ok=True)
        for split_name, split_images in splits.items():
            split_ids = {img["id"] for img in split_images}
            split_anns = [a for a in cleaned_annotations if a["image_id"] in split_ids]
            split_data = {
                "images": split_images,
                "annotations": split_anns,
                "categories": categories,
            }
            out_file = split_dir / f"{split_name}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(split_data, f, indent=2, ensure_ascii=False)
            print(f"[OK] {split_name}: {len(split_images)} images, {len(split_anns)} annotations -> {out_file}")


if __name__ == "__main__":
    main()
