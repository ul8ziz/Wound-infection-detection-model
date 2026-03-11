"""
Visualize Cleaned Dataset
=========================

Samples images from annotations_cleaned.json and draws polygons + bboxes with class labels.
Saves to data/cleaned_visualizations/ for manual inspection.

Usage:
    cd scripts
    python visualize_cleaned_dataset.py
    python visualize_cleaned_dataset.py --input ../data/annotations_cleaned.json --num-samples 10
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# Distinct colors for 8 classes (BGR for OpenCV)
CLASS_COLORS = [
    (255, 0, 0),    # BGR: red
    (0, 255, 0),    # green
    (0, 0, 255),    # blue
    (255, 255, 0),  # cyan
    (255, 0, 255),  # magenta
    (0, 255, 255),  # yellow
    (128, 0, 255),  # purple
    (255, 128, 0),  # orange
]


def main():
    parser = argparse.ArgumentParser(description="Visualize cleaned wound dataset")
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "data" / "annotations_cleaned.json"),
        help="Path to cleaned COCO JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "cleaned_visualizations"),
        help="Output directory for visualization images",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of sample images to visualize",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = (PROJECT_ROOT / input_path).resolve()

    if not input_path.exists():
        print(f"[ERROR] File not found: {input_path}")
        return 1

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]
    categories = {c["id"]: c["name"] for c in data["categories"]}

    anns_by_img = {}
    for ann in annotations:
        img_id = ann["image_id"]
        if img_id not in anns_by_img:
            anns_by_img[img_id] = []
        anns_by_img[img_id].append(ann)

    # Sample images that have annotations
    img_ids_with_anns = [img["id"] for img in images if img["id"] in anns_by_img]
    if not img_ids_with_anns:
        print("[ERROR] No images with annotations")
        return 1

    random.seed(args.seed)
    n = min(args.num_samples, len(img_ids_with_anns))
    sampled_ids = random.sample(img_ids_with_anns, n)

    img_by_id = {img["id"]: img for img in images}
    data_root = input_path.parent
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[DEBUG] Visualizing {n} images -> {output_dir}")

    for img_id in sampled_ids:
        img_info = img_by_id[img_id]
        img_path = data_root / img_info["file_name"]
        if not img_path.exists():
            print(f"[WARN] Image not found: {img_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Could not load: {img_path}")
            continue

        h, w = img.shape[:2]
        overlay = img.copy()

        for ann in anns_by_img[img_id]:
            cat_id = ann["category_id"]
            cat_name = categories.get(cat_id, f"id_{cat_id}")
            color = CLASS_COLORS[(cat_id - 1) % len(CLASS_COLORS)]

            # Draw polygon
            for seg in ann.get("segmentation", []):
                if not isinstance(seg, list) or len(seg) < 6:
                    continue
                poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                cv2.polylines(overlay, [poly], True, color, 2)
                cv2.fillPoly(overlay, [poly], color)

            # Draw bbox
            bbox = ann.get("bbox", [])
            if len(bbox) == 4:
                x, y, bw, bh = [int(v) for v in bbox]
                cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color, 2)

            # Label (above bbox)
            if len(bbox) == 4:
                x, y = int(bbox[0]), int(bbox[1])
                (tw, th), _ = cv2.getTextSize(cat_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(overlay, (x, y - th - 4), (x + tw, y), color, -1)
                cv2.putText(
                    overlay, cat_name, (x, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                )

        # Blend for transparency
        alpha = 0.4
        result = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        out_name = f"img_{img_id}_{img_path.stem}.jpg"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), result)
        print(f"  Saved: {out_name}")

    print(f"[OK] Visualizations saved to {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
