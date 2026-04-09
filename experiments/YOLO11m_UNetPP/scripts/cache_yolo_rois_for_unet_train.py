from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from pipeline_utils import get_device, load_config
from train_model import build_yolo_model


def resolve_image_path(root: Path, file_name: str) -> Path:
    candidate = root / file_name
    if candidate.is_file():
        return candidate
    alt = root / "augmented" / file_name
    if alt.is_file():
        return alt
    return candidate


def bbox_xyxy_from_coco_xywh(bbox: List[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    return float(x), float(y), float(x + w), float(y + h)


def bbox_iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def build_cache(
    config: dict,
    ann_key: str,
    output_path: Path,
    yolo_weights: Path,
    min_iou: float,
    yolo_conf: float,
) -> dict:
    device = get_device()
    model = build_yolo_model(str(yolo_weights))

    ann_path = (PROJECT_ROOT / config[ann_key]).resolve()
    if ann_key == "ann_train":
        data_root = (PROJECT_ROOT / config.get("data_root_train", config["data_root"])).resolve()
    else:
        data_root = (PROJECT_ROOT / config["data_root"]).resolve()

    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    wound_cat_ids = {cat["id"] for cat in coco["categories"] if cat["name"] == "wound"}
    anns_by_image: Dict[int, List[dict]] = {}
    for ann in coco["annotations"]:
        if ann["category_id"] in wound_cat_ids:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

    matches: Dict[str, dict] = {}
    n_images = 0
    n_with_boxes = 0
    n_fallback = 0

    for image_id, anns in anns_by_image.items():
        img_info = images[image_id]
        img_path = resolve_image_path(data_root, img_info["file_name"])
        if not img_path.is_file():
            continue

        n_images += 1
        result = model(str(img_path), conf=yolo_conf, verbose=False)[0]

        pred_boxes: List[Tuple[Tuple[float, float, float, float], float]] = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy()
            for box, score, cls_id in zip(boxes_xyxy, scores, classes):
                if int(cls_id) != 0:
                    continue
                pred_boxes.append(((float(box[0]), float(box[1]), float(box[2]), float(box[3])), float(score)))
        if pred_boxes:
            n_with_boxes += 1

        for ann in anns:
            ann_id = ann.get("id")
            if ann_id is None:
                continue
            gt_box = bbox_xyxy_from_coco_xywh(ann["bbox"])
            best_box: Optional[Tuple[float, float, float, float]] = None
            best_score = 0.0
            best_iou = 0.0
            for box_xyxy, score in pred_boxes:
                iou = bbox_iou_xyxy(gt_box, box_xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_box = box_xyxy
                    best_score = score

            fallback_gt = best_box is None or best_iou < min_iou
            if fallback_gt:
                n_fallback += 1

            matches[str(ann_id)] = {
                "ann_id": int(ann_id),
                "image_id": int(image_id),
                "file_name": img_info["file_name"],
                "gt_bbox_xyxy": list(gt_box),
                "bbox_xyxy": list(best_box) if best_box is not None else None,
                "score": float(best_score),
                "iou": float(best_iou),
                "fallback_gt": bool(fallback_gt),
            }

    payload = {
        "ann_key": ann_key,
        "annotation_file": str(ann_path),
        "data_root": str(data_root),
        "yolo_weights": str(yolo_weights),
        "yolo_conf": float(yolo_conf),
        "min_iou": float(min_iou),
        "stats": {
            "images_processed": n_images,
            "images_with_wound_boxes": n_with_boxes,
            "annotations_cached": len(matches),
            "fallback_gt_annotations": n_fallback,
        },
        "matches": matches,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache YOLO wound ROIs for U-Net crop training.")
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument("--ann-key", default="ann_train", choices=["ann_train", "ann_val", "ann_test"])
    parser.add_argument("--output", default=str(SCRIPT_DIR / "results" / "roi_cache" / "train_yolo_rois.json"))
    parser.add_argument("--weights", default=str(SCRIPT_DIR / "checkpoints" / "yolo" / "best.pt"))
    parser.add_argument("--min-iou", type=float, default=0.05)
    parser.add_argument("--yolo-conf", type=float, default=0.001)
    args = parser.parse_args()

    config = load_config(args.config)
    payload = build_cache(
        config=config,
        ann_key=args.ann_key,
        output_path=Path(args.output).resolve(),
        yolo_weights=Path(args.weights).resolve(),
        min_iou=float(args.min_iou),
        yolo_conf=float(args.yolo_conf),
    )
    print(json.dumps(payload["stats"], indent=2))
    print(f"Saved ROI cache to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
