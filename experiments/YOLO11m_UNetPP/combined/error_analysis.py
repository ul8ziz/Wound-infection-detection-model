"""Rule-based error taxonomy for combined YOLO + U-Net++ vs COCO GT."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from experiment_io import get_combined_dirs, get_unet_best_checkpoint_path
from .config import combined_config_from_dict
from .inference import combined_inference


def _bbox_iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    bb = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = aa + bb - inter
    return float(inter / max(union, 1e-6))


def _centroid(mask: np.ndarray) -> Tuple[float, float]:
    m = (mask > 0).astype(np.float32)
    s = m.sum()
    if s < 1e-6:
        return 0.0, 0.0
    ys, xs = np.where(m > 0)
    return float(xs.mean()), float(ys.mean())


def classify_case(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    gt_bbox_xywh: Optional[List[float]],
    yolo_xyxy: Optional[List[float]],
    pred_has_detection: bool,
) -> List[str]:
    labels: List[str] = []
    gt_area = float((gt_mask > 0).sum())
    pr_area = float((pred_mask > 0).sum())

    if not pred_has_detection or pr_area < 1:
        labels.append("missed_detection")
        return labels

    smooth = 1e-6
    inter = ((gt_mask > 0) & (pred_mask > 0)).sum()
    union = ((gt_mask > 0) | (pred_mask > 0)).sum()
    iou = (inter + smooth) / (union - inter + smooth)
    dice = (2 * inter + smooth) / (gt_area + pr_area + smooth)

    if gt_bbox_xywh is not None and yolo_xyxy is not None:
        gx, gy, gw, gh = gt_bbox_xywh
        g_box = [gx, gy, gx + gw, gy + gh]
        biou = _bbox_iou_xyxy(g_box, yolo_xyxy)
        if biou < 0.3:
            labels.append("poor_bbox_localization")
        elif biou < 0.5:
            labels.append("moderate_bbox_iou")

    gcc = _centroid(gt_mask)
    pcc = _centroid(pred_mask)
    shift = float(np.hypot(gcc[0] - pcc[0], gcc[1] - pcc[1]))
    if shift > 40:
        labels.append("shifted_roi_or_mask")

    if pr_area < 0.5 * gt_area and iou < 0.3:
        labels.append("under_segmentation")
    if pr_area > 1.5 * max(gt_area, 1.0) and iou < 0.4:
        labels.append("over_segmentation")

    n_cc = cv2.connectedComponents((pred_mask > 0).astype(np.uint8))[0] - 1
    if n_cc > 2:
        labels.append("fragmented_mask")

    if gt_area < 1e-6 and pr_area > 50:
        labels.append("false_positive_mask")

    if iou < 0.25 and dice < 0.3:
        labels.append("boundary_or_alignment_error")

    if not labels:
        labels.append("ok_or_minor")

    return labels


def run_error_analysis(
    config: dict,
    script_dir: Path,
    split: str = "val",
    max_images: int = 0,
    out_dir: Optional[Path] = None,
) -> None:
    from train_model import build_unet_model, build_yolo_model, load_unet_checkpoint
    from pipeline_utils import get_device

    cfg_inf = combined_config_from_dict(config)
    combined_dirs = get_combined_dirs(script_dir, config)
    out = out_dir or (combined_dirs["results"] / "error_analysis")
    out.mkdir(parents=True, exist_ok=True)
    qual = out / "qualitative"
    qual.mkdir(exist_ok=True)

    device = get_device()
    yolo_best = script_dir / "checkpoints" / "yolo" / "best.pt"
    unet_best = get_unet_best_checkpoint_path(script_dir, config)
    if not yolo_best.exists() or not unet_best.exists():
        print("Missing checkpoints.")
        return

    yolo_model = build_yolo_model(str(yolo_best))
    unet_model = build_unet_model(config)
    load_unet_checkpoint(unet_model, unet_best, device)
    unet_model.to(device)
    unet_model.eval()

    project_root = script_dir.parent.parent
    ann_key = "ann_val" if split == "val" else "ann_test"
    ann_path = (project_root / config[ann_key]).resolve()
    data_root = (project_root / config["data_root"]).resolve()

    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    img_lookup = {img["id"]: img for img in coco["images"]}
    cat_ids = {c["id"] for c in coco["categories"]}
    img_anns: Dict[int, list] = {}
    for ann in coco["annotations"]:
        if ann["category_id"] in cat_ids:
            img_anns.setdefault(ann["image_id"], []).append(ann)

    rows: List[Dict[str, Any]] = []
    n = 0
    for img_id, img_info in img_lookup.items():
        if max_images and n >= max_images:
            break
        img_path = str(data_root / img_info["file_name"])
        if not Path(img_path).exists():
            continue

        orig_h, orig_w = img_info["height"], img_info["width"]
        gt_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        gt_bbox_xywh: Optional[List[float]] = None
        for ann in img_anns.get(img_id, []):
            if gt_bbox_xywh is None and ann.get("bbox"):
                gt_bbox_xywh = [float(x) for x in ann["bbox"]]
            for seg in ann.get("segmentation", []):
                if len(seg) < 6:
                    continue
                poly = np.array(seg, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(gt_mask, [poly], 1)

        pred = combined_inference(
            yolo_model, unet_model, img_path, device, config, cfg=cfg_inf,
        )
        pred_has = bool(pred.get("masks"))
        combined_pred = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for m in pred.get("masks", []):
            if m.shape == (orig_h, orig_w):
                combined_pred = np.maximum(combined_pred, m)

        yolo_xy = pred["boxes_yolo_xyxy"][0] if pred.get("boxes_yolo_xyxy") else None

        labels = classify_case(
            gt_mask,
            combined_pred,
            gt_bbox_xywh,
            yolo_xy,
            pred_has,
        )

        row = {
            "file_name": img_info["file_name"],
            "labels": ";".join(labels),
            "n_pred_instances": len(pred.get("masks", [])),
        }
        rows.append(row)

        for lab in labels:
            if lab == "ok_or_minor":
                continue
            sub = qual / lab.replace(" ", "_")
            sub.mkdir(exist_ok=True)
            vis = cv2.imread(img_path)
            if vis is None:
                continue
            ov = vis.copy()
            ov[combined_pred > 0] = (0, 255, 0)
            gt_vis = vis.copy()
            gt_vis[gt_mask > 0] = (0, 128, 255)
            side = np.hstack([vis, ov, gt_vis])
            cv2.imwrite(str(sub / f"{Path(img_info['file_name']).stem}.png"), side)

        n += 1

    csv_path = out / f"error_summary_{split}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file_name", "labels", "n_pred_instances"])
        w.writeheader()
        w.writerows(rows)

    # markdown report
    from collections import Counter

    cnt: Counter = Counter()
    for r in rows:
        for lab in r["labels"].split(";"):
            cnt[lab] += 1

    lines = [
        "# Combined pipeline error analysis\n",
        f"Split: **{split}**  \n",
        "\n## Counts by label\n",
    ]
    for k, v in cnt.most_common():
        lines.append(f"- {k}: {v}\n")
    lines.append(f"\nCSV: `{csv_path}`\n")

    with open(out / f"error_report_{split}.md", "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Wrote {csv_path} and error_report_{split}.md ({n} images)")
