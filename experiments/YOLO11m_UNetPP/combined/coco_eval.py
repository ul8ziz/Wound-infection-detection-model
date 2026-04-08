"""COCO AP evaluation for combined pipeline predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config import CombinedInferenceConfig, combined_config_from_dict
from .inference import combined_inference


def _build_coco_results(
    coco_gt,
    data_root: Path,
    yolo_model,
    unet_model,
    device,
    config: dict,
    cfg: CombinedInferenceConfig,
    img_ids: Optional[List[int]] = None,
    yolo_cache: Optional[Dict[str, Any]] = None,
) -> tuple:
    """Returns (bbox_results, segm_results) lists for COCOeval.

    *yolo_cache*: if provided, maps image_path -> YOLO result object to skip
    redundant YOLO forward passes during grid search.
    """
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        return [], []

    img_ids = img_ids or coco_gt.getImgIds()
    bbox_results: List[dict] = []
    segm_results: List[dict] = []

    for img_id in img_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = str(data_root / img_info["file_name"])
        if not Path(img_path).exists():
            continue

        yr = yolo_cache.get(img_path) if yolo_cache else None
        pred = combined_inference(
            yolo_model,
            unet_model,
            img_path,
            device,
            config,
            cfg=cfg,
            yolo_result=yr,
        )
        if "error" in pred or not pred.get("masks"):
            continue

        orig_h, orig_w = img_info["height"], img_info["width"]

        for mask_np, box, score in zip(pred["masks"], pred["boxes"], pred["scores"]):
            if mask_np.shape != (orig_h, orig_w):
                continue

            rle = mask_utils.encode(np.asfortranarray(mask_np.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("utf-8")

            segm_results.append({
                "image_id": img_id,
                "category_id": 1,
                "segmentation": rle,
                "score": float(score),
            })

            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            bbox_results.append({
                "image_id": img_id,
                "category_id": 1,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score),
            })

    return bbox_results, segm_results


def evaluate_combined_coco(
    config: dict,
    script_dir: Path,
    yolo_model,
    unet_model,
    device,
    cfg: Optional[CombinedInferenceConfig] = None,
    ann_key: str = "ann_test",
    write_json: bool = True,
    yolo_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """COCO AP for combined pipeline; bbox geometry follows ``cfg.coco_bbox_mode`` via inference."""
    try:
        from pycocotools.coco import COCO as CocoAPI
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("  [SKIP] pycocotools not available — COCO AP evaluation skipped.")
        return {}

    cfg = cfg or combined_config_from_dict(config)

    project_root = script_dir.parent.parent
    ann_path = str((project_root / config[ann_key]).resolve())
    data_root = (project_root / config["data_root"]).resolve()

    coco_gt = CocoAPI(ann_path)
    if "info" not in coco_gt.dataset:
        coco_gt.dataset["info"] = {}
    if "licenses" not in coco_gt.dataset:
        coco_gt.dataset["licenses"] = []

    bbox_results, segm_results = _build_coco_results(
        coco_gt,
        data_root,
        yolo_model,
        unet_model,
        device,
        config,
        cfg,
        yolo_cache=yolo_cache,
    )

    metrics: Dict[str, float] = {}

    for iou_type, results in [("bbox", bbox_results), ("segm", segm_results)]:
        if not results:
            continue
        coco_dt = coco_gt.loadRes(results)
        evaluator = COCOeval(coco_gt, coco_dt, iouType=iou_type)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

        metrics[f"coco_{iou_type}_AP"] = float(evaluator.stats[0])
        metrics[f"coco_{iou_type}_AP50"] = float(evaluator.stats[1])
        metrics[f"coco_{iou_type}_AP75"] = float(evaluator.stats[2])

    if "coco_bbox_AP50" in metrics and "coco_segm_AP50" in metrics:
        metrics["coco_combined_AP50"] = (
            metrics["coco_bbox_AP50"] + metrics["coco_segm_AP50"]
        ) / 2.0

    if write_json:
        results_dir = script_dir / "results" / "combined"
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "coco_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    return metrics


def pixel_metrics_on_split(
    config: dict,
    script_dir: Path,
    yolo_model,
    unet_model,
    device,
    cfg: Optional[CombinedInferenceConfig] = None,
    ann_key: str = "ann_val",
    yolo_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Mean Dice / IoU (pixel-level merge) on a COCO split.

    Returns both *conditional* metrics (only images with predictions) and
    *full-split* metrics (missed images count as Dice=0, IoU=0).
    """
    cfg = cfg or combined_config_from_dict(config)
    project_root = script_dir.parent.parent
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

    import cv2

    dice_scores: List[float] = []
    iou_scores: List[float] = []
    n_total = 0
    n_missed = 0

    for img_id, img_info in img_lookup.items():
        img_path = str(data_root / img_info["file_name"])
        if not Path(img_path).exists():
            continue
        n_total += 1

        yr = yolo_cache.get(img_path) if yolo_cache else None
        pred = combined_inference(
            yolo_model, unet_model, img_path, device, config, cfg=cfg,
            yolo_result=yr,
        )
        has_pred = not ("error" in pred) and bool(pred.get("masks"))

        orig_h, orig_w = img_info["height"], img_info["width"]
        gt_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for ann in img_anns.get(img_id, []):
            for seg in ann.get("segmentation", []):
                if len(seg) < 6:
                    continue
                poly = np.array(seg, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(gt_mask, [poly], 1)

        if not has_pred:
            n_missed += 1
            dice_scores.append(0.0)
            iou_scores.append(0.0)
            continue

        combined_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for m in pred["masks"]:
            if m.shape == (orig_h, orig_w):
                combined_mask = np.maximum(combined_mask, m)

        smooth = 1e-6
        p_flat = combined_mask.flatten().astype(float)
        t_flat = gt_mask.flatten().astype(float)
        inter = (p_flat * t_flat).sum()
        union = p_flat.sum() + t_flat.sum()
        dice = (2 * inter + smooth) / (union + smooth)
        iou = (inter + smooth) / (union - inter + smooth)
        dice_scores.append(float(dice))
        iou_scores.append(float(iou))

    n_full = max(1, len(dice_scores))
    n_detected = n_full - n_missed
    cond_n = max(1, n_detected)

    full_dice = sum(dice_scores) / n_full
    full_iou = sum(iou_scores) / n_full
    cond_dice = (sum(dice_scores) - 0.0 * n_missed) / cond_n
    cond_iou = (sum(iou_scores) - 0.0 * n_missed) / cond_n

    return {
        "mean_dice": full_dice,
        "mean_iou": full_iou,
        "mean_dice_conditional": cond_dice,
        "mean_iou_conditional": cond_iou,
        "n_images_total": n_total,
        "n_images_evaluated": n_detected,
        "n_images_missed": n_missed,
    }


def balanced_score(metrics: Dict[str, float], weights: Optional[Any] = None) -> float:
    """Seven-term configurable blend: bbox/segm AP50/AP75, combined_AP50, mean_dice."""
    from .config import BalancedScoreWeights

    w = weights or BalancedScoreWeights()
    if isinstance(w, BalancedScoreWeights):
        bb50 = w.bbox_AP50
        sg50 = w.segm_AP50
        c50 = w.combined_AP50
        bb75 = w.bbox_AP75
        sg75 = w.segm_AP75
        md = w.mean_dice
    else:
        bb50 = float(w.get("bbox_AP50", 0.25))
        sg50 = float(w.get("segm_AP50", 0.20))
        c50 = float(w.get("combined_AP50", 0.20))
        bb75 = float(w.get("bbox_AP75", 0.15))
        sg75 = float(w.get("segm_AP75", 0.15))
        md = float(w.get("mean_dice", 0.05))

    return (
        bb50 * metrics.get("coco_bbox_AP50", 0.0)
        + sg50 * metrics.get("coco_segm_AP50", 0.0)
        + c50 * metrics.get("coco_combined_AP50", 0.0)
        + bb75 * metrics.get("coco_bbox_AP75", 0.0)
        + sg75 * metrics.get("coco_segm_AP75", 0.0)
        + md * metrics.get("mean_dice", 0.0)
    )
