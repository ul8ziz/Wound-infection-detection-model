"""
Combined YOLO11m-seg + U-Net++ inference with configurable ROI, upscale, and post-processing.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from pipeline_utils import IMAGENET_MEAN, IMAGENET_STD

from .config import CombinedInferenceConfig, combined_config_from_dict
from .geometry import tight_bbox_from_binary_mask, xyxy_to_padded_roi
from .marker import calculate_pixels_per_cm_from_marker
from .postprocess import (
    apply_postprocess_chain,
    filter_min_area,
    resolve_postprocess,
    resolve_refinement_postprocess,
)

logger = logging.getLogger(__name__)


def _unet_probs_tta(
    unet_model: nn.Module,
    crop_tensor: torch.Tensor,
    enable_tta: bool,
) -> np.ndarray:
    pred = torch.sigmoid(unet_model(crop_tensor))
    if enable_tta:
        flipped = torch.flip(crop_tensor, dims=[-1])
        pred_flip = torch.sigmoid(unet_model(flipped))
        pred_flip = torch.flip(pred_flip, dims=[-1])
        pred = (pred + pred_flip) * 0.5
    return pred.squeeze().cpu().numpy()


def _mask_nms_instances(
    instances: List[Dict[str, Any]],
    iou_thresh: float,
) -> List[Dict[str, Any]]:
    if len(instances) <= 1:
        return instances

    order = sorted(range(len(instances)), key=lambda i: -instances[i]["score"])
    keep: List[Dict[str, Any]] = []
    used = [False] * len(instances)

    for oi in order:
        if used[oi]:
            continue
        cur = instances[oi]
        keep.append(cur)
        used[oi] = True
        m_i = cur["mask"].astype(float)
        area_i = m_i.sum()
        for j in order:
            if used[j] or j == oi:
                continue
            m_j = instances[j]["mask"].astype(float)
            inter = (m_i * m_j).sum()
            area_j = m_j.sum()
            iou = inter / max(area_i + area_j - inter, 1e-6)
            if iou >= iou_thresh:
                used[j] = True
    return keep


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = (a > 0).astype(float)
    b = (b > 0).astype(float)
    inter = (a * b).sum()
    u = a.sum() + b.sum() - inter
    return float(inter / max(u, 1e-6))


def _merge_overlapping_instances(
    instances: List[Dict[str, Any]],
    iou_thresh: float,
) -> List[Dict[str, Any]]:
    """Greedily merge instances whose masks overlap above ``iou_thresh`` (pixel IoU)."""
    if len(instances) <= 1:
        return instances

    sorted_inst = sorted(instances, key=lambda x: -x["score"])
    merged: List[Dict[str, Any]] = []
    for ins in sorted_inst:
        placed = False
        for m in merged:
            if _mask_iou(ins["mask"], m["mask"]) >= iou_thresh:
                m["mask"] = np.maximum(m["mask"], ins["mask"]).astype(np.uint8)
                m["score"] = max(m["score"], ins["score"])
                pr = m["padded_roi"]
                q = ins["padded_roi"]
                m["padded_roi"] = [
                    min(pr[0], q[0]),
                    min(pr[1], q[1]),
                    max(pr[2], q[2]),
                    max(pr[3], q[3]),
                ]
                yl = m["yolo_xyxy"]
                z = ins["yolo_xyxy"]
                m["yolo_xyxy"] = [
                    min(yl[0], z[0]),
                    min(yl[1], z[1]),
                    max(yl[2], z[2]),
                    max(yl[3], z[3]),
                ]
                placed = True
                break
        if not placed:
            merged.append(
                {
                    "mask": ins["mask"].copy(),
                    "yolo_xyxy": list(ins["yolo_xyxy"]),
                    "padded_roi": list(ins["padded_roi"]),
                    "score": ins["score"],
                },
            )
    return merged


def _select_wound_indices(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    conf_thresh: float,
    wound_class_id: int,
    strategy: str,
    img_hw: Optional[Tuple[int, int]] = None,
) -> List[int]:
    """Return wound box indices according to *strategy*.

    Single-box strategies return a list with at most one index.
    Multi-box strategies return all valid indices (sorted).
    """
    idxs: List[int] = []
    for i in range(len(boxes_xyxy)):
        if int(classes[i]) != wound_class_id:
            continue
        if scores[i] < conf_thresh:
            continue
        idxs.append(i)

    if not idxs:
        return []

    def _area(i: int) -> float:
        return float(
            (boxes_xyxy[i, 2] - boxes_xyxy[i, 0])
            * (boxes_xyxy[i, 3] - boxes_xyxy[i, 1])
        )

    if strategy == "highest_conf_single":
        return [max(idxs, key=lambda i: scores[i])]

    if strategy == "largest_area_single":
        return [max(idxs, key=_area)]

    if strategy == "confidence_times_area":
        return [max(idxs, key=lambda i: float(scores[i]) * _area(i))]

    if strategy == "closest_to_center":
        if img_hw is None:
            return [max(idxs, key=lambda i: scores[i])]
        ih, iw = img_hw
        cx_img, cy_img = iw / 2.0, ih / 2.0
        def _dist(i: int) -> float:
            bx = (boxes_xyxy[i, 0] + boxes_xyxy[i, 2]) / 2.0
            by = (boxes_xyxy[i, 1] + boxes_xyxy[i, 3]) / 2.0
            return (bx - cx_img) ** 2 + (by - cy_img) ** 2
        return [min(idxs, key=_dist)]

    if strategy == "largest_area":
        idxs.sort(key=lambda i: -_area(i))
        return idxs

    if strategy == "all_above_thresh":
        idxs.sort(key=lambda i: -scores[i])
        return idxs

    idxs.sort(key=lambda i: -scores[i])
    return idxs


def _predict_roi_probs(
    unet_model: nn.Module,
    image_rgb: np.ndarray,
    yolo_xyxy: np.ndarray,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    unet_hw: Tuple[int, int],
    enable_tta: bool,
    roi_padding: float,
) -> Optional[Dict[str, Any]]:
    img_h, img_w = image_rgb.shape[:2]
    x1, y1, x2, y2 = float(yolo_xyxy[0]), float(yolo_xyxy[1]), float(yolo_xyxy[2]), float(yolo_xyxy[3])
    cx1, cy1, cx2, cy2 = xyxy_to_padded_roi(x1, y1, x2, y2, img_w, img_h, roi_padding)

    crop = image_rgb[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None

    uh, uw = unet_hw[0], unet_hw[1]
    crop_resized = cv2.resize(crop, (uw, uh), interpolation=cv2.INTER_LINEAR)
    crop_tensor = torch.from_numpy(crop_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    crop_tensor = (crop_tensor.to(device) - mean) / std

    probs = _unet_probs_tta(unet_model, crop_tensor, enable_tta)
    crop_h, crop_w = cy2 - cy1, cx2 - cx1
    prob_up = cv2.resize(probs.astype(np.float32), (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
    return {
        "probs_small": probs.astype(np.float32),
        "prob_up": prob_up,
        "crop_box": (cx1, cy1, cx2, cy2),
    }


def _full_prob_canvas(
    prob_up: np.ndarray,
    crop_box: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
) -> np.ndarray:
    img_h, img_w = image_shape
    canvas = np.zeros((img_h, img_w), dtype=np.float32)
    x1, y1, x2, y2 = crop_box
    canvas[y1:y2, x1:x2] = prob_up.astype(np.float32)
    return canvas


def _multi_scale_weights(cfg: CombinedInferenceConfig, n_scales: int) -> List[float]:
    weights = [float(w) for w in cfg.multi_scale_weights]
    if len(weights) != n_scales or sum(weights) <= 0:
        return [1.0] * n_scales
    return weights


def _fuse_multiscale_probabilities(
    predictions: List[Dict[str, Any]],
    image_shape: Tuple[int, int],
    cfg: CombinedInferenceConfig,
) -> Tuple[np.ndarray, List[float]]:
    if len(predictions) == 1:
        pred = predictions[0]
        return _full_prob_canvas(pred["prob_up"], pred["crop_box"], image_shape), list(pred["crop_box"])

    if cfg.multi_scale_fusion == "stability_select":
        masks = []
        canvases = []
        for pred in predictions:
            canvas = _full_prob_canvas(pred["prob_up"], pred["crop_box"], image_shape)
            canvases.append(canvas)
            masks.append((canvas >= cfg.unet_mask_thresh).astype(np.uint8))
        mean_ious: List[float] = []
        for i, mask_i in enumerate(masks):
            if len(masks) == 1:
                mean_ious.append(1.0)
                continue
            pair_scores = []
            for j, mask_j in enumerate(masks):
                if i == j:
                    continue
                pair_scores.append(_mask_iou(mask_i, mask_j))
            mean_ious.append(float(np.mean(pair_scores)) if pair_scores else 1.0)
        best_idx = int(np.argmax(mean_ious))
        crop_box = predictions[best_idx]["crop_box"]
        return canvases[best_idx], [float(crop_box[0]), float(crop_box[1]), float(crop_box[2]), float(crop_box[3])]

    weights = _multi_scale_weights(cfg, len(predictions))
    prob_sum = np.zeros(image_shape, dtype=np.float32)
    weight_sum = np.zeros(image_shape, dtype=np.float32)
    x1s: List[int] = []
    y1s: List[int] = []
    x2s: List[int] = []
    y2s: List[int] = []

    for pred, weight in zip(predictions, weights):
        crop_box = pred["crop_box"]
        x1, y1, x2, y2 = crop_box
        prob_sum[y1:y2, x1:x2] += pred["prob_up"].astype(np.float32) * float(weight)
        weight_sum[y1:y2, x1:x2] += float(weight)
        x1s.append(x1)
        y1s.append(y1)
        x2s.append(x2)
        y2s.append(y2)

    fused = np.divide(prob_sum, np.maximum(weight_sum, 1e-6))
    return fused, [float(min(x1s)), float(min(y1s)), float(max(x2s)), float(max(y2s))]


def _refine_one_roi(
    unet_model: nn.Module,
    image_rgb: np.ndarray,
    yolo_xyxy: np.ndarray,
    score: float,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    unet_hw: Tuple[int, int],
    cfg: CombinedInferenceConfig,
) -> Optional[Dict[str, Any]]:
    img_h, img_w = image_rgb.shape[:2]
    x1, y1, x2, y2 = float(yolo_xyxy[0]), float(yolo_xyxy[1]), float(yolo_xyxy[2]), float(yolo_xyxy[3])
    post_ops = resolve_postprocess(cfg.postprocess_preset, cfg.postprocess)
    refinement_ops = resolve_refinement_postprocess(cfg.refinement_postprocess)

    if cfg.multi_scale_refinement:
        roi_paddings = cfg.multi_scale_roi_paddings or [cfg.roi_padding]
    else:
        roi_paddings = [cfg.roi_padding]

    predictions: List[Dict[str, Any]] = []
    for roi_padding in roi_paddings:
        pred = _predict_roi_probs(
            unet_model=unet_model,
            image_rgb=image_rgb,
            yolo_xyxy=yolo_xyxy,
            device=device,
            mean=mean,
            std=std,
            unet_hw=unet_hw,
            enable_tta=cfg.enable_tta,
            roi_padding=float(roi_padding),
        )
        if pred is not None:
            predictions.append(pred)

    if not predictions:
        return None

    padded_roi = [
        float(min(pred["crop_box"][0] for pred in predictions)),
        float(min(pred["crop_box"][1] for pred in predictions)),
        float(max(pred["crop_box"][2] for pred in predictions)),
        float(max(pred["crop_box"][3] for pred in predictions)),
    ]

    if (not cfg.multi_scale_refinement) and cfg.mask_upscale != "linear_probs":
        pred0 = predictions[0]
        x1r, y1r, x2r, y2r = pred0["crop_box"]
        crop_h, crop_w = y2r - y1r, x2r - x1r
        m_small = (pred0["probs_small"] >= cfg.unet_mask_thresh).astype(np.uint8)
        m_roi = cv2.resize(m_small, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
        full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        full_mask[y1r:y2r, x1r:x2r] = m_roi
    else:
        fused_probs_full, padded_roi = _fuse_multiscale_probabilities(predictions, (img_h, img_w), cfg)
        full_mask = (fused_probs_full >= cfg.unet_mask_thresh).astype(np.uint8)

    full_mask = apply_postprocess_chain(full_mask, post_ops)
    if refinement_ops:
        full_mask = apply_postprocess_chain(full_mask, refinement_ops)

    if cfg.min_mask_area > 0:
        full_mask = filter_min_area(full_mask, cfg.min_mask_area)

    return {
        "mask": full_mask,
        "yolo_xyxy": [x1, y1, x2, y2],
        "padded_roi": padded_roi,
        "score": float(score),
    }


@torch.no_grad()
def combined_inference(
    yolo_model,
    unet_model: nn.Module,
    image_path: str,
    device: torch.device,
    config: dict,
    enable_tta: Optional[bool] = None,
    cfg: Optional[CombinedInferenceConfig] = None,
    yolo_result=None,
) -> Dict[str, Any]:
    cfg = cfg or combined_config_from_dict(config)
    if enable_tta is not None:
        cfg = replace(cfg, enable_tta=enable_tta)

    unet_cfg = config.get("unet", {})
    unet_size = tuple(unet_cfg.get("input_size", [256, 256]))

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        return {"error": f"Could not load {image_path}"}
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = image_rgb.shape[:2]

    if yolo_result is None:
        yolo_results = yolo_model(
            image_path,
            conf=cfg.yolo_min_conf_inference,
            verbose=False,
        )
    else:
        yolo_results = yolo_result if isinstance(yolo_result, (list, tuple)) else [yolo_result]

    empty = {
        "boxes": [],
        "masks": [],
        "scores": [],
        "boxes_yolo_xyxy": [],
        "boxes_padded_roi": [],
        "boxes_mask_tight_xyxy": [],
        "image_shape": (img_h, img_w),
        "pixels_per_cm": None,
        "coco_bbox_mode": cfg.coco_bbox_mode,
    }

    if not yolo_results or len(yolo_results) == 0:
        return empty

    result = yolo_results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return empty

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    marker_ppcm = calculate_pixels_per_cm_from_marker(
        result,
        marker_class_id=1,
        marker_real_cm=cfg.marker_real_cm,
    )

    wound_class_id = 0
    indices = _select_wound_indices(
        boxes_xyxy,
        scores,
        classes,
        cfg.yolo_conf_thresh,
        wound_class_id,
        cfg.bbox_selection_strategy,
        img_hw=(img_h, img_w),
    )

    if not indices:
        out = dict(empty)
        out["pixels_per_cm"] = marker_ppcm
        return out

    unet_model.eval()
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(device)

    instances: List[Dict[str, Any]] = []
    for i in indices:
        inst = _refine_one_roi(
            unet_model,
            image_rgb,
            boxes_xyxy[i],
            float(scores[i]),
            device,
            mean,
            std,
            unet_size,
            cfg,
        )
        if inst is not None and inst["mask"].sum() >= 1:
            instances.append(inst)

    if not instances:
        out = dict(empty)
        out["pixels_per_cm"] = marker_ppcm
        return out

    if cfg.bbox_selection_strategy == "merge_overlapping":
        instances = _merge_overlapping_instances(instances, cfg.merge_iou_thresh)

    instances = _mask_nms_instances(instances, cfg.merge_iou_thresh)

    masks = [x["mask"] for x in instances]
    scores_o = [x["score"] for x in instances]
    boxes_yolo = [x["yolo_xyxy"] for x in instances]
    boxes_padded = [x["padded_roi"] for x in instances]
    mask_tight = []
    for m in masks:
        x1, y1, x2, y2 = tight_bbox_from_binary_mask(m)
        mask_tight.append([x1, y1, x2, y2])

    coco_boxes: List[List[float]] = []
    for i in range(len(masks)):
        mode = cfg.coco_bbox_mode
        if mode == "yolo_xyxy":
            coco_boxes.append(boxes_yolo[i])
        elif mode == "mask_tight":
            coco_boxes.append(mask_tight[i])
        else:
            coco_boxes.append(boxes_padded[i])

    return {
        "masks": masks,
        "scores": scores_o,
        "boxes": coco_boxes,
        "boxes_yolo_xyxy": boxes_yolo,
        "boxes_padded_roi": boxes_padded,
        "boxes_mask_tight_xyxy": mask_tight,
        "image_shape": (img_h, img_w),
        "pixels_per_cm": marker_ppcm,
        "coco_bbox_mode": cfg.coco_bbox_mode,
    }
