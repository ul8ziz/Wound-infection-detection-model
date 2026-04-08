"""
Combined YOLO + U-Net++ inference configuration.

``unet.roi_padding`` in config.yaml applies only to **training** ROI crops.
``combined.roi_padding`` applies to **inference** only.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional


@dataclass
class BalancedScoreWeights:
    """Weights for tuning balanced_score (must sum to 1.0 for interpretability)."""

    bbox_AP50: float = 0.35
    segm_AP50: float = 0.35
    bbox_AP75: float = 0.15
    segm_AP75: float = 0.15

    def as_tuple(self) -> tuple:
        return (
            self.bbox_AP50,
            self.segm_AP50,
            self.bbox_AP75,
            self.segm_AP75,
        )


def _default_postprocess() -> List[Dict[str, Any]]:
    return []


@dataclass
class CombinedInferenceConfig:
    """All tunable combined-pipeline settings (single source of truth)."""

    yolo_conf_thresh: float = 0.25
    """Minimum YOLO confidence to keep a wound detection."""

    yolo_min_conf_inference: float = 0.001
    """Ultralytics `conf=` when caching raw outputs for tuning (low = keep all, filter in Python)."""

    unet_mask_thresh: float = 0.5
    """Threshold on U-Net probability map (after optional linear upscale)."""

    roi_padding: float = 0.1
    """Relative pad on YOLO box width/height (same convention as training ROI expansion)."""

    mask_upscale: str = "linear_probs"
    """``linear_probs``: resize sigmoid probs with INTER_LINEAR, then threshold. ``nearest_binary``: threshold on U-Net grid, nearest upscale."""

    bbox_selection_strategy: str = "highest_conf"
    """``highest_conf`` | ``largest_area`` | ``merge_overlapping`` — how to handle multiple wound boxes."""

    merge_iou_thresh: float = 0.5
    """IoU threshold for mask NMS / merge_overlapping."""

    coco_bbox_mode: str = "yolo_xyxy"
    """``yolo_xyxy`` | ``mask_tight`` | ``padded_roi`` — COCO bbox AP submission geometry."""

    min_mask_area: int = 0
    """Drop predicted mask components smaller than this (full-image pixels), 0 = disabled."""

    postprocess: List[Dict[str, Any]] = field(default_factory=_default_postprocess)
    """Ordered list of post-process ops; see ``combined/postprocess.py``."""

    postprocess_preset: str = "none"
    """Named preset for tuning grid (overrides ``postprocess`` when set from tuner)."""

    enable_tta: bool = True

    debug_save_intermediates: bool = False
    debug_output_dir: str = "results/combined/debug"
    debug_max_images: int = 16

    balanced_score_weights: BalancedScoreWeights = field(default_factory=BalancedScoreWeights)

    # passthrough for area calibration (unchanged)
    marker_real_cm: float = 3.0
    pixels_per_cm: float = 26.0

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name == "balanced_score_weights" and isinstance(v, BalancedScoreWeights):
                d[f.name] = {
                    "bbox_AP50": v.bbox_AP50,
                    "segm_AP50": v.segm_AP50,
                    "bbox_AP75": v.bbox_AP75,
                    "segm_AP75": v.segm_AP75,
                }
            else:
                d[f.name] = v
        return d


def _nested_get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def combined_config_from_dict(config: Dict[str, Any]) -> CombinedInferenceConfig:
    """Build :class:`CombinedInferenceConfig` from project YAML root (expects ``combined`` key)."""
    c = config.get("combined") or {}
    bw = c.get("balanced_score_weights") or {}
    weights = BalancedScoreWeights(
        bbox_AP50=float(bw.get("bbox_AP50", 0.35)),
        segm_AP50=float(bw.get("segm_AP50", 0.35)),
        bbox_AP75=float(bw.get("bbox_AP75", 0.15)),
        segm_AP75=float(bw.get("segm_AP75", 0.15)),
    )
    postprocess = c.get("postprocess")
    if postprocess is None:
        postprocess = []
    return CombinedInferenceConfig(
        yolo_conf_thresh=float(c.get("yolo_conf_thresh", 0.25)),
        yolo_min_conf_inference=float(c.get("yolo_min_conf_inference", 0.001)),
        unet_mask_thresh=float(c.get("unet_mask_thresh", 0.5)),
        roi_padding=float(c.get("roi_padding", 0.1)),
        mask_upscale=str(c.get("mask_upscale", "linear_probs")),
        bbox_selection_strategy=str(c.get("bbox_selection_strategy", "highest_conf")),
        merge_iou_thresh=float(c.get("merge_iou_thresh", 0.5)),
        coco_bbox_mode=str(c.get("coco_bbox_mode", "yolo_xyxy")),
        min_mask_area=int(c.get("min_mask_area", 0)),
        postprocess=list(postprocess),
        postprocess_preset=str(c.get("postprocess_preset", "none")),
        enable_tta=bool(c.get("enable_tta", True)),
        debug_save_intermediates=bool(c.get("debug_save_intermediates", False)),
        debug_output_dir=str(c.get("debug_output_dir", "results/combined/debug")),
        debug_max_images=int(c.get("debug_max_images", 16)),
        balanced_score_weights=weights,
        marker_real_cm=float(c.get("marker_real_cm", 3.0)),
        pixels_per_cm=float(c.get("pixels_per_cm", 26.0)),
    )


def merge_combined_config(base: CombinedInferenceConfig, overrides: Dict[str, Any]) -> CombinedInferenceConfig:
    """Return a copy with keys from ``overrides`` (subset of dataclass fields)."""
    from dataclasses import replace

    kwargs = {f.name: getattr(base, f.name) for f in fields(base)}
    for k, v in overrides.items():
        if k not in kwargs:
            continue
        if k == "balanced_score_weights" and isinstance(v, dict):
            cur = kwargs[k]
            if isinstance(cur, BalancedScoreWeights):
                kwargs[k] = replace(cur, **{kk: float(v[kk]) for kk in v if hasattr(cur, kk)})
        else:
            kwargs[k] = v
    return CombinedInferenceConfig(**kwargs)
