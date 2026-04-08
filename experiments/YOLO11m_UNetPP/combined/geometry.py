"""ROI geometry helpers: padding, clamping, consistent rounding."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def xyxy_to_padded_roi(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
    roi_padding: float,
) -> Tuple[int, int, int, int]:
    """
    Expand an axis-aligned box by ``roi_padding`` fraction of its width/height,
    then clamp to image bounds (same spirit as ``WoundROIDataset._expand_bbox``).
    """
    bw = max(x2 - x1, 0.0)
    bh = max(y2 - y1, 0.0)
    pad_x = bw * roi_padding
    pad_y = bh * roi_padding
    cx1 = int(np.floor(x1 - pad_x))
    cy1 = int(np.floor(y1 - pad_y))
    cx2 = int(np.ceil(x2 + pad_x))
    cy2 = int(np.ceil(y2 + pad_y))
    cx1 = max(0, cx1)
    cy1 = max(0, cy1)
    cx2 = min(img_w, cx2)
    cy2 = min(img_h, cy2)
    if cx2 <= cx1 or cy2 <= cy1:
        # degenerate — fall back to integer-rounded tight box
        cx1 = max(0, int(np.floor(x1)))
        cy1 = max(0, int(np.floor(y1)))
        cx2 = min(img_w, int(np.ceil(x2)))
        cy2 = min(img_h, int(np.ceil(y2)))
    return cx1, cy1, cx2, cy2


def tight_bbox_from_binary_mask(mask: np.ndarray) -> Tuple[float, float, float, float]:
    """Axis-aligned bbox [x1,y1,x2,y2] from nonzero pixels; empty mask returns zeros."""
    m = (mask > 0).astype(np.uint8)
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(xs.min()), float(ys.min()), float(xs.max()) + 1.0, float(ys.max()) + 1.0


def bbox_xyxy_to_coco_xywh(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
    return float(x1), float(y1), float(x2 - x1), float(y2 - y1)
