"""Marker-based scale estimation from YOLO detections.

The 3×3 cm reference marker is detected by YOLO alongside wounds.
Scale is estimated from the segmentation mask area (preferred, more robust
to bbox padding) or falls back to bounding-box dimensions.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _marker_size_from_mask(
    yolo_result,
    idx: int,
    img_hw: Optional[tuple] = None,
) -> Optional[float]:
    """Estimate marker side-length in pixels from its segmentation mask.

    Uses ``sqrt(area)`` which is exact for a perfect square and a good
    approximation for near-square markers.
    """
    if yolo_result.masks is None or idx >= len(yolo_result.masks):
        return None
    mask_data = yolo_result.masks.data[idx].cpu().numpy()
    if img_hw is not None and mask_data.shape[:2] != img_hw:
        mask_data = cv2.resize(mask_data, (img_hw[1], img_hw[0]),
                               interpolation=cv2.INTER_NEAREST)
    area_px = float((mask_data > 0.5).sum())
    if area_px < 9:
        return None
    return math.sqrt(area_px)


def calculate_pixels_per_cm_from_marker(
    yolo_result,
    marker_class_id: int = 1,
    marker_real_cm: float = 3.0,
) -> Optional[float]:
    """Return pixels-per-cm estimated from the best detected marker.

    Prefers mask-based estimation (``sqrt(mask_area)`` → side length) over
    bbox-based estimation (``(w+h)/2``).  Both assume the marker is
    approximately square.
    """
    if yolo_result.boxes is None:
        return None

    classes = yolo_result.boxes.cls.cpu().numpy()
    boxes = yolo_result.boxes.xyxy.cpu().numpy()
    confs = yolo_result.boxes.conf.cpu().numpy()

    marker_idxs = np.where(classes == marker_class_id)[0]
    if len(marker_idxs) == 0:
        return None

    best_idx = int(marker_idxs[confs[marker_idxs].argmax()])
    conf = float(confs[best_idx])

    img_hw = None
    if hasattr(yolo_result, "orig_shape"):
        img_hw = yolo_result.orig_shape

    mask_side = _marker_size_from_mask(yolo_result, best_idx, img_hw)
    if mask_side is not None:
        ppcm = mask_side / marker_real_cm
        logger.debug("Marker scale from mask: side=%.1f px, ppcm=%.2f, conf=%.3f",
                     mask_side, ppcm, conf)
        return float(ppcm)

    x1, y1, x2, y2 = boxes[best_idx]
    bbox_side = ((x2 - x1) + (y2 - y1)) / 2.0
    ppcm = bbox_side / marker_real_cm
    logger.debug("Marker scale from bbox: side=%.1f px, ppcm=%.2f, conf=%.3f",
                 bbox_side, ppcm, conf)
    return float(ppcm)
