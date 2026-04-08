"""Marker-based scale estimation from YOLO detections."""

from __future__ import annotations

from typing import Optional

import numpy as np


def calculate_pixels_per_cm_from_marker(
    yolo_result,
    marker_class_id: int = 1,
    marker_real_cm: float = 3.0,
) -> Optional[float]:
    if yolo_result.boxes is None:
        return None

    classes = yolo_result.boxes.cls.cpu().numpy()
    boxes = yolo_result.boxes.xyxy.cpu().numpy()

    marker_idxs = np.where(classes == marker_class_id)[0]
    if len(marker_idxs) == 0:
        return None

    best_idx = marker_idxs[
        yolo_result.boxes.conf.cpu().numpy()[marker_idxs].argmax()
    ]
    x1, y1, x2, y2 = boxes[best_idx]
    marker_px = ((x2 - x1) + (y2 - y1)) / 2.0
    return float(marker_px / marker_real_cm)
