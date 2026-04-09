"""Binary mask post-processing for combined pipeline."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np


def _largest_component(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros_like(m)
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    return (labels == idx).astype(np.uint8) * 255


def _remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            out[labels == i] = 255
    return out


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8) * 255
    h, w = m.shape
    flood = m.copy()
    flood_fill_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_fill_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(m, holes)


def apply_postprocess_chain(
    mask: np.ndarray,
    ops: Optional[List[Dict[str, Any]]],
) -> np.ndarray:
    """Apply ordered post-process ops to a binary mask (H,W) uint8 or float."""
    out = (mask > 0).astype(np.uint8) * 255
    if not ops:
        return (out > 0).astype(np.uint8)
    for op in ops:
        t = op.get("type", "")
        if t == "remove_small_components":
            out = _remove_small_components(out, int(op.get("min_size", 50)))
        elif t == "keep_largest":
            out = _largest_component(out)
        elif t == "binary_open":
            k = int(op.get("kernel", 3))
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            out = cv2.morphologyEx(out, cv2.MORPH_OPEN, ker)
        elif t == "binary_close":
            k = int(op.get("kernel", 3))
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, ker)
        elif t == "fill_holes":
            out = _fill_holes(out)
        elif t == "smooth_contour":
            eps = float(op.get("epsilon_px", 2.0))
            contours, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            canvas = np.zeros_like(out)
            for c in contours:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, eps, True)
                cv2.fillPoly(canvas, [approx], 255)
            out = np.maximum(out, canvas) if canvas.max() > 0 else out
        else:
            continue
    return (out > 0).astype(np.uint8)


PRESETS: Dict[str, List[Dict[str, Any]]] = {
    "none": [],
    "remove_small_cc": [{"type": "remove_small_components", "min_size": 50}],
    "binary_closing": [{"type": "binary_close", "kernel": 3}],
    "open_close": [
        {"type": "binary_open", "kernel": 3},
        {"type": "binary_close", "kernel": 3},
    ],
    "fill_holes": [{"type": "fill_holes"}],
    "close_fill": [
        {"type": "binary_close", "kernel": 3},
        {"type": "fill_holes"},
    ],
    "keep_largest_component": [{"type": "keep_largest"}],
    "opening_then_closing": [
        {"type": "binary_open", "kernel": 3},
        {"type": "binary_close", "kernel": 5},
    ],
    "closing_then_fill": [
        {"type": "binary_close", "kernel": 5},
        {"type": "fill_holes"},
    ],
    "largest_then_fill": [
        {"type": "keep_largest"},
        {"type": "fill_holes"},
    ],
    "largest_close_fill": [
        {"type": "keep_largest"},
        {"type": "binary_close", "kernel": 5},
        {"type": "fill_holes"},
    ],
    "boundary_refine": [
        {"type": "binary_close", "kernel": 3},
        {"type": "fill_holes"},
        {"type": "smooth_contour", "epsilon_px": 1.5},
    ],
    "boundary_refine_strong": [
        {"type": "binary_close", "kernel": 5},
        {"type": "fill_holes"},
        {"type": "smooth_contour", "epsilon_px": 2.0},
    ],
}


def resolve_postprocess(
    preset: str,
    explicit: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Prefer explicit chain; else use named preset."""
    if explicit:
        return list(explicit)
    return list(PRESETS.get(preset, PRESETS["none"]))


def resolve_refinement_postprocess(mode: str) -> List[Dict[str, Any]]:
    """Resolve optional post-threshold refinement chain."""
    if not mode or mode == "none":
        return []
    return list(PRESETS.get(mode, PRESETS["none"]))


def filter_min_area(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask
    m = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 1
    return out
