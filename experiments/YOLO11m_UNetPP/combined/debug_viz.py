"""Save intermediate pipeline visualizations for debugging."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from pipeline_utils import IMAGENET_MEAN, IMAGENET_STD

from .config import CombinedInferenceConfig, combined_config_from_dict
from .geometry import tight_bbox_from_binary_mask, xyxy_to_padded_roi
from .inference import _select_wound_indices
from .postprocess import apply_postprocess_chain, filter_min_area, resolve_postprocess

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


def _draw_gt_overlay(vis: np.ndarray, gt_mask: Optional[np.ndarray], gt_bboxes: Optional[list] = None) -> np.ndarray:
    """Draw GT mask contours (orange) and GT bboxes (cyan) on *vis*."""
    if gt_mask is not None:
        cnts, _ = cv2.findContours(
            (gt_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(vis, cnts, -1, (0, 128, 255), 2)
    if gt_bboxes:
        for b in gt_bboxes:
            cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 0), 2)
    return vis


@torch.no_grad()
def save_combined_debug_steps(
    yolo_model,
    unet_model: nn.Module,
    image_path: str,
    device: torch.device,
    config: dict,
    out_dir: Path,
    stem: str,
    cfg: Optional[CombinedInferenceConfig] = None,
    gt_mask: Optional[np.ndarray] = None,
    gt_bboxes: Optional[list] = None,
) -> None:
    """
    Save 12-step diagnostic images for the **selected wound detection** on *image_path*.

    Uses `_select_wound_indices` from the inference module to ensure the
    debug panels exactly match the box the production pipeline would process.

    Panels:
      01  Original image
      02  GT bbox + GT mask overlay
      03  All YOLO predicted boxes
      04  Selected ROI box (YOLO tight vs padded)
      05  Cropped ROI before resize
      06  Resized ROI sent to U-Net (de-normalised)
      07  Raw U-Net probability map
      08  Thresholded binary mask (256x256)
      09  Post-processed mask (ROI scale)
      10  Projected mask on original image
      11  Final bbox + mask overlay
      12  Prediction vs GT comparison (side-by-side contours)
    """
    cfg = cfg or combined_config_from_dict(config)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unet_size = tuple(config["unet"].get("input_size", [256, 256]))
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(device)

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        logger.warning("Could not read %s", image_path)
        return
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = image_rgb.shape[:2]

    yolo_results = yolo_model(image_path, conf=cfg.yolo_min_conf_inference, verbose=False)
    if not yolo_results or len(yolo_results) == 0:
        return
    result = yolo_results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    wound_idxs = _select_wound_indices(
        boxes_xyxy, scores, classes,
        cfg.yolo_conf_thresh, wound_class_id=0,
        strategy=cfg.bbox_selection_strategy,
        img_hw=(img_h, img_w),
    )
    if not wound_idxs:
        return
    i = wound_idxs[0]
    box = boxes_xyxy[i]
    sc = float(scores[i])

    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    cx1, cy1, cx2, cy2 = xyxy_to_padded_roi(x1, y1, x2, y2, img_w, img_h, cfg.roi_padding)

    # ── Panel 01: Original ────────────────────────────────────────────────
    cv2.imwrite(str(out_dir / f"{stem}_01_original.png"), image_bgr)

    # ── Panel 02: GT overlay ──────────────────────────────────────────────
    vis2 = image_bgr.copy()
    _draw_gt_overlay(vis2, gt_mask, gt_bboxes)
    cv2.imwrite(str(out_dir / f"{stem}_02_gt_overlay.png"), vis2)

    # ── Panel 03: All YOLO boxes ──────────────────────────────────────────
    vis3 = image_bgr.copy()
    for j in range(len(boxes_xyxy)):
        color = (0, 255, 0) if int(classes[j]) == 0 else (255, 0, 0)
        b = boxes_xyxy[j]
        cv2.rectangle(vis3, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 2)
        cv2.putText(vis3, f"c{int(classes[j])}:{scores[j]:.2f}",
                    (int(b[0]), int(b[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.imwrite(str(out_dir / f"{stem}_03_yolo_all_boxes.png"), vis3)

    # ── Panel 04: Selected ROI box (tight YOLO in orange, padded in cyan) ─
    vis4 = image_bgr.copy()
    cv2.rectangle(vis4, (int(x1), int(y1)), (int(x2), int(y2)), (0, 140, 255), 2)
    cv2.rectangle(vis4, (cx1, cy1), (cx2, cy2), (255, 255, 0), 2)
    cv2.putText(vis4, f"sel#{i} conf={sc:.2f}", (cx1, cy1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.imwrite(str(out_dir / f"{stem}_04_selected_roi.png"), vis4)

    crop = image_rgb[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return

    # ── Panel 05: Cropped ROI before resize ───────────────────────────────
    cv2.imwrite(str(out_dir / f"{stem}_05_crop_before_resize.png"),
                cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

    uh, uw = unet_size[0], unet_size[1]
    crop_resized = cv2.resize(crop, (uw, uh), interpolation=cv2.INTER_LINEAR)
    crop_tensor = torch.from_numpy(crop_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    crop_tensor = (crop_tensor.to(device) - mean) / std

    unet_model.eval()
    probs = _unet_probs_tta(unet_model, crop_tensor, cfg.enable_tta)

    # ── Panel 06: Resized U-Net input (de-normalised) ─────────────────────
    denorm = crop_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    denorm = denorm * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    denorm = np.clip(denorm, 0, 1)
    denorm_u8 = (denorm * 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / f"{stem}_06_unet_input_denorm.png"),
                cv2.cvtColor(denorm_u8, cv2.COLOR_RGB2BGR))

    # ── Panel 07: Raw probability map ─────────────────────────────────────
    prob_vis = (np.clip(probs, 0, 1) * 255).astype(np.uint8)
    prob_color = cv2.applyColorMap(prob_vis, cv2.COLORMAP_JET)
    cv2.imwrite(str(out_dir / f"{stem}_07_unet_prob_map.png"), prob_color)

    # ── Panel 08: Thresholded mask (256x256) ──────────────────────────────
    th_small = (probs >= cfg.unet_mask_thresh).astype(np.uint8) * 255
    cv2.imwrite(str(out_dir / f"{stem}_08_thresh_mask_small.png"), th_small)

    # ── Panel 09: Post-processed mask at ROI scale ────────────────────────
    crop_h, crop_w = cy2 - cy1, cx2 - cx1
    post_ops = resolve_postprocess(cfg.postprocess_preset, cfg.postprocess)
    if cfg.mask_upscale == "linear_probs":
        prob_up = cv2.resize(probs.astype(np.float32), (crop_w, crop_h),
                             interpolation=cv2.INTER_LINEAR)
        m_roi = (prob_up >= cfg.unet_mask_thresh).astype(np.uint8)
    else:
        m_small = (probs >= cfg.unet_mask_thresh).astype(np.uint8)
        m_roi = cv2.resize(m_small, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
    m_pp = apply_postprocess_chain(m_roi, post_ops)
    if cfg.min_mask_area > 0:
        m_pp = filter_min_area(m_pp, cfg.min_mask_area)
    cv2.imwrite(str(out_dir / f"{stem}_09_postprocessed_roi.png"), (m_pp * 255).astype(np.uint8))

    # ── Panel 10: Projected mask on original ──────────────────────────────
    full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    full_mask[cy1:cy2, cx1:cx2] = (m_pp > 0).astype(np.uint8)
    mask_vis = (full_mask * 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / f"{stem}_10_projected_mask.png"), mask_vis)

    # ── Panel 11: Final bbox + mask overlay ───────────────────────────────
    vis11 = image_bgr.copy()
    green_layer = np.zeros_like(vis11)
    green_layer[:, :, 1] = 255
    vis11[full_mask > 0] = cv2.addWeighted(vis11, 0.55, green_layer, 0.45, 0)[full_mask > 0]
    tb = tight_bbox_from_binary_mask(full_mask)
    cv2.rectangle(vis11, (int(tb[0]), int(tb[1])), (int(tb[2]), int(tb[3])), (0, 255, 0), 2)
    cv2.rectangle(vis11, (cx1, cy1), (cx2, cy2), (255, 255, 0), 1)
    cv2.imwrite(str(out_dir / f"{stem}_11_final_overlay.png"), vis11)

    # ── Panel 12: Prediction vs GT comparison ─────────────────────────────
    vis12 = image_bgr.copy()
    pred_cnts, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis12, pred_cnts, -1, (0, 255, 0), 2)
    if gt_mask is not None and gt_mask.shape == (img_h, img_w):
        gt_cnts, _ = cv2.findContours(
            (gt_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(vis12, gt_cnts, -1, (0, 128, 255), 2)
        inter = ((full_mask > 0) & (gt_mask > 0)).sum()
        union = ((full_mask > 0) | (gt_mask > 0)).sum()
        iou = inter / max(union, 1)
        cv2.putText(vis12, f"IoU={iou:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(vis12, "Green=Pred  Orange=GT", (10, img_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(out_dir / f"{stem}_12_pred_vs_gt.png"), vis12)
