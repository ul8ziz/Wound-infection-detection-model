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
from .geometry import xyxy_to_padded_roi
from .postprocess import apply_postprocess_chain, resolve_postprocess

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
) -> None:
    """
    Save up to 10 diagnostic images for the **first wound detection** on ``image_path``.

    Filenames: ``{stem}_01_original.png`` … ``{stem}_10_overlay_gt.png`` (GT panel skipped if no gt_mask).
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

    wound_idxs = [
        i for i in range(len(boxes_xyxy))
        if int(classes[i]) == 0 and scores[i] >= cfg.yolo_conf_thresh
    ]
    if not wound_idxs:
        return
    i = wound_idxs[0]
    box = boxes_xyxy[i]
    sc = float(scores[i])

    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    cx1, cy1, cx2, cy2 = xyxy_to_padded_roi(x1, y1, x2, y2, img_w, img_h, cfg.roi_padding)

    # 1 original
    cv2.imwrite(str(out_dir / f"{stem}_01_original.png"), image_bgr)

    # 2 YOLO overlay
    vis2 = image_bgr.copy()
    for j in range(len(boxes_xyxy)):
        if int(classes[j]) != 0:
            continue
        b = boxes_xyxy[j]
        cv2.rectangle(vis2, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
        cv2.putText(vis2, f"{scores[j]:.2f}", (int(b[0]), int(b[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.imwrite(str(out_dir / f"{stem}_02_yolo_boxes.png"), vis2)

    crop = image_rgb[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return

    # 3 extracted ROI (padded)
    cv2.imwrite(str(out_dir / f"{stem}_03_roi_padded.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

    # 4 YOLO tight vs padded overlay on small crop canvas — skip duplicate; overlay on full image
    vis4 = image_bgr.copy()
    cv2.rectangle(vis4, (int(x1), int(y1)), (int(x2), int(y2)), (255, 128, 0), 2)
    cv2.rectangle(vis4, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)
    cv2.imwrite(str(out_dir / f"{stem}_04_yolo_vs_padded.png"), vis4)

    uh, uw = unet_size[0], unet_size[1]
    crop_resized = cv2.resize(crop, (uw, uh), interpolation=cv2.INTER_LINEAR)
    crop_tensor = torch.from_numpy(crop_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    crop_tensor = (crop_tensor.to(device) - mean) / std

    unet_model.eval()
    probs = _unet_probs_tta(unet_model, crop_tensor, cfg.enable_tta)

    # 5 resized input (denorm vis)
    denorm = crop_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    denorm = denorm * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    denorm = np.clip(denorm, 0, 1)
    denorm_u8 = (denorm * 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / f"{stem}_05_unet_input_denorm.png"), cv2.cvtColor(denorm_u8, cv2.COLOR_RGB2BGR))

    prob_vis = (np.clip(probs, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / f"{stem}_06_unet_prob_map.png"), prob_vis)

    th_small = (probs >= cfg.unet_mask_thresh).astype(np.uint8) * 255
    cv2.imwrite(str(out_dir / f"{stem}_07_thresh_unet_grid.png"), th_small)

    crop_h, crop_w = cy2 - cy1, cx2 - cx1
    post_ops = resolve_postprocess(cfg.postprocess_preset, cfg.postprocess)
    if cfg.mask_upscale == "linear_probs":
        prob_up = cv2.resize(probs.astype(np.float32), (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        m_roi = (prob_up >= cfg.unet_mask_thresh).astype(np.uint8)
    else:
        m_small = (probs >= cfg.unet_mask_thresh).astype(np.uint8)
        m_roi = cv2.resize(m_small, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
    m_pp = apply_postprocess_chain(m_roi, post_ops)
    cv2.imwrite(str(out_dir / f"{stem}_08_postprocessed_roi.png"), (m_pp * 255).astype(np.uint8))

    full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    full_mask[cy1:cy2, cx1:cx2] = (m_pp > 0).astype(np.uint8)
    vis9 = image_bgr.copy()
    overlay = vis9.copy()
    overlay[:, :, 1] = np.clip(overlay[:, :, 1] + (full_mask * 120).astype(np.uint8), 0, 255)
    cv2.addWeighted(vis9, 0.65, overlay, 0.35, 0, vis9)
    cv2.imwrite(str(out_dir / f"{stem}_09_mask_on_image.png"), vis9)

    vis10 = image_bgr.copy()
    overlay2 = vis10.copy()
    overlay2[:, :, 1] = np.clip(overlay2[:, :, 1] + (full_mask * 100).astype(np.uint8), 0, 255)
    cv2.addWeighted(vis10, 0.6, overlay2, 0.4, 0, vis10)
    cv2.rectangle(vis10, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
    if gt_mask is not None and gt_mask.shape == (img_h, img_w):
        cnts, _ = cv2.findContours((gt_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis10, cnts, -1, (0, 128, 255), 2)
    cv2.imwrite(str(out_dir / f"{stem}_10_overlay_bbox_mask_gt.png"), vis10)
