"""
سكريبت تدريب نموذج Mask R-CNN لاكتشاف وقياس الجروح
Training Script for Mask R-CNN Wound Detection and Measurement Model
=====================================================================

هذا الملف يدمج جميع وظائف التدريب في مكان واحد:
- بناء النموذج
- تدريب النموذج
- تقييم النموذج
- حفظ/تحميل checkpoints
- تشغيل inference
- توليد التقارير

This file combines all training functions in one place:
- Model building
- Model training
- Model evaluation
- Checkpoint saving/loading
- Inference
- Report generation

Usage:
    python train_model.py
"""

import os
import sys
import math
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# Try importing pycocotools
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import pycocotools.mask as mask_util
    HAS_COCO = True
except ImportError:
    HAS_COCO = False
    mask_util = None

# Fix encoding for Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python < 3.7
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Also add scripts directory for augmentation_strategy
project_root = current_dir.parent
scripts_dir = project_root / "scripts"
if (scripts_dir / "augmentation_strategy.py").exists():
    sys.path.insert(0, str(scripts_dir))

# Import pipeline utils
try:
    from pipeline_utils import (
        set_seed, 
        get_device, 
        create_dataset, 
        make_dataloaders
    )
except ImportError:
    # Fallback for CLI usage depending on where it's run
    sys.path.append(str(current_dir))
    from pipeline_utils import get_device, create_dataset, make_dataloaders, set_seed

# ============================================================================
# Model Building Functions
# ============================================================================

def build_model(num_classes: int, hidden_layer: int = 256, pretrained_backbone: bool = True):
    """
    Builds the Mask R-CNN model with a ResNet-50-FPN backbone.
    
    Args:
        num_classes (int): Number of classes (including background).
        hidden_layer (int): Size of the mask predictor hidden layer.
        pretrained_backbone (bool): Whether to use pretrained backbone weights.
    
    Returns:
        torch.nn.Module: Mask R-CNN model
    """
    weights = "DEFAULT" if pretrained_backbone else None
    model = maskrcnn_resnet50_fpn(weights=weights)

    # 1. Replace the box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 2. Replace the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model

# ============================================================================
# Training Functions
# ============================================================================

def train_one_epoch(
    model: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    data_loader: torch.utils.data.DataLoader, 
    device: torch.device, 
    epoch: int, 
    print_freq: int = 10,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scheduler_step_per_iter: bool = False,
    max_norm: float = 1.0
) -> Dict[str, float]:
    """
    Trains the model for one epoch.
    
    Returns:
        Dict[str, float]: Averaged losses dictionary
    """
    model.train()
    metric_logger = {}
    
    header = f'Epoch: [{epoch}]'
    
    total_loss_accum = 0.0
    num_batches = len(data_loader)
    
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Optimizer zero_grad
        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        # Backward pass
        if scaler is not None:
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        # Scheduler stepping: only if explicitly requested per-iteration
        if scheduler is not None and scheduler_step_per_iter:
            scheduler.step()

        # Update logs
        total_loss_accum += loss_value
        for k, v in loss_dict.items():
            if k not in metric_logger:
                metric_logger[k] = 0.0
            metric_logger[k] += v.item()

        if i % print_freq == 0:
            print(f"{header} [{i}/{num_batches}] Loss: {loss_value:.4f}")

    # Average losses
    avg_loss = total_loss_accum / num_batches
    avg_components = {k: v / num_batches for k, v in metric_logger.items()}
    avg_components['total_loss'] = avg_loss
    
    return avg_components

@torch.no_grad()
def validate_one_epoch(
    model: nn.Module, 
    data_loader: torch.utils.data.DataLoader, 
    device: torch.device,
    track_predictions: bool = False
) -> Dict[str, float]:
    """
    Computes validation LOSS (not metrics).
    Note: Torchvision detection models compute loss only in train() mode.
    We use train() mode with torch.no_grad() to get loss dict without gradients.
    For metrics evaluation, use evaluate_metrics() which runs in eval() mode.
    
    Args:
        track_predictions: If True, also run model in eval() mode to track prediction scores
    
    Returns:
        Dict[str, float]: Validation losses dictionary
    """
    model.train() 
    
    total_loss_accum = 0.0
    metric_logger = {}
    num_batches = len(data_loader)
    
    # Track prediction scores if requested
    all_scores = []
    predictions_per_thresh = {0.3: 0, 0.5: 0, 0.7: 0}
    
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.item()
        total_loss_accum += loss_value
        
        for k, v in loss_dict.items():
            if k not in metric_logger:
                metric_logger[k] = 0.0
            metric_logger[k] += v.item()
        
        # Track predictions if requested
        if track_predictions:
            model.eval()
            with torch.no_grad():
                outputs = model(images)
                for output in outputs:
                    scores = output['scores'].cpu().numpy()
                    all_scores.extend(scores.tolist())
                    for thresh in predictions_per_thresh:
                        predictions_per_thresh[thresh] += np.sum(scores >= thresh)
            model.train()

    avg_loss = total_loss_accum / max(1, num_batches)
    avg_components = {k: v / max(1, num_batches) for k, v in metric_logger.items()}
    avg_components['total_loss'] = avg_loss
    
    if track_predictions and len(all_scores) > 0:
        avg_components['pred_mean_score'] = float(np.mean(all_scores))
        avg_components['pred_median_score'] = float(np.median(all_scores))
        avg_components['pred_max_score'] = float(np.max(all_scores))
        avg_components['pred_min_score'] = float(np.min(all_scores))
        for thresh, count in predictions_per_thresh.items():
            avg_components[f'pred_at_thresh_{thresh}'] = count
    
    return avg_components

# ============================================================================
# Evaluation Functions
# ============================================================================

@torch.no_grad()
def evaluate_metrics(
    model: nn.Module, 
    data_loader: torch.utils.data.DataLoader, 
    device: torch.device
):
    """
    Evaluates model using COCO metrics (if available) or fallback custom metrics.
    Returns both bbox and segmentation metrics for Mask R-CNN.
    
    Returns:
        Dict: Metrics dictionary with AP scores
    """
    model.eval()
    cpu_device = torch.device("cpu")
    
    dataset = data_loader.dataset
    if isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset

    if hasattr(dataset, 'coco') and HAS_COCO and hasattr(dataset, 'ann_file'):
        print("Using COCO evaluator...")
        coco_gt = dataset.coco
        
        coco_results_bbox = []
        coco_results_segm = []
        
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            
            for target, output in zip(targets, outputs):
                image_id = target["image_id"].item()
                
                boxes = output["boxes"].tolist()
                scores = output["scores"].tolist()
                labels = output["labels"].tolist()
                
                # Process masks if available
                has_masks = "masks" in output and len(output["masks"]) > 0
                masks_np = None
                if has_masks:
                    masks = output["masks"]
                    masks_binary = (masks > 0.5).squeeze(1).byte()
                    masks_np = masks_binary.numpy()
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    w = max(0, x2 - x1)
                    h = max(0, y2 - y1)
                    x = x1
                    y = y1
                    
                    res_bbox = {
                        "image_id": image_id,
                        "category_id": int(labels[i]),
                        "bbox": [x, y, w, h],
                        "score": float(scores[i])
                    }
                    coco_results_bbox.append(res_bbox)
                    
                    # Add segmentation if masks available
                    if has_masks and i < len(masks_np):
                        mask = masks_np[i]
                        if mask.dtype != np.uint8:
                            mask = mask.astype(np.uint8)
                        rle = mask_util.encode(np.asfortranarray(mask))
                        if isinstance(rle['counts'], bytes):
                            rle['counts'] = rle['counts'].decode('utf-8')
                        
                        res_segm = res_bbox.copy()
                        res_segm["segmentation"] = rle
                        coco_results_segm.append(res_segm)
        
        if not coco_results_bbox:
            print("⚠️  No predictions generated (all scores below threshold or no detections).")
            return {"bbox_AP": 0.0, "bbox_AP50": 0.0, "bbox_AP75": 0.0, "combined_AP50": 0.0}
        
        try:
            # Fix missing 'info' field in COCO dataset if needed
            try:
                if hasattr(coco_gt, 'dataset') and isinstance(coco_gt.dataset, dict):
                    if 'info' not in coco_gt.dataset:
                        coco_gt.dataset['info'] = {
                            "description": "Wound Infection Detection Dataset",
                            "version": "1.0",
                            "year": 2025
                        }
            except (AttributeError, TypeError):
                pass
            
            # Evaluate bbox metrics
            coco_dt_bbox = coco_gt.loadRes(coco_results_bbox)
            coco_eval_bbox = COCOeval(coco_gt, coco_dt_bbox, "bbox")
            coco_eval_bbox.evaluate()
            coco_eval_bbox.accumulate()
            coco_eval_bbox.summarize()
            
            metrics = {
                "bbox_AP": coco_eval_bbox.stats[0],
                "bbox_AP50": coco_eval_bbox.stats[1],
                "bbox_AP75": coco_eval_bbox.stats[2]
            }
            
            # Evaluate segmentation metrics if masks available
            if coco_results_segm:
                coco_dt_segm = coco_gt.loadRes(coco_results_segm)
                coco_eval_segm = COCOeval(coco_gt, coco_dt_segm, "segm")
                coco_eval_segm.evaluate()
                coco_eval_segm.accumulate()
                coco_eval_segm.summarize()
                
                metrics.update({
                    "segm_AP": coco_eval_segm.stats[0],
                    "segm_AP50": coco_eval_segm.stats[1],
                    "segm_AP75": coco_eval_segm.stats[2]
                })
                
                metrics["combined_AP50"] = (metrics["bbox_AP50"] + metrics["segm_AP50"]) / 2.0
            else:
                metrics["combined_AP50"] = metrics["bbox_AP50"]
            
            return metrics
            
        except Exception as e:
            print(f"COCO eval failed: {type(e).__name__}: {e}")
            print("Falling back to custom metrics.")
    
    # Fallback Custom Metrics
    print("Running fallback metrics (Precision/Recall @ IoU 0.5)...")
    tp = 0
    fp = 0
    fn = 0
    
    iou_threshold = 0.5
    total_raw_predictions = 0
    total_filtered_predictions = 0
    all_raw_scores = []
    
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        
        for target, output in zip(targets, outputs):
            gt_boxes = target["boxes"]
            pred_boxes = output["boxes"]
            pred_scores = output["scores"]
            
            total_raw_predictions += len(pred_scores)
            if len(pred_scores) > 0:
                all_raw_scores.extend(pred_scores.tolist())
            
            keep = pred_scores > 0.5
            pred_boxes_filtered = pred_boxes[keep]
            total_filtered_predictions += len(pred_boxes_filtered)
            
            if len(gt_boxes) == 0:
                fp += len(pred_boxes_filtered)
                continue
            
            if len(pred_boxes_filtered) == 0:
                fn += len(gt_boxes)
                continue
                
            # Compute IoU matrix
            box_area = lambda boxes: (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            area1 = box_area(gt_boxes)
            area2 = box_area(pred_boxes_filtered)
            
            lt = torch.max(gt_boxes[:, None, :2], pred_boxes_filtered[:, :2])
            rb = torch.min(gt_boxes[:, None, 2:], pred_boxes_filtered[:, 2:])
            wh = (rb - lt).clamp(min=0)
            inter = wh[:, :, 0] * wh[:, :, 1]
            
            union = area1[:, None] + area2 - inter
            iou = inter / union
            
            matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool)
            matched_pred = torch.zeros(len(pred_boxes_filtered), dtype=torch.bool)
            
            for i in range(len(gt_boxes)):
                max_iou, max_idx = iou[i].max(dim=0)
                if max_iou > iou_threshold:
                    if not matched_pred[max_idx]:
                        matched_gt[i] = True
                        matched_pred[max_idx] = True
                        tp += 1
            
            fn += len(gt_boxes) - matched_gt.sum().item()
            fp += len(pred_boxes_filtered) - matched_pred.sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    if len(all_raw_scores) > 0:
        print(f"  Raw predictions: {total_raw_predictions}, Filtered (score>0.5): {total_filtered_predictions}")
        print(f"  Score stats: min={min(all_raw_scores):.3f}, max={max(all_raw_scores):.3f}, mean={np.mean(all_raw_scores):.3f}")
    else:
        print(f"  ⚠️  No raw predictions generated - model may need more training")
    
    return {
        "precision": precision, 
        "recall": recall, 
        "f1": f1,
        "bbox_AP50": f1,
        "combined_AP50": f1,
        "raw_predictions_count": total_raw_predictions,
        "filtered_predictions_count": total_filtered_predictions
    }

# ============================================================================
# Checkpoint Functions
# ============================================================================

def save_checkpoint(
    state: Dict, 
    output_dir: str, 
    filename: str = "last.pt", 
    is_best: bool = False,
    best_metric_name: str = "metric",
    current_metric: float = 0.0
):
    """
    Saves checkpoint. Always saves as 'last.pt', and also as 'best.pt' if is_best=True.
    
    Args:
        state: Dictionary containing model state, optimizer, scheduler, epoch, best_metric, etc.
        output_dir: Directory to save checkpoints
        filename: Filename for the checkpoint (default: "last.pt")
        is_best: If True, also save as "best.pt"
        best_metric_name: Name of the metric being tracked
        current_metric: Current metric value
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    torch.save(state, save_path)
    
    if is_best:
        best_path = os.path.join(output_dir, "best.pt")
        torch.save(state, best_path)
        epoch = state.get('epoch', 'unknown')
        best_metric = state.get('best_metric', 'unknown')
        print(f"  → Checkpoint saved: {save_path}")
        print(f"  → Best model saved: {best_path} (epoch {epoch}, {best_metric_name}={best_metric:.4f})")
    else:
        epoch = state.get('epoch', 'unknown')
        print(f"  → Checkpoint saved: {save_path} (epoch {epoch})")

def load_checkpoint(
    model: nn.Module, 
    path: str, 
    optimizer: Optional[torch.optim.Optimizer] = None, 
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> Dict:
    """
    Load checkpoint from file.
    Handles PyTorch 2.6+ compatibility with numpy arrays in checkpoints.
    
    Args:
        model: Model to load state into
        path: Path to checkpoint file
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
    
    Returns:
        Dict: Checkpoint dictionary
    """
    # For PyTorch 2.6+, we need to handle numpy scalars specially
    numpy_scalar = None
    try:
        numpy_scalar = np._core.multiarray.scalar
    except AttributeError:
        try:
            numpy_scalar = np.core.multiarray.scalar
        except AttributeError:
            pass
    
    # Add numpy scalar to safe globals BEFORE any torch.load call (PyTorch 2.6+)
    if numpy_scalar is not None:
        if hasattr(torch.serialization, 'add_safe_globals'):
            try:
                torch.serialization.add_safe_globals([numpy_scalar])
            except Exception:
                pass
    
    # Try to load with safe_globals context manager first (if available)
    checkpoint = None
    load_success = False
    
    if numpy_scalar is not None and hasattr(torch.serialization, 'safe_globals'):
        try:
            with torch.serialization.safe_globals([numpy_scalar]):
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            load_success = True
        except Exception:
            pass
    
    # Fallback: use weights_only=False (now that numpy_scalar is in safe globals)
    if not load_success:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    
    model.load_state_dict(checkpoint["model"])
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint

# ============================================================================
# Inference Functions
# ============================================================================

@torch.no_grad()
def run_inference(
    model: nn.Module,
    image: Union[torch.Tensor, np.ndarray, str],
    device: torch.device,
    conf_thresh: float = 0.5
) -> Dict:
    """
    Run inference on a single image.
    
    Args:
        model: Trained Mask R-CNN model
        image: Image as tensor [C, H, W], numpy array [H, W, C], or path to image file
        device: torch device
        conf_thresh: Confidence threshold for filtering predictions
        
    Returns:
        Dictionary with filtered and raw predictions
    """
    import cv2
    from albumentations import Compose, Resize, Normalize
    from albumentations.pytorch import ToTensorV2
    
    model.eval()
    
    # Load and preprocess image
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Could not load image: {image}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image, Path):
        img = cv2.imread(str(image))
        if img is None:
            raise ValueError(f"Could not load image: {image}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image, np.ndarray):
        img = image
        if len(img.shape) == 3 and img.shape[2] == 3:
            pass  # Already RGB
        else:
            raise ValueError(f"Expected RGB image, got shape: {img.shape}")
    elif isinstance(image, torch.Tensor):
        if image.dim() == 3:
            img = image.permute(1, 2, 0).cpu().numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)
        else:
            raise ValueError(f"Expected 3D tensor [C, H, W], got shape: {image.shape}")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    
    # Resize and normalize
    transform = Compose([
        Resize(height=512, width=512),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    transformed = transform(image=img)
    img_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Run inference
    outputs = model(img_tensor)
    output = outputs[0]  # Single image
    
    # Extract raw predictions
    raw_boxes = output['boxes'].cpu().numpy()
    raw_scores = output['scores'].cpu().numpy()
    raw_labels = output['labels'].cpu().numpy()
    raw_masks = output['masks'].cpu().numpy() if 'masks' in output else None
    
    # Filter by confidence
    keep = raw_scores >= conf_thresh
    filtered_boxes = raw_boxes[keep]
    filtered_scores = raw_scores[keep]
    filtered_labels = raw_labels[keep]
    filtered_masks = raw_masks[keep] if raw_masks is not None else None
    
    return {
        'filtered': {
            'boxes': filtered_boxes,
            'scores': filtered_scores,
            'labels': filtered_labels,
            'masks': filtered_masks
        },
        'raw': {
            'boxes': raw_boxes,
            'scores': raw_scores,
            'labels': raw_labels,
            'masks': raw_masks
        },
        'num_detections': len(filtered_boxes),
        'num_raw': len(raw_boxes),
        'conf_thresh': conf_thresh
    }

def run_wound_inference(
    model: nn.Module,
    image: Union[torch.Tensor, np.ndarray, str],
    device: torch.device,
    conf_thresh: float = 0.3,
    wound_class_ids: List[int] = None,
    infection_class_ids: List[int] = None,
    marker_class_id: int = None,
    marker_size_cm2: float = 9.0  # 3x3 cm = 9 cm²
) -> Dict:
    """
    Run inference and compute wound area and infection indicators.
    
    Args:
        model: Trained Mask R-CNN model
        image: Image as tensor, numpy array, or path
        device: torch device
        conf_thresh: Confidence threshold
        wound_class_ids: List of class IDs for wound regions
        infection_class_ids: List of class IDs for infection indicators
        marker_class_id: Class ID for the reference marker
        marker_size_cm2: Physical size of marker in cm² (default 3x3 cm = 9 cm²)
        
    Returns:
        Dictionary with wound area, infection flags, and detections
    """
    # Run inference
    result = run_inference(model, image, device, conf_thresh)
    
    filtered = result['filtered']
    boxes = filtered['boxes']
    scores = filtered['scores']
    labels = filtered['labels']
    masks = filtered['masks']
    
    # Get image dimensions (assuming 512x512 after resize)
    img_h, img_w = 512, 512
    image_area_pixels = img_h * img_w
    
    # Find marker
    marker_found = False
    marker_area_pixels = None
    pixel_to_cm2_ratio = None
    
    if marker_class_id is not None:
        marker_indices = np.where(labels == marker_class_id)[0]
        if len(marker_indices) > 0:
            marker_idx = marker_indices[np.argmax(scores[marker_indices])]
            marker_found = True
            
            if masks is not None:
                marker_mask = masks[marker_idx, 0]
                marker_area_pixels = np.sum(marker_mask > 0.5)
            else:
                x1, y1, x2, y2 = boxes[marker_idx]
                marker_area_pixels = (x2 - x1) * (y2 - y1)
            
            if marker_area_pixels > 0:
                pixel_to_cm2_ratio = marker_size_cm2 / marker_area_pixels
    
    # Compute wound area
    wound_area_pixels = 0.0
    wound_area_cm2 = None
    
    if wound_class_ids is not None:
        wound_indices = np.where(np.isin(labels, wound_class_ids))[0]
        if len(wound_indices) > 0 and masks is not None:
            for idx in wound_indices:
                wound_mask = masks[idx, 0]
                wound_area_pixels += np.sum(wound_mask > 0.5)
            
            if marker_found and pixel_to_cm2_ratio is not None:
                wound_area_cm2 = wound_area_pixels * pixel_to_cm2_ratio
    
    wound_area_ratio = wound_area_pixels / image_area_pixels if image_area_pixels > 0 else 0.0
    
    # Check for infection indicators
    infection_flags = {}
    if infection_class_ids is not None:
        for inf_class_id in infection_class_ids:
            inf_indices = np.where(labels == inf_class_id)[0]
            detected = len(inf_indices) > 0
            max_score = float(np.max(scores[inf_indices])) if detected else 0.0
            infection_flags[inf_class_id] = {
                'detected': detected,
                'max_score': max_score,
                'count': len(inf_indices)
            }
    
    # Create detections list
    detections = []
    for i in range(len(boxes)):
        detections.append({
            'class_id': int(labels[i]),
            'score': float(scores[i]),
            'box': boxes[i].tolist(),
            'has_mask': masks is not None and masks[i] is not None
        })
    
    return {
        'wound_area_cm2': wound_area_cm2,
        'wound_area_pixels': float(wound_area_pixels),
        'wound_area_ratio': wound_area_ratio,
        'marker_found': marker_found,
        'marker_area_pixels': float(marker_area_pixels) if marker_area_pixels is not None else None,
        'pixel_to_cm2_ratio': pixel_to_cm2_ratio,
        'infection_flags': infection_flags,
        'num_detections': len(boxes),
        'detections': detections,
        'raw_stats': {
            'num_raw': result['num_raw'],
            'num_filtered': result['num_detections'],
            'conf_thresh': conf_thresh
        }
    }

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    # Data paths
    "data_root": "../data",  # Original data root (data/ in project root)
    "ann_file_train": "../data/splits/train.json",  # Training annotations
    "ann_file_val": "../data/splits/val.json",  # Validation annotations
    "ann_file_full": "../data/annotations.json",  # Full annotations
    
    # Training settings
    "output_dir": "../checkpoints_medical_aug",  # Checkpoints in project root
    "seed": 42,
    "batch_size": 4,
    "num_workers": 0,  # Set to 0 for Windows compatibility
    "epochs": 50,
    "lr": 0.005,
    "image_size": (512, 512),
    
    # Medical Augmentation Strategy Settings
    "use_medical_augmentation": True,  # Enable comprehensive medical augmentation
    "preserve_marker": True,           # Preserve marker geometry (critical for area measurements)
    "intensity": "moderate"            # Augmentation intensity: "light", "moderate", "aggressive"
}

# ============================================================================
# Main Training Function
# ============================================================================

def main():
    """Main training function."""
    
    print("=" * 80)
    print("Medical Augmentation Training Script")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Setup
    set_seed(CONFIG["seed"])
    device = get_device(CONFIG.get("device_prefer_cuda", True))
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    train_ann = (script_dir / CONFIG["ann_file_train"]).resolve()
    val_ann = (script_dir / CONFIG["ann_file_val"]).resolve()
    ann_full = (script_dir / CONFIG["ann_file_full"]).resolve()
    data_root = (script_dir / CONFIG["data_root"]).resolve()
    
    # Check if split files exist
    if not train_ann.exists():
        print(f"[WARNING] Split file {train_ann} not found.")
        print(f"   Using full annotation file: {ann_full}")
        train_ann = ann_full
        val_ann = ann_full
    
    # Create datasets
    print("=" * 80)
    print("Loading Datasets")
    print("=" * 80)
    print(f"Train annotations: {train_ann}")
    print(f"Val annotations: {val_ann}")
    print(f"Data root: {data_root}")
    print()
    print(f"Medical Augmentation: {CONFIG['use_medical_augmentation']}")
    print(f"Preserve Marker: {CONFIG['preserve_marker']}")
    print(f"Intensity: {CONFIG['intensity']}")
    print()
    
    try:
        train_dataset = create_dataset(
            root=str(data_root),
            annotation_file=str(train_ann),
            train=True,
            image_size=CONFIG["image_size"],
            use_medical_augmentation=CONFIG["use_medical_augmentation"],
            preserve_marker=CONFIG["preserve_marker"],
            intensity=CONFIG["intensity"]
        )
        
        val_dataset = create_dataset(
            root=str(data_root),
            annotation_file=str(val_ann),
            train=False,
            image_size=CONFIG["image_size"],
            use_medical_augmentation=CONFIG["use_medical_augmentation"],
            preserve_marker=CONFIG["preserve_marker"],
            intensity=CONFIG["intensity"]
        )
        
        print(f"[OK] Train dataset loaded: {len(train_dataset)} samples")
        print(f"[OK] Val dataset loaded: {len(val_dataset)} samples")
        
    except Exception as e:
        print(f"[ERROR] Error loading datasets: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Create data loaders
    print()
    print("Creating data loaders...")
    train_loader, val_loader = make_dataloaders(
        train_dataset, 
        val_dataset,
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"]
    )
    print(f"[OK] Train batches: {len(train_loader)}")
    print(f"[OK] Val batches: {len(val_loader)}")
    print()
    
    # Build model
    print("=" * 80)
    print("Building Model")
    print("=" * 80)
    
    if hasattr(train_dataset, 'coco_json'):
        num_classes = len(train_dataset.coco_json['categories']) + 1
    elif hasattr(train_dataset, 'coco'):
        if isinstance(train_dataset.coco, dict):
            num_classes = len(train_dataset.coco['categories']) + 1
        else:
            num_classes = len(train_dataset.coco.loadCats(train_dataset.coco.getCatIds())) + 1
    else:
        num_classes = 17  # Default fallback
        print(f"[WARNING] Could not determine num_classes, using default: {num_classes}")
    
    print(f"Number of classes (including background): {num_classes}")
    
    model = build_model(num_classes=num_classes, pretrained_backbone=True)
    model.to(device)
    print(f"[OK] Model created and moved to {device}")
    print()
    
    # Optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=CONFIG["lr"], momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    print(f"Optimizer: SGD (lr={CONFIG['lr']}, momentum=0.9, weight_decay=0.0005)")
    print(f"Scheduler: StepLR (step_size=5, gamma=0.1)")
    print()
    
    # Training results storage
    results = {
        "config": CONFIG.copy(),
        "train_losses": [],
        "val_losses": [],
        "metrics_per_epoch": [],
        "best_metric": -1.0,
        "best_epoch": 0,
        "training_start": datetime.now().isoformat(),
        "training_time": None,
        "device": str(device),
        "num_classes": num_classes
    }
    
    # Output directory
    output_dir = script_dir / CONFIG["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()
    
    # Training loop
    print("=" * 80)
    print(f"Starting Training for {CONFIG['epochs']} Epochs")
    print("=" * 80)
    print()
    
    start_time = time.time()
    best_metric = -1.0
    best_epoch = 0
    best_metric_name = "combined_AP50"
    
    try:
        for epoch in range(CONFIG["epochs"]):
            epoch_start = time.time()
            
            print(f"Epoch [{epoch+1}/{CONFIG['epochs']}]")
            print("-" * 80)
            
            # Train
            train_stats = train_one_epoch(
                model, optimizer, train_loader, device, epoch,
                scheduler=lr_scheduler,
                scheduler_step_per_iter=False
            )
            results["train_losses"].append(train_stats["total_loss"])
            
            # Validate
            val_stats = validate_one_epoch(model, val_loader, device)
            results["val_losses"].append(val_stats["total_loss"])
            
            # Evaluate metrics
            print("Evaluating metrics...")
            metrics = evaluate_metrics(model, val_loader, device)
            results["metrics_per_epoch"].append(metrics)
            
            # Determine best metric
            current_metric = metrics.get(
                "combined_AP50", 
                metrics.get("bbox_AP50", metrics.get("f1", 0.0))
            )
            
            # Print epoch summary
            print(f"Train Loss: {train_stats['total_loss']:.4f} | Val Loss: {val_stats['total_loss']:.4f}")
            print(f"Current {best_metric_name}: {current_metric:.4f}")
            
            # Save checkpoints
            is_best = current_metric > best_metric
            if is_best:
                best_metric = current_metric
                best_epoch = epoch + 1
                results["best_metric"] = best_metric
                results["best_epoch"] = best_epoch
                print(f"✓ NEW BEST! {best_metric_name}: {best_metric:.4f} (Epoch {best_epoch})")
            else:
                gap = best_metric - current_metric
                print(f"  (Best: {best_metric:.4f}, Gap: {gap:.4f})")
            
            # Save checkpoints
            save_checkpoint(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "best_metric": best_metric,
                    "best_metric_name": best_metric_name,
                    "current_metric": current_metric,
                    "config": str(CONFIG)
                },
                str(output_dir),
                filename="last.pt",
                is_best=False,
                best_metric_name=best_metric_name,
                current_metric=current_metric
            )
            
            if is_best:
                save_checkpoint(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch + 1,
                        "best_metric": best_metric,
                        "best_metric_name": best_metric_name,
                        "current_metric": current_metric,
                        "config": str(CONFIG)
                    },
                    str(output_dir),
                    filename="best.pt",
                    is_best=True,
                    best_metric_name=best_metric_name,
                    current_metric=current_metric
                )
            
            epoch_time = time.time() - epoch_start
            print(f"Epoch time: {epoch_time:.2f}s")
            print()
            
            # Step scheduler
            lr_scheduler.step()
        
        results["training_time"] = time.time() - start_time
        results["training_end"] = datetime.now().isoformat()
        
    except KeyboardInterrupt:
        print("\n[WARNING] Training interrupted by user")
        results["training_time"] = time.time() - start_time
        results["training_end"] = datetime.now().isoformat()
        results["interrupted"] = True
    except Exception as e:
        print(f"\n[ERROR] Error during training: {e}")
        import traceback
        traceback.print_exc()
        results["training_time"] = time.time() - start_time
        results["training_end"] = datetime.now().isoformat()
        results["error"] = str(e)
    
    # Final evaluation
    print("=" * 80)
    print("Final Evaluation")
    print("=" * 80)
    print()
    
    try:
        final_metrics = evaluate_metrics(model, val_loader, device)
        results["final_metrics"] = final_metrics
        
        print("Final Metrics:")
        for key, value in final_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print()
        
    except Exception as e:
        print(f"[WARNING] Error in final evaluation: {e}")
        results["final_metrics"] = {}
    
    # Save results
    print("=" * 80)
    print("Saving Results")
    print("=" * 80)
    print()
    
    results_file = output_dir / "training_results.json"
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[OK] Results saved to: {results_file}")
    except Exception as e:
        print(f"[ERROR] Error saving results: {e}")
    
    # Generate report
    report_file = output_dir / "training_report.md"
    try:
        generate_report(results, report_file)
        print(f"[OK] Report saved to: {report_file}")
    except Exception as e:
        print(f"[ERROR] Error generating report: {e}")
        import traceback
        traceback.print_exc()
    
    # Print summary
    print()
    print("=" * 80)
    print("Training Summary")
    print("=" * 80)
    print(f"Total training time: {results['training_time']:.2f} seconds ({results['training_time']/60:.2f} minutes)")
    print(f"Best {best_metric_name}: {best_metric:.4f} at epoch {best_epoch}")
    print(f"Final train loss: {results['train_losses'][-1]:.4f}")
    print(f"Final val loss: {results['val_losses'][-1]:.4f}")
    print()
    print(f"Checkpoints saved to: {output_dir}")
    print(f"  - best.pt (epoch {best_epoch})")
    print(f"  - last.pt (epoch {CONFIG['epochs']})")
    print()
    print("=" * 80)
    print("[OK] Training Complete!")
    print("=" * 80)
    
    return results

# ============================================================================
# Report Generation
# ============================================================================

def generate_report(results: dict, output_file: Path):
    """Generate comprehensive markdown report."""
    
    report = []
    report.append("# Medical Augmentation Training Report\n\n")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    report.append("=" * 80 + "\n\n")
    
    # Configuration
    report.append("## Configuration\n\n")
    config = results["config"]
    report.append("### Training Settings\n\n")
    report.append(f"- **Epochs**: {config['epochs']}\n")
    report.append(f"- **Batch Size**: {config['batch_size']}\n")
    report.append(f"- **Learning Rate**: {config['lr']}\n")
    report.append(f"- **Image Size**: {config['image_size']}\n")
    report.append(f"- **Device**: {results.get('device', 'unknown')}\n")
    report.append(f"- **Number of Classes**: {results.get('num_classes', 'unknown')}\n\n")
    
    report.append("### Augmentation Settings\n\n")
    report.append(f"- **Medical Augmentation**: {config['use_medical_augmentation']}\n")
    report.append(f"- **Preserve Marker**: {config['preserve_marker']}\n")
    report.append(f"- **Intensity**: {config['intensity']}\n\n")
    
    # Results
    report.append("## Results\n\n")
    report.append(f"- **Best Metric**: {results['best_metric']:.4f}\n")
    report.append(f"- **Best Epoch**: {results['best_epoch']}\n")
    report.append(f"- **Training Time**: {results['training_time']:.2f} seconds ({results['training_time']/60:.2f} minutes)\n")
    report.append(f"- **Training Start**: {results['training_start']}\n")
    report.append(f"- **Training End**: {results['training_end']}\n\n")
    
    # Final metrics
    if "final_metrics" in results and results["final_metrics"]:
        report.append("## Final Metrics\n\n")
        for key, value in results["final_metrics"].items():
            if isinstance(value, (int, float)):
                report.append(f"- **{key}**: {value:.4f}\n")
            else:
                report.append(f"- **{key}**: {value}\n")
        report.append("\n")
    
    # Loss progression
    if results["train_losses"] and results["val_losses"]:
        report.append("## Loss Progression\n\n")
        report.append("| Epoch | Train Loss | Val Loss |\n")
        report.append("|-------|------------|----------|\n")
        for i, (train_loss, val_loss) in enumerate(zip(results["train_losses"], results["val_losses"])):
            report.append(f"| {i+1} | {train_loss:.4f} | {val_loss:.4f} |\n")
        report.append("\n")
    
    # Metrics progression
    if results["metrics_per_epoch"]:
        report.append("## Metrics Progression\n\n")
        first_metrics = results["metrics_per_epoch"][0]
        metric_keys = [k for k in first_metrics.keys() if isinstance(first_metrics[k], (int, float))]
        
        if metric_keys:
            header = "| Epoch | " + " | ".join(metric_keys) + " |\n"
            report.append(header)
            report.append("|-------|" + "|".join(["---" for _ in metric_keys]) + "|\n")
            
            for i, metrics in enumerate(results["metrics_per_epoch"]):
                row = f"| {i+1} | " + " | ".join([f"{metrics.get(k, 0):.4f}" for k in metric_keys]) + " |\n"
                report.append(row)
            report.append("\n")
    
    # Analysis
    report.append("## Analysis\n\n")
    
    if results["train_losses"] and results["val_losses"]:
        initial_train_loss = results["train_losses"][0]
        final_train_loss = results["train_losses"][-1]
        initial_val_loss = results["val_losses"][0]
        final_val_loss = results["val_losses"][-1]
        
        train_improvement = ((initial_train_loss - final_train_loss) / initial_train_loss) * 100
        val_improvement = ((initial_val_loss - final_val_loss) / initial_val_loss) * 100
        
        report.append("### Loss Improvement\n\n")
        report.append(f"- **Train Loss**: {initial_train_loss:.4f} -> {final_train_loss:.4f} ({train_improvement:+.2f}%)\n")
        report.append(f"- **Val Loss**: {initial_val_loss:.4f} -> {final_val_loss:.4f} ({val_improvement:+.2f}%)\n\n")
    
    if results["metrics_per_epoch"]:
        first_metric = results["metrics_per_epoch"][0].get("combined_AP50", results["metrics_per_epoch"][0].get("bbox_AP50", 0.0))
        best_metric = results["best_metric"]
        if first_metric > 0:
            metric_improvement = ((best_metric - first_metric) / first_metric) * 100
            report.append("### Metric Improvement\n\n")
            report.append(f"- **Initial Metric**: {first_metric:.4f}\n")
            report.append(f"- **Best Metric**: {best_metric:.4f}\n")
            report.append(f"- **Improvement**: {metric_improvement:+.2f}%\n\n")
    
    # Save report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(report)

# ============================================================================
# CLI Training Function (for backward compatibility)
# ============================================================================

def run_training_cli():
    """CLI interface for training (backward compatibility)."""
    parser = argparse.ArgumentParser(description="Wound Detection Training")
    parser.add_argument("--data-root", default="data", help="Path to data root")
    parser.add_argument("--train-ann", required=True, help="Path to training annotation file")
    parser.add_argument("--val-ann", required=True, help="Path to validation annotation file")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Run strictly 1 batch for sanity check")
    
    args = parser.parse_args()
    
    # Validate annotation files exist
    if not os.path.exists(args.train_ann):
        raise FileNotFoundError(f"Training annotation file not found: {args.train_ann}")
    if not os.path.exists(args.val_ann):
        raise FileNotFoundError(f"Validation annotation file not found: {args.val_ann}")
    
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = create_dataset(args.data_root, args.train_ann, train=True)
    val_dataset = create_dataset(args.data_root, args.val_ann, train=False)
    
    if args.dry_run:
        indices = range(4)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        val_dataset = torch.utils.data.Subset(val_dataset, indices)
        print("Dry run: reduced dataset size.")

    train_loader, val_loader = make_dataloaders(train_dataset, val_dataset, batch_size=args.batch_size)
    
    # Model
    base_dataset = train_dataset
    if isinstance(train_dataset, torch.utils.data.Subset):
        base_dataset = train_dataset.dataset

    if hasattr(base_dataset, 'coco_json'):
        num_classes = len(base_dataset.coco_json['categories']) + 1
    else:
        num_classes = len(base_dataset.coco['categories']) + 1
    print(f"Detected {num_classes-1} classes + background")
    
    model = build_model(num_classes)
    model.to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    best_metric = 0.0
    
    for epoch in range(args.epochs):
        loss_dict = train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss_dict = validate_one_epoch(model, val_loader, device)
        
        print(f"Epoch {epoch} Val Loss: {val_loss_dict['total_loss']:.4f}")
        
        metrics = evaluate_metrics(model, val_loader, device)
        
        current_metric = metrics.get("combined_AP50", metrics.get("bbox_AP50", metrics.get("f1", 0.0)))
        is_best = current_metric > best_metric
        if is_best:
            best_metric = current_metric
            
        save_checkpoint({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric
        }, args.output_dir, "last.pt", is_best)
        
        lr_scheduler.step()

# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    try:
        results = main()
        if results:
            print("\n[OK] Script completed successfully!")
            sys.exit(0)
        else:
            print("\n[ERROR] Script completed with errors.")
            sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
