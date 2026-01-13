import os
import torch
import torch.nn as nn
import torch.optim as optim
import math
import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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

# Import pipeline utils if running from same dir, or handle relative import
try:
    from pipeline_utils import get_device, create_dataset, make_dataloaders, set_seed
except ImportError:
    # Fallback for CLI usage depending on where it's run
    sys.path.append(str(Path(__file__).parent))
    from pipeline_utils import get_device, create_dataset, make_dataloaders, set_seed

def build_model(num_classes: int, hidden_layer: int = 256, pretrained_backbone: bool = True):
    """
    Builds the Mask R-CNN model with a ResNet-50-FPN backbone.
    Args:
        num_classes (int): Number of classes (including background).
        hidden_layer (int): Size of the mask predictor hidden layer.
        pretrained_backbone (bool): Whether to use pretrained backbone weights.
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
    Returns averaged losses.
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
        # Otherwise, step per-epoch in the main training loop
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
    """
    # Important: To get loss from torchvision models, we must be in train mode.
    # However, we disable grad.
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

@torch.no_grad()
def evaluate_metrics(
    model: nn.Module, 
    data_loader: torch.utils.data.DataLoader, 
    device: torch.device
):
    """
    Evaluates model using COCO metrics (if available) or fallback custom metrics.
    Returns both bbox and segmentation metrics for Mask R-CNN.
    """
    model.eval()
    cpu_device = torch.device("cpu")
    
    dataset = data_loader.dataset
    if isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset

    if hasattr(dataset, 'coco') and HAS_COCO and hasattr(dataset, 'ann_file'):
        print("Using COCO evaluator...")
        # Use COCO API object
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
                    # Mask R-CNN returns masks as [N, 1, H, W] float in [0, 1]
                    masks = output["masks"]
                    # Threshold to binary and squeeze channel dimension
                    masks_binary = (masks > 0.5).squeeze(1).byte()
                    masks_np = masks_binary.numpy()
                
                for i, box in enumerate(boxes):
                    # COCO bbox format: [x, y, w, h]
                    # Output is [x1, y1, x2, y2]
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
                        # Convert to RLE using pycocotools
                        # mask_util.encode expects Fortran order (row-major) and uint8
                        if mask.dtype != np.uint8:
                            mask = mask.astype(np.uint8)
                        rle = mask_util.encode(np.asfortranarray(mask))
                        # Decode bytes to string if needed
                        if isinstance(rle['counts'], bytes):
                            rle['counts'] = rle['counts'].decode('utf-8')
                        
                        res_segm = res_bbox.copy()
                        res_segm["segmentation"] = rle
                        coco_results_segm.append(res_segm)
        
        if not coco_results_bbox:
            print("⚠️  No predictions generated (all scores below threshold or no detections).")
            print("   This may indicate:")
            print("   - Model needs more training")
            print("   - Learning rate too high/low")
            print("   - Try lowering confidence threshold in inference")
            return {"bbox_AP": 0.0, "bbox_AP50": 0.0, "bbox_AP75": 0.0, "combined_AP50": 0.0}
        
        try:
            # Fix missing 'info' field in COCO dataset if needed
            # COCO API stores dataset dict in coco_gt.dataset attribute
            try:
                if hasattr(coco_gt, 'dataset') and isinstance(coco_gt.dataset, dict):
                    if 'info' not in coco_gt.dataset:
                        coco_gt.dataset['info'] = {
                            "description": "Wound Infection Detection Dataset",
                            "version": "1.0",
                            "year": 2025
                        }
            except (AttributeError, TypeError):
                # If we can't access dataset, try to work around it
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
                
                # Combined score for checkpointing
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
            
            # Track raw predictions
            total_raw_predictions += len(pred_scores)
            if len(pred_scores) > 0:
                all_raw_scores.extend(pred_scores.tolist())
            
            # Filter low confidence (threshold 0.5 for fallback metrics)
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
            # area1: (N,), area2: (M,)
            # inter: (N, M)
            box_area = lambda boxes: (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            area1 = box_area(gt_boxes)
            area2 = box_area(pred_boxes_filtered)
            
            lt = torch.max(gt_boxes[:, None, :2], pred_boxes_filtered[:, :2])
            rb = torch.min(gt_boxes[:, None, 2:], pred_boxes_filtered[:, 2:])
            wh = (rb - lt).clamp(min=0)
            inter = wh[:, :, 0] * wh[:, :, 1]
            
            union = area1[:, None] + area2 - inter
            iou = inter / union
            
            # Simple matching: for each GT, find max IoU prediction
            # If max IoU > threshold, TP. Else FN.
            # Used predictions count as FP? 
            # Simplified P/R calculation:
            
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
    
    # Print diagnostic information
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    if len(all_raw_scores) > 0:
        print(f"  Raw predictions: {total_raw_predictions}, Filtered (score>0.5): {total_filtered_predictions}")
        print(f"  Score stats: min={min(all_raw_scores):.3f}, max={max(all_raw_scores):.3f}, mean={np.mean(all_raw_scores):.3f}")
    else:
        print(f"  ⚠️  No raw predictions generated - model may need more training")
    
    # Return metrics dict compatible with checkpointing logic
    return {
        "precision": precision, 
        "recall": recall, 
        "f1": f1,
        "bbox_AP50": f1,  # Use f1 as proxy for AP50 in fallback
        "combined_AP50": f1,
        "raw_predictions_count": total_raw_predictions,
        "filtered_predictions_count": total_filtered_predictions
    }

def save_checkpoint(
    state: Dict, 
    output_dir: str, 
    filename: str = "last.pt", 
    is_best: bool = False
):
    """
    Saves checkpoint. Always saves as 'last.pt', and also as 'best.pt' if is_best=True.
    
    Args:
        state: Dictionary containing model state, optimizer, scheduler, epoch, best_metric, etc.
        output_dir: Directory to save checkpoints
        filename: Filename for the checkpoint (default: "last.pt")
        is_best: If True, also save as "best.pt"
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    torch.save(state, save_path)
    
    if is_best:
        best_path = os.path.join(output_dir, "best.pt")
        torch.save(state, best_path)
        epoch = state.get('epoch', 'unknown')
        best_metric = state.get('best_metric', 'unknown')
        metric_name = state.get('best_metric_name', 'metric')
        print(f"  → Checkpoint saved: {save_path}")
        print(f"  → Best model saved: {best_path} (epoch {epoch}, {metric_name}={best_metric:.4f})")
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
    For trusted checkpoints, uses weights_only=False to allow numpy scalars.
    """
    import numpy as np
    
    # For PyTorch 2.6+, we need to handle numpy scalars specially
    # Try to get numpy scalar type (different paths for different numpy versions)
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
        # This is safe for trusted checkpoints from our own training
        # numpy_scalar should already be in safe globals from above
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    
    model.load_state_dict(checkpoint["model"])
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint

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
        Dictionary with:
            - 'filtered': dict with boxes, labels, scores, masks (after conf_thresh)
            - 'raw': dict with all predictions before filtering
            - 'num_detections': number of filtered detections
            - 'num_raw': number of raw detections
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
        # Assume [C, H, W] format, convert to numpy
        if image.dim() == 3:
            img = image.permute(1, 2, 0).cpu().numpy()
            # Denormalize if needed (assuming ImageNet normalization)
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
        wound_class_ids: List of class IDs for wound regions (e.g., [1, 2])
        infection_class_ids: List of class IDs for infection indicators (e.g., [3, 4, 5])
        marker_class_id: Class ID for the reference marker (e.g., 6)
        marker_size_cm2: Physical size of marker in cm² (default 3x3 cm = 9 cm²)
        
    Returns:
        Dictionary with:
            - wound_area_cm2: Total wound area in cm² (None if marker not found)
            - wound_area_ratio: Wound area / image area
            - marker_found: bool
            - infection_flags: dict mapping class_id -> (detected: bool, max_score: float)
            - num_detections: total number of detections
            - detections: list of all detections with details
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
            # Use the marker with highest confidence
            marker_idx = marker_indices[np.argmax(scores[marker_indices])]
            marker_found = True
            
            # Compute marker area from mask if available
            if masks is not None:
                marker_mask = masks[marker_idx, 0]  # [H, W]
                marker_area_pixels = np.sum(marker_mask > 0.5)
            else:
                # Fallback: compute from bounding box
                x1, y1, x2, y2 = boxes[marker_idx]
                marker_area_pixels = (x2 - x1) * (y2 - y1)
            
            # Compute conversion ratio
            if marker_area_pixels > 0:
                pixel_to_cm2_ratio = marker_size_cm2 / marker_area_pixels
    
    # Compute wound area
    wound_area_pixels = 0.0
    wound_area_cm2 = None
    
    if wound_class_ids is not None:
        wound_indices = np.where(np.isin(labels, wound_class_ids))[0]
        if len(wound_indices) > 0 and masks is not None:
            # Sum all wound mask areas
            for idx in wound_indices:
                wound_mask = masks[idx, 0]
                wound_area_pixels += np.sum(wound_mask > 0.5)
            
            # Convert to cm² if marker found
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

def run_training_cli():
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
    
    # Create datasets with separate train/val annotation files
    train_dataset = create_dataset(args.data_root, args.train_ann, train=True)
    val_dataset = create_dataset(args.data_root, args.val_ann, train=False)
    
    if args.dry_run:
        indices = range(4)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        val_dataset = torch.utils.data.Subset(val_dataset, indices)
        print("Dry run: reduced dataset size.")

    train_loader, val_loader = make_dataloaders(train_dataset, val_dataset, batch_size=args.batch_size)
    
    # Model
    # Determine num_classes automatically from dataset if possible, or assume +1 for background
    # Standard COCO has 80 classes. If we have custom classes, we should count them.
    # We'll assume the annotation file has categories.
    
    base_dataset = train_dataset
    if isinstance(train_dataset, torch.utils.data.Subset):
        base_dataset = train_dataset.dataset

    # Get num_classes from coco_json if available, otherwise from coco
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
        
        # Determine best metric (prefer combined_AP50, then bbox_AP50, then f1)
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

if __name__ == "__main__":
    run_training_cli()

# ====================================================================
# How to run:
# 1. Ensure pipeline_utils.py is in the same directory.
# 2. Run via CLI:
#    python training_engine.py --data-root /path/to/data --ann-file /path/to/annotations.json --epochs 50
# 3. Or import in a notebook as shown in fixed_training_pipeline.ipynb.
# ====================================================================

