"""
Wound-Only Segmentation Training
==================================

Mask R-CNN wound-only baseline. Single class: wound.
Dataset: data/wound_focus_clean/ (train_wound_only.json, val_wound_only.json, test_wound_only.json).

Usage:
    cd experiments/maskrcnn
    python train_model.py
    python train_model.py --config improved   # Improved pipeline (768px, cosine LR, lighter aug)
    python train_model.py --epochs 1   # Quick test

Outputs:
    - checkpoints/ (best_model.pth, last_checkpoint.pth, training_history.json)
    - results/ (metrics, plots, predictions, baseline_vs_improved_comparison.json when --config improved)
    - reports/ (baseline) or reports_wound_only/ (improved)
"""

import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import pycocotools.mask as mask_util
    HAS_COCO = True
except ImportError:
    HAS_COCO = False
    mask_util = None

# Path setup
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if (SCRIPTS_DIR / "augmentation_strategy.py").exists():
    sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_utils import (
    set_seed,
    get_device,
    create_dataset,
    make_dataloaders,
    WOUND_ONLY_CLASSES,
)

# Fix encoding for Windows (only if not in Jupyter/IPython)
if sys.platform == "win32":
    try:
        from IPython import get_ipython
        _in_jupyter = get_ipython() is not None
    except ImportError:
        _in_jupyter = False
    if not _in_jupyter:
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
                sys.stderr.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

# ============================================================================
# Configuration
# ============================================================================

CONFIG_BASELINE = {
    "data_root": str(PROJECT_ROOT / "data" / "wound_focus_clean"),
    "ann_file_train": str(PROJECT_ROOT / "data" / "wound_focus_clean" / "train_wound_only.json"),
    "ann_file_val": str(PROJECT_ROOT / "data" / "wound_focus_clean" / "val_wound_only.json"),
    "ann_file_test": str(PROJECT_ROOT / "data" / "wound_focus_clean" / "test_wound_only.json"),
    "output_dir": str(SCRIPT_DIR / "checkpoints"),
    "results_dir": str(SCRIPT_DIR / "results"),
    "reports_dir": str(SCRIPT_DIR / "reports"),
    "seed": 42,
    "batch_size": 2,
    "num_workers": 0,
    "epochs": 50,
    "lr": 0.001,
    "image_size": (512, 512),
    "early_stop_patience": 12,
    "early_stop_min_delta": 0.003,
    "loss_clip_max": 100.0,
    "loss_skip_threshold": 1000.0,
    "skip_invalid_targets": True,
    "device_prefer_cuda": True,
    "use_medical_augmentation": True,
    "preserve_marker": True,
    "intensity": "moderate",
    "num_qualitative_samples": 8,
    "conf_thresh_qualitative": 0.5,
    "mask_eval_threshold": 0.5,
    "lr_schedule": "step",  # step | cosine
    # Pixel-to-cm conversion when no marker: 1 cm ≈ N pixels (typical 512×512 wound image, ~20cm FOV)
    "pixels_per_cm": 26.0,
}

def _get_config_improved():
    c = dict(CONFIG_BASELINE)
    c.update({
        "image_size": (768, 768),
        "intensity": "light",
        "early_stop_min_delta": 0.005,
        "lr_schedule": "cosine",
        "reports_dir": str(SCRIPT_DIR / "reports_wound_only"),
        "pixels_per_cm": 38.4,  # 768×768: ~20cm FOV
    })
    return c

CONFIG_IMPROVED = _get_config_improved()

# Active config (set by --config CLI)
CONFIG = dict(CONFIG_BASELINE)

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

# ============================================================================
# Model Building
# ============================================================================

def build_model(num_classes: int, hidden_layer: int = 256, pretrained_backbone: bool = True):
    """Build Mask R-CNN ResNet-50-FPN for wound-only (num_classes=2)."""
    weights = "DEFAULT" if pretrained_backbone else None
    model = maskrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model


def validate_dataset_labels(
    train_loader: torch.utils.data.DataLoader,
    num_classes: int,
    num_batches: int = 5,
) -> List[int]:
    """Validate labels in [1, num_classes-1]. Returns unique labels seen."""
    all_labels: List[int] = []
    batches_checked = 0
    for batch in train_loader:
        _, targets = batch
        for target in targets:
            if "labels" in target:
                all_labels.extend(target["labels"].tolist())
        batches_checked += 1
        if batches_checked >= num_batches:
            break
    unique = sorted(set(all_labels))
    if batches_checked > 0 and len(all_labels) == 0:
        raise ValueError("No labels found in sampled batches. Check dataset annotations.")
    min_label, max_label = 1, num_classes - 1
    for label in unique:
        if label < min_label or label > max_label:
            raise ValueError(
                f"Dataset label {label} out of range [1, {max_label}]. "
                f"Model num_classes={num_classes}."
            )
    return unique


# ============================================================================
# Wound-Only Dataset Validation (pre-training)
# ============================================================================

# Expected counts from dataset_build_report.md
_EXPECTED_TRAIN = 257
_EXPECTED_VAL = 57
_EXPECTED_TEST = 55
_EXPECTED_CATEGORIES = [{"id": 1, "name": "wound"}]


def _validate_categories(coco: dict, split_name: str) -> bool:
    """Verify categories are exactly one wound class."""
    cats = coco.get("categories", [])
    if len(cats) != 1:
        print(f"  [FAIL] {split_name}: expected 1 category, got {len(cats)}")
        return False
    if cats[0] != _EXPECTED_CATEGORIES[0]:
        print(f"  [FAIL] {split_name}: expected {_EXPECTED_CATEGORIES[0]}, got {cats[0]}")
        return False
    print(f"  [OK] {split_name}: categories = {cats}")
    return True


def _validate_image_paths(coco: dict, root: Path, split_name: str) -> bool:
    """Verify all image paths resolve and files exist."""
    def _norm(p: str) -> str:
        return str(p).replace("\\", "/")
    images = coco.get("images", [])
    missing = []
    for img in images:
        fn = _norm(img.get("file_name", ""))
        if not fn:
            missing.append((img.get("id"), "empty file_name"))
            continue
        full_path = root / fn
        if not full_path.exists():
            missing.append((img.get("id"), str(full_path)))
    if missing:
        for img_id, path in missing[:5]:
            print(f"  [FAIL] {split_name}: image {img_id} not found: {path}")
        if len(missing) > 5:
            print(f"  [FAIL] {split_name}: ... and {len(missing) - 5} more")
        return False
    print(f"  [OK] {split_name}: all {len(images)} images found")
    return True


def _validate_annotations(coco: dict, split_name: str) -> bool:
    """Verify annotations have non-empty segmentation and valid bbox."""
    anns = coco.get("annotations", [])
    invalid = []
    for ann in anns:
        seg = ann.get("segmentation", [])
        if not seg:
            invalid.append((ann.get("id"), "empty segmentation"))
            continue
        bbox = ann.get("bbox", [])
        if len(bbox) < 4 or bbox[2] <= 0 or bbox[3] <= 0:
            invalid.append((ann.get("id"), "invalid bbox"))
    if invalid:
        for ann_id, reason in invalid[:5]:
            print(f"  [FAIL] {split_name}: annotation {ann_id}: {reason}")
        if len(invalid) > 5:
            print(f"  [FAIL] {split_name}: ... and {len(invalid) - 5} more")
        return False
    print(f"  [OK] {split_name}: all {len(anns)} annotations valid")
    return True


def _validate_counts(coco: dict, split_name: str, expected: int) -> bool:
    """Verify image count matches build report."""
    n = len(coco.get("images", []))
    if n != expected:
        print(f"  [FAIL] {split_name}: expected {expected} images, got {n}")
        return False
    print(f"  [OK] {split_name}: {n} images (matches expected)")
    return True


def _validate_dataset_sample(root: Path, ann_file: Path, image_size: Tuple[int, int]) -> bool:
    """Sample a few images via WoundDataset and verify masks."""
    try:
        dataset = create_dataset(
            root=str(root),
            annotation_file=str(ann_file),
            train=False,
            image_size=image_size,
            use_medical_augmentation=False,
            target_classes=WOUND_ONLY_CLASSES,
        )
        n_samples = min(5, len(dataset))
        for i in range(n_samples):
            _, target = dataset[i]
            masks = target.get("masks")
            if masks is None or masks.numel() == 0:
                print(f"  [FAIL] Sample {i}: no masks in target")
                return False
            if masks.sum().item() <= 0:
                print(f"  [FAIL] Sample {i}: mask is empty")
                return False
        print(f"  [OK] Sampled {n_samples} images; masks non-empty")
        return True
    except Exception as e:
        print(f"  [FAIL] Dataset sample error: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_wound_dataset(
    data_root: Optional[Path] = None,
    ann_train: Optional[Path] = None,
    ann_val: Optional[Path] = None,
    ann_test: Optional[Path] = None,
    image_size: Tuple[int, int] = (512, 512),
) -> int:
    """
    Pre-training validation for wound-only dataset. Verifies COCO files, categories,
    image paths, annotations, counts, and a WoundDataset sample.
    Returns 0 on success, 1 on any failure.
    """
    data_root = data_root or Path(CONFIG["data_root"])
    ann_train = ann_train or Path(CONFIG["ann_file_train"])
    ann_val = ann_val or Path(CONFIG["ann_file_val"])
    ann_test = ann_test or Path(CONFIG["ann_file_test"])

    print("=" * 60)
    print("Wound-Only Dataset Validation")
    print("=" * 60)
    print(f"Data root: {data_root}")
    print()

    if not data_root.exists():
        print(f"[FAIL] Data root does not exist: {data_root}")
        return 1

    all_ok = True
    for split_name, ann_path, expected in [
        ("train", ann_train, _EXPECTED_TRAIN),
        ("val", ann_val, _EXPECTED_VAL),
        ("test", ann_test, _EXPECTED_TEST),
    ]:
        print(f"--- {split_name} ---")
        if not ann_path.exists():
            print(f"  [FAIL] Annotation file not found: {ann_path}")
            all_ok = False
            continue
        with open(ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)
        if not _validate_categories(coco, split_name):
            all_ok = False
        if not _validate_image_paths(coco, data_root, split_name):
            all_ok = False
        if not _validate_annotations(coco, split_name):
            all_ok = False
        if not _validate_counts(coco, split_name, expected):
            all_ok = False
        print()

    print("--- Dataset sample (WoundDataset) ---")
    if not _validate_dataset_sample(data_root, ann_train, image_size):
        all_ok = False
    print()

    print("=" * 60)
    if all_ok:
        print("PASS: All validation checks passed.")
        return 0
    else:
        print("FAIL: One or more validation checks failed.")
        return 1


# ============================================================================
# Training Helpers
# ============================================================================

def _is_valid_target(target: Dict) -> bool:
    boxes = target.get("boxes")
    if boxes is None or boxes.numel() == 0:
        return False
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        return False
    if (boxes[:, 2] <= boxes[:, 0]).any() or (boxes[:, 3] <= boxes[:, 1]).any():
        return False
    labels = target.get("labels")
    if labels is not None and len(boxes) != len(labels):
        return False
    masks = target.get("masks")
    if masks is not None:
        if masks.numel() == 0 or masks.sum().item() <= 0:
            return False
        if len(boxes) != len(masks):
            return False
    return True


def _filter_valid_batch(
    images: List[torch.Tensor],
    targets: List[Dict],
) -> Tuple[List[torch.Tensor], List[Dict]]:
    valid_images, valid_targets = [], []
    for image, target in zip(images, targets):
        if _is_valid_target(target):
            valid_images.append(image)
            valid_targets.append(target)
    return valid_images, valid_targets


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
    max_norm: float = 1.0,
    loss_clip_max: Optional[float] = None,
    loss_skip_threshold: Optional[float] = None,
    skip_invalid_targets: bool = True,
) -> Dict[str, float]:
    """Train one epoch. Returns averaged losses."""
    model.train()
    metric_logger = {}
    header = f"Epoch: [{epoch}]"
    total_loss_accum = 0.0
    num_batches = len(data_loader)
    valid_batches = 0
    skipped_batches = 0

    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if skip_invalid_targets:
            images, targets = _filter_valid_batch(images, targets)
            if len(images) == 0:
                skipped_batches += 1
                continue

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.item()
        if not math.isfinite(loss_value):
            skipped_batches += 1
            continue
        if loss_skip_threshold is not None and loss_value > loss_skip_threshold:
            skipped_batches += 1
            continue
        if loss_clip_max is not None:
            losses = torch.clamp(losses, max=loss_clip_max)
            loss_value = losses.item()

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

        if scheduler is not None and scheduler_step_per_iter:
            scheduler.step()

        total_loss_accum += loss_value
        valid_batches += 1
        for k, v in loss_dict.items():
            metric_logger[k] = metric_logger.get(k, 0.0) + v.item()

        if i % print_freq == 0:
            print(f"{header} [{i}/{num_batches}] Loss: {loss_value:.4f}")

    avg_loss = total_loss_accum / max(1, valid_batches)
    avg_components = {k: v / max(1, valid_batches) for k, v in metric_logger.items()}
    avg_components["total_loss"] = avg_loss
    avg_components["skipped_batches"] = skipped_batches
    return avg_components


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_clip_max: Optional[float] = None,
    loss_skip_threshold: Optional[float] = None,
    skip_invalid_targets: bool = True,
) -> Dict[str, float]:
    """Compute validation loss (not metrics). Use evaluate_metrics for AP."""
    model.train()
    total_loss_accum = 0.0
    metric_logger = {}
    valid_batches = 0
    skipped_batches = 0

    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if skip_invalid_targets:
            images, targets = _filter_valid_batch(images, targets)
            if len(images) == 0:
                skipped_batches += 1
                continue

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        if not math.isfinite(loss_value):
            skipped_batches += 1
            continue
        if loss_skip_threshold is not None and loss_value > loss_skip_threshold:
            skipped_batches += 1
            continue
        if loss_clip_max is not None:
            losses = torch.clamp(losses, max=loss_clip_max)
            loss_value = losses.item()
        total_loss_accum += loss_value
        valid_batches += 1
        for k, v in loss_dict.items():
            metric_logger[k] = metric_logger.get(k, 0.0) + v.item()

    avg_loss = total_loss_accum / max(1, valid_batches)
    avg_components = {k: v / max(1, valid_batches) for k, v in metric_logger.items()}
    avg_components["total_loss"] = avg_loss
    avg_components["skipped_batches"] = skipped_batches
    return avg_components


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate_metrics(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict:
    """Evaluate COCO metrics (bbox, segm, combined_AP50) or fallback."""
    model.eval()
    cpu_device = torch.device("cpu")
    dataset = data_loader.dataset
    if isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset

    if hasattr(dataset, "coco") and HAS_COCO and hasattr(dataset, "ann_file"):
        print("Using COCO evaluator...")
        coco_gt = dataset.coco
        inv_class_mapping = {}
        if hasattr(dataset, "class_mapping") and dataset.class_mapping:
            inv_class_mapping = {v: k for k, v in dataset.class_mapping.items()}

        coco_results_bbox = []
        coco_results_segm = []
        _debug_logged = False

        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

            for idx, (target, output) in enumerate(zip(targets, outputs)):
                image_id = target["image_id"].item()
                img_info = coco_gt.loadImgs(image_id)
                if not img_info:
                    continue
                img_info = img_info[0]
                orig_h, orig_w = img_info["height"], img_info["width"]
                _, pred_h, pred_w = images[idx].shape
                scale_x = orig_w / pred_w
                scale_y = orig_h / pred_h

                boxes = output["boxes"].tolist()
                scores = output["scores"].tolist()
                labels = output["labels"].tolist()
                has_masks = "masks" in output and len(output["masks"]) > 0
                masks_np = None
                if has_masks:
                    masks = output["masks"]
                    masks_binary = (masks > 0.5).squeeze(1).byte()
                    masks_np = masks_binary.numpy()

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    w, h = max(0, x2 - x1), max(0, y2 - y1)
                    x_orig = x1 * scale_x
                    y_orig = y1 * scale_y
                    w_orig = w * scale_x
                    h_orig = h * scale_y
                    res_bbox = {
                        "image_id": image_id,
                        "category_id": inv_class_mapping.get(int(labels[i]), int(labels[i])),
                        "bbox": [x_orig, y_orig, w_orig, h_orig],
                        "score": float(scores[i]),
                    }
                    coco_results_bbox.append(res_bbox)
                    if has_masks and i < len(masks_np):
                        mask = masks_np[i]
                        if mask.dtype != np.uint8:
                            mask = mask.astype(np.uint8)
                        if mask.shape[0] != orig_h or mask.shape[1] != orig_w:
                            mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                        rle = mask_util.encode(np.asfortranarray(mask))
                        if isinstance(rle["counts"], bytes):
                            rle["counts"] = rle["counts"].decode("utf-8")
                        if not _debug_logged:
                            print(
                                f"[COCO eval] pred=({pred_h},{pred_w}) orig=({orig_h},{orig_w}) "
                                f"bbox={len(coco_results_bbox)} segm={len(coco_results_segm)+1}"
                            )
                            _debug_logged = True
                        res_segm = res_bbox.copy()
                        res_segm["segmentation"] = rle
                        coco_results_segm.append(res_segm)

        print(f"[COCO eval] Total: bbox={len(coco_results_bbox)} segm={len(coco_results_segm)}")
        if not coco_results_bbox:
            print("⚠️  No predictions generated.")
            return {"bbox_AP": 0.0, "bbox_AP50": 0.0, "bbox_AP75": 0.0, "combined_AP50": 0.0}

        try:
            if hasattr(coco_gt, "dataset") and isinstance(coco_gt.dataset, dict) and "info" not in coco_gt.dataset:
                coco_gt.dataset["info"] = {"description": "Wound Infection Detection", "version": "1.0", "year": 2025}
            coco_dt_bbox = coco_gt.loadRes(coco_results_bbox)
            coco_eval_bbox = COCOeval(coco_gt, coco_dt_bbox, "bbox")
            coco_eval_bbox.evaluate()
            coco_eval_bbox.accumulate()
            coco_eval_bbox.summarize()
            metrics = {
                "bbox_AP": coco_eval_bbox.stats[0],
                "bbox_AP50": coco_eval_bbox.stats[1],
                "bbox_AP75": coco_eval_bbox.stats[2],
            }
            if coco_results_segm:
                coco_dt_segm = coco_gt.loadRes(coco_results_segm)
                coco_eval_segm = COCOeval(coco_gt, coco_dt_segm, "segm")
                coco_eval_segm.evaluate()
                coco_eval_segm.accumulate()
                coco_eval_segm.summarize()
                metrics.update({
                    "segm_AP": coco_eval_segm.stats[0],
                    "segm_AP50": coco_eval_segm.stats[1],
                    "segm_AP75": coco_eval_segm.stats[2],
                })
                metrics["combined_AP50"] = (metrics["bbox_AP50"] + metrics["segm_AP50"]) / 2.0
            else:
                metrics["combined_AP50"] = metrics["bbox_AP50"]
            return metrics
        except Exception as e:
            print(f"COCO eval failed: {e}. Falling back to custom metrics.")

    # Fallback
    print("Running fallback metrics (Precision/Recall @ IoU 0.5)...")
    tp = fp = fn = 0
    iou_threshold = 0.5
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        for target, output in zip(targets, outputs):
            gt_boxes = target["boxes"]
            pred_boxes = output["boxes"]
            pred_scores = output["scores"]
            keep = pred_scores > 0.5
            pred_boxes_filtered = pred_boxes[keep]
            if len(gt_boxes) == 0:
                fp += len(pred_boxes_filtered)
                continue
            if len(pred_boxes_filtered) == 0:
                fn += len(gt_boxes)
                continue
            box_area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
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
                if max_iou > iou_threshold and not matched_pred[max_idx]:
                    matched_gt[i] = True
                    matched_pred[max_idx] = True
                    tp += 1
            fn += len(gt_boxes) - matched_gt.sum().item()
            fp += len(pred_boxes_filtered) - matched_pred.sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "bbox_AP50": f1,
        "combined_AP50": f1,
    }


# ============================================================================
# Checkpoints
# ============================================================================

def save_best_checkpoint(
    model: nn.Module,
    epoch: int,
    best_combined_AP50: float,
    bbox_AP50: float,
    segm_AP50: float,
    config: Dict,
    class_mapping: Optional[Dict] = None,
    output_dir: Union[str, Path] = None,
    filename: str = "best_model.pth",
):
    """Save best model by combined_AP50."""
    state = {
        "model": model.state_dict(),
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_combined_AP50": best_combined_AP50,
        "bbox_AP50": bbox_AP50,
        "segm_AP50": segm_AP50,
        "config": config,
        "class_mapping": class_mapping or {},
    }
    out = Path(output_dir) if output_dir else Path(".")
    out.mkdir(parents=True, exist_ok=True)
    torch.save(state, out / filename)
    print(f"  → Best model saved: {out / filename} (epoch {epoch}, combined_AP50={best_combined_AP50:.4f})")


def save_last_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    metrics: Dict,
    output_dir: Path,
    filename: str = "last_checkpoint.pth",
    scaler: Optional[object] = None,
):
    """Save last checkpoint for resume."""
    out = Path(output_dir) if output_dir else Path(".")
    out.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler and hasattr(scaler, "state_dict") else None,
        "epoch": epoch,
        "metrics": metrics,
    }
    torch.save(state, out / filename)
    print(f"  → Last checkpoint saved: {out / filename} (epoch {epoch})")


def load_checkpoint(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[object] = None,
) -> Dict:
    """Load checkpoint. Supports best_model.pth and last_checkpoint.pth."""
    numpy_scalar = None
    try:
        numpy_scalar = np._core.multiarray.scalar
    except AttributeError:
        try:
            numpy_scalar = np.core.multiarray.scalar
        except AttributeError:
            pass
    if numpy_scalar and hasattr(torch.serialization, "add_safe_globals"):
        try:
            torch.serialization.add_safe_globals([numpy_scalar])
        except Exception:
            pass

    load_success = False
    if numpy_scalar and hasattr(torch.serialization, "safe_globals"):
        try:
            with torch.serialization.safe_globals([numpy_scalar]):
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            load_success = True
        except Exception:
            pass
    if not load_success:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    model_state = checkpoint.get("model") or checkpoint.get("model_state_dict")
    if model_state is None:
        raise KeyError(f"Checkpoint missing model state: {path}")
    model.load_state_dict(model_state)
    if optimizer and checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler and checkpoint.get("scheduler"):
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler and checkpoint.get("scaler") and hasattr(scaler, "load_state_dict"):
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


# ============================================================================
# Qualitative Predictions & Inference
# ============================================================================

def save_qualitative_predictions(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: Path,
    num_samples: int = 8,
    conf_thresh: float = 0.5,
) -> int:
    """Save prediction overlays to output_dir/predictions/."""
    model.eval()
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    dataset = data_loader.dataset
    if isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    saved_count = 0

    with torch.no_grad():
        for images, targets in data_loader:
            if saved_count >= num_samples:
                break
            outputs = model(list(img.to(device) for img in images))
            for img_tensor, target, output in zip(images, targets, outputs):
                if saved_count >= num_samples:
                    break
                image_id = target["image_id"].item()
                img_info = dataset.images.get(image_id)
                if not img_info:
                    continue
                file_name = img_info.get("file_name", "")
                stem = Path(file_name).stem if file_name else f"img_{image_id}"

                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                img_np = np.clip(img_np * STD + MEAN, 0, 1)
                img_np = (img_np * 255).astype(np.uint8)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                scores = output["scores"].cpu().numpy()
                masks = output.get("masks")
                if masks is not None:
                    masks = masks.cpu().numpy()
                keep = scores >= conf_thresh
                if not keep.any():
                    conf_str = f"{float(scores.max()) if len(scores) > 0 else 0:.2f}"
                else:
                    h, w = img_np.shape[:2]
                    combined_mask = np.zeros((h, w), dtype=np.uint8)
                    for i in np.where(keep)[0]:
                        m = masks[i, 0]
                        if m.shape[0] != h or m.shape[1] != w:
                            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                        combined_mask = np.maximum(combined_mask, (m > 0.5).astype(np.uint8))
                    overlay = img_np.copy()
                    overlay[combined_mask > 0] = (overlay[combined_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
                    img_np = overlay
                    conf_str = f"{float(scores[keep].max()):.2f}"
                cv2.imwrite(str(pred_dir / f"pred_{stem}_conf_{conf_str}.png"), img_np)
                saved_count += 1
    return saved_count


def save_qualitative_predictions_val_test(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: Path,
    num_val: int = 4,
    num_test: int = 4,
    conf_thresh: float = 0.5,
) -> int:
    """Save predictions to val/ and test/ subdirs for generalization comparison."""
    pred_base = output_dir / "predictions"
    val_dir = pred_base / "val"
    test_dir = pred_base / "test"
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    def _save_to_dir(data_loader, dest_dir, n_samples):
        model.eval()
        dataset = data_loader.dataset
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
        cnt = 0
        with torch.no_grad():
            for images, targets in data_loader:
                if cnt >= n_samples:
                    break
                outputs = model(list(img.to(device) for img in images))
                for img_tensor, target, output in zip(images, targets, outputs):
                    if cnt >= n_samples:
                        break
                    image_id = target["image_id"].item()
                    img_info = dataset.images.get(image_id)
                    if not img_info:
                        continue
                    file_name = img_info.get("file_name", "")
                    stem = Path(file_name).stem if file_name else f"img_{image_id}"
                    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                    img_np = np.clip(img_np * STD + MEAN, 0, 1)
                    img_np = (img_np * 255).astype(np.uint8)
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    scores = output["scores"].cpu().numpy()
                    masks = output.get("masks")
                    if masks is not None:
                        masks = masks.cpu().numpy()
                    keep = scores >= conf_thresh
                    if keep.any() and masks is not None:
                        h, w = img_np.shape[:2]
                        combined_mask = np.zeros((h, w), dtype=np.uint8)
                        for i in np.where(keep)[0]:
                            m = masks[i, 0]
                            if m.shape[0] != h or m.shape[1] != w:
                                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                            combined_mask = np.maximum(combined_mask, (m > 0.5).astype(np.uint8))
                        overlay = img_np.copy()
                        overlay[combined_mask > 0] = (overlay[combined_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
                        img_np = overlay
                    conf_str = f"{float(scores.max()) if len(scores) > 0 else 0:.2f}"
                    cv2.imwrite(str(dest_dir / f"pred_{stem}_conf_{conf_str}.png"), img_np)
                    cnt += 1
        return cnt

    total = _save_to_dir(val_loader, val_dir, num_val) + _save_to_dir(test_loader, test_dir, num_test)
    return total


def save_boundary_quality_examples(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: Path,
    num_good: int = 3,
    num_poor: int = 3,
    conf_thresh: float = 0.5,
) -> int:
    """Save good_boundary_* and poor_boundary_* examples based on confidence and mask quality."""
    model.eval()
    dataset = data_loader.dataset
    if isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    examples = []  # (score, mask_area, img_np, stem, is_good_candidate)

    with torch.no_grad():
        for images, targets in data_loader:
            outputs = model(list(img.to(device) for img in images))
            for img_tensor, target, output in zip(images, targets, outputs):
                image_id = target["image_id"].item()
                img_info = dataset.images.get(image_id)
                if not img_info:
                    continue
                file_name = img_info.get("file_name", "")
                stem = Path(file_name).stem if file_name else f"img_{image_id}"
                scores = output["scores"].cpu().numpy()
                masks = output.get("masks")
                if masks is None or len(scores) == 0:
                    continue
                keep = scores >= conf_thresh
                if not keep.any():
                    continue
                best_idx = np.argmax(scores[keep])
                idx = np.where(keep)[0][best_idx]
                score = float(scores[idx])
                m = masks[idx, 0].cpu().numpy()
                mask_area = float((m > 0.5).sum())
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                img_np = np.clip(img_np * STD + MEAN, 0, 1)
                img_np = (img_np * 255).astype(np.uint8)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                h, w = img_np.shape[:2]
                if m.shape[0] != h or m.shape[1] != w:
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                overlay = img_np.copy()
                overlay[(m > 0.5)] = (overlay[(m > 0.5)] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
                examples.append((score, mask_area, overlay, stem))

    if len(examples) < 2:
        return 0
    examples.sort(key=lambda x: x[0], reverse=True)
    good = examples[:num_good]
    poor = examples[-num_poor:] if len(examples) >= num_poor else examples[-(len(examples) // 2):]
    bnd_dir = output_dir / "predictions"
    bnd_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, (score, area, img, stem) in enumerate(good):
        cv2.imwrite(str(bnd_dir / f"good_boundary_{i+1}_{stem}_conf_{score:.2f}.png"), img)
        saved += 1
    for i, (score, area, img, stem) in enumerate(poor):
        cv2.imwrite(str(bnd_dir / f"poor_boundary_{i+1}_{stem}_conf_{score:.2f}.png"), img)
        saved += 1
    return saved


def calculate_wound_area(
    predictions: Dict,
    marker_class_id: Optional[int] = None,
    marker_size_cm: float = 3.0,
    pixels_per_cm: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute wound area in cm².

    All wound instances (label == 1) are union-merged into a single binary
    mask before computing area, so disjoint or overlapping wound regions are
    both handled correctly.

    With marker: derive pixel_to_cm from the marker mask area (sqrt approach).
    Without: use pixels_per_cm if provided.

    Returns:
        (area_cm2, pixel_to_cm). pixel_to_cm is None when using pixels_per_cm
        fallback or when calibration is unavailable.
    """
    labels = predictions["labels"].cpu().numpy()
    masks = predictions["masks"].cpu().numpy()
    wound_idx = np.where(labels == 1)[0]
    if len(wound_idx) == 0:
        return None, None

    # Union of all wound instances (handles disjoint regions correctly)
    wound_mask = np.zeros(masks.shape[2:], dtype=bool)
    for idx in wound_idx:
        wound_mask |= (masks[idx][0] > 0.5)

    wound_area_pixels = float(wound_mask.sum())
    if wound_area_pixels == 0:
        return None, None

    if marker_class_id is not None:
        marker_idx = np.where(labels == marker_class_id)[0]
        if len(marker_idx) > 0:
            marker_mask = masks[marker_idx[0]][0] > 0.5
            marker_area_pixels = float(marker_mask.sum())
            if marker_area_pixels > 0:
                # marker is 3×3 cm → side = sqrt(area_px) → pixel_to_cm = side_cm / side_px
                pixel_to_cm = marker_size_cm / np.sqrt(marker_area_pixels)
                return wound_area_pixels * (pixel_to_cm ** 2), pixel_to_cm

    if pixels_per_cm is not None and pixels_per_cm > 0:
        area_cm2 = wound_area_pixels / (pixels_per_cm ** 2)
        return area_cm2, None

    # No marker and no pixels_per_cm: return raw pixel count (caller must handle)
    return wound_area_pixels, None


def predict_image(
    image_path: str,
    model: torch.nn.Module,
    device: torch.device,
    data_root: Path,
    image_size: Tuple[int, int] = (512, 512),
    conf_threshold: float = 0.5,
    marker_class_id: Optional[int] = None,
) -> Tuple[Dict, Dict]:
    """Predict on image path. Returns (result_dict, filtered_predictions). Infection from filename."""
    model.eval()
    img_path = Path(image_path)
    if not img_path.is_absolute():
        candidate = data_root / image_path
        img_path = candidate if candidate.exists() else Path(image_path)
    image = cv2.imread(str(img_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, image_size)
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std

    with torch.no_grad():
        output = model([image_tensor.to(device)])[0]
    keep = output["scores"] >= conf_threshold
    filtered = {
        "boxes": output["boxes"][keep],
        "labels": output["labels"][keep],
        "scores": output["scores"][keep],
        "masks": output["masks"][keep],
    }

    pixels_per_cm = CONFIG.get("pixels_per_cm")
    wound_area_cm2, pixel_to_cm = calculate_wound_area(
        filtered, marker_class_id=marker_class_id, pixels_per_cm=pixels_per_cm
    )
    wound_area_px = None
    if wound_area_cm2 is not None and pixel_to_cm is None and pixels_per_cm is None:
        wound_area_px = int(wound_area_cm2)
        wound_area_cm2 = None
    stem = img_path.stem
    has_infection = "not_infected" not in stem.lower()
    result = {
        "image_path": str(img_path),
        "num_detections": len(filtered["labels"]),
        "wound_area_cm2": float(wound_area_cm2) if wound_area_cm2 else None,
        "wound_area_px": int(wound_area_px) if wound_area_px else None,
        "has_infection": has_infection,
        "infection_label": "Infected" if has_infection else "Not infected",
    }
    return result, filtered


def visualize_prediction(
    image_path: str,
    predictions: Dict,
    data_root: Path,
    image_size: Tuple[int, int] = (512, 512),
    wound_color: Tuple[int, int, int] = (0, 255, 0),
    box_color: Tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """Overlay wound mask and bbox. Returns RGB image."""
    img_path = Path(image_path)
    if not img_path.is_absolute():
        candidate = data_root / image_path
        img_path = candidate if candidate.exists() else Path(image_path)
    image = cv2.imread(str(img_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
    image = cv2.resize(image, image_size)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    masks = predictions["masks"].cpu().numpy()
    labels = predictions["labels"].cpu().numpy()
    scores = predictions["scores"].cpu().numpy()
    boxes = predictions["boxes"].cpu().numpy()
    wound_idx = np.where(labels == 1)[0]
    if len(wound_idx) == 0:
        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    best_idx = wound_idx[np.argmax(scores[wound_idx])]
    mask = masks[best_idx][0] > 0.5
    box = boxes[best_idx]
    overlay = image_rgb.copy()
    overlay[mask] = overlay[mask] * 0.55 + np.array(wound_color) * 0.45
    result_img = np.clip(overlay, 0, 255).astype(np.uint8)
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(result_img, (x1, y1), (x2, y2), box_color, 2)
    score = float(scores[best_idx])
    text = f"Wound {score:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(result_img, (x1, y1 - th - 6), (x1 + tw + 4, y1), box_color, -1)
    cv2.putText(result_img, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return result_img


def predict_single_image(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    sample_index: int = 0,
    conf_thresh: float = 0.5,
) -> Tuple[np.ndarray, Dict]:
    """Predict on single dataset sample. Returns (image_with_overlay, info_dict)."""
    model.eval()
    dataset = data_loader.dataset
    if isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    img_tensor, target = dataset[sample_index]
    image_id = target["image_id"].item()
    img_info = dataset.images.get(image_id)
    file_name = img_info.get("file_name", "") if img_info else ""
    stem = Path(file_name).stem if file_name else f"img_{image_id}"
    has_infection = "not_infected" not in stem.lower()
    infection_label = "Infected" if has_infection else "Not infected"

    with torch.no_grad():
        output = model([img_tensor.to(device)])[0]
    boxes = output["boxes"].cpu().numpy()
    scores = output["scores"].cpu().numpy()
    masks = output.get("masks")
    if masks is not None:
        masks = masks.cpu().numpy()

    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np * STD + MEAN, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h, w = img_np.shape[:2]

    wound_area_px = 0
    wound_area_cm2 = None
    best_box = None
    best_score = 0.0
    keep = scores >= conf_thresh
    if keep.any() and masks is not None:
        idx = np.argmax(scores[keep])
        keep_indices = np.where(keep)[0]
        best_idx = keep_indices[idx]
        best_box = boxes[best_idx]
        best_score = float(scores[best_idx])
        m = masks[best_idx, 0]
        if m.shape[0] != h or m.shape[1] != w:
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        binary = (m > 0.5).astype(np.uint8)
        wound_area_px = int(np.sum(binary))
        pixels_per_cm = CONFIG.get("pixels_per_cm")
        if pixels_per_cm and pixels_per_cm > 0:
            wound_area_cm2 = wound_area_px / (pixels_per_cm ** 2)
        overlay = img_np.copy()
        overlay[binary > 0] = (overlay[binary > 0] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
        img_np = overlay
        x1, y1, x2, y2 = map(int, best_box)
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 0, 255), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = 30
    area_text = f"{wound_area_cm2:.2f} cm²" if wound_area_cm2 is not None else f"{wound_area_px} px"
    cv2.putText(img_np, f"Wound area: {area_text}", (10, y_offset), font, 0.7, (0, 0, 0), 2)
    cv2.putText(img_np, f"Infection: {infection_label}", (10, y_offset + 35), font, 0.7, (0, 255, 0) if has_infection else (0, 165, 255), 2)
    if best_score > 0:
        cv2.putText(img_np, f"Conf: {best_score:.2f}", (10, y_offset + 70), font, 0.7, (0, 0, 0), 2)

    info = {
        "file_name": file_name,
        "wound_area_px": wound_area_px,
        "wound_area_cm2": round(wound_area_cm2, 2) if wound_area_cm2 is not None else None,
        "infection": infection_label,
        "has_infection": has_infection,
        "confidence": best_score,
        "sample_index": sample_index,
    }
    return img_np, info


# ============================================================================
# Plotting & Reports
# ============================================================================

def save_training_curves(train_losses: List[float], val_losses: List[float], output_path: Path) -> None:
    """Save train vs val loss curve."""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Loss", marker="o", markersize=3)
    ax.plot(epochs, val_losses, label="Val Loss", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train vs Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_ap_curves(metrics_per_epoch: List[Dict], output_dir: Path) -> None:
    """Save bbox AP, segm AP, combined_AP50 in one chart."""
    if not metrics_per_epoch:
        return
    epochs = range(1, len(metrics_per_epoch) + 1)
    first = metrics_per_epoch[0] or {}
    fig, ax = plt.subplots(figsize=(10, 6))
    for k in ["bbox_AP", "bbox_AP50", "bbox_AP75"]:
        if k in first:
            ax.plot(epochs, [m.get(k, 0) for m in metrics_per_epoch], label=k, marker="o", markersize=3)
    for k in ["segm_AP", "segm_AP50", "segm_AP75"]:
        if k in first:
            ax.plot(epochs, [m.get(k, 0) for m in metrics_per_epoch], label=k, marker="s", markersize=3)
    if "combined_AP50" in first:
        ax.plot(epochs, [m.get("combined_AP50", 0) for m in metrics_per_epoch], label="combined_AP50", marker="^", markersize=3, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AP")
    ax.set_title("Bbox, Segmentation & Combined AP Metrics")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "ap_curves.png", dpi=150)
    plt.close(fig)


def display_results_curves(results_dir: Path, max_show: int = 2) -> None:
    """Display saved curve images (training_curves.png, ap_curves.png) in Jupyter."""
    try:
        from IPython.display import display, Image as IPImage
    except ImportError:
        return
    curve_files = [
        Path(results_dir) / "training_curves.png",
        Path(results_dir) / "ap_curves.png",
    ]
    for fp in curve_files[:max_show]:
        if fp.exists():
            display(IPImage(filename=str(fp)))


def display_results_predictions(results_dir: Path, n_show: int = 8) -> None:
    """Display qualitative prediction images in a grid (Jupyter)."""
    try:
        from IPython.display import display
    except ImportError:
        return
    pred_dir = Path(results_dir) / "predictions"
    if not pred_dir.exists():
        return
    pred_files = sorted(pred_dir.glob("*.png"))
    n_show = min(n_show, len(pred_files))
    if n_show == 0:
        return
    cols = min(4, n_show)
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()
    for i, fp in enumerate(pred_files[:n_show]):
        img = cv2.imread(str(fp))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(img)
        axes[i].set_title(fp.stem.replace("pred_", "").replace("_", " "), fontsize=8)
        axes[i].axis("off")
    for i in range(n_show, len(axes)):
        axes[i].axis("off")
    fig.suptitle(f"Qualitative Predictions ({n_show} samples)", fontsize=14)
    fig.tight_layout()
    plt.show()


def generate_wound_only_report(results: Dict, output_dir: Path, test_metrics: Optional[Dict] = None) -> None:
    """Generate wound_only_training_report.md and review_summary_for_chatgpt.md."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config = results.get("config", CONFIG)
    best_metric = results.get("best_metric", 0)
    best_epoch = results.get("best_epoch", 0)
    best_bbox = results.get("best_bbox_AP50", 0)
    best_segm = results.get("best_segm_AP50", 0)
    train_size = results.get("train_size", "?")
    val_size = results.get("val_size", "?")
    test_size = results.get("test_size", "?")
    training_time = results.get("training_time", 0)

    report_lines = [
        "# Wound-Only Segmentation Training Report\n\n",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        "## Purpose\n\nSingle class: wound. Dataset: wound_focus_clean.\n\n",
        f"## Dataset\n\nTrain: {config.get('ann_file_train', '')}\nVal: {config.get('ann_file_val', '')}\nTest: {config.get('ann_file_test', '')}\n\n",
        f"## Model\n\nMask R-CNN ResNet-50-FPN, num_classes=2, image_size={config.get('image_size')}\n\n",
        f"## Sizes\n\nTrain: {train_size}, Val: {val_size}, Test: {test_size}\n\n",
        f"## Best Metrics\n\nBest epoch: {best_epoch}\ncombined_AP50: {best_metric:.4f}\nbbox_AP50: {best_bbox:.4f}\nsegm_AP50: {best_segm:.4f}\n\n",
        f"Training time: {training_time:.2f}s\n\n",
    ]
    if test_metrics:
        report_lines.append("## Test Metrics\n\n")
        for k, v in test_metrics.items():
            if isinstance(v, (int, float)):
                report_lines.append(f"- {k}: {v:.4f}\n")
        report_lines.append("\n")
    report_lines.append("See results/predictions/ for qualitative outputs.\n\n")

    with open(output_dir / "wound_only_training_report.md", "w", encoding="utf-8") as f:
        f.writelines(report_lines)

    review_lines = [
        "# Wound-Only Baseline — Review Summary\n\n",
        f"Train: {train_size}, Val: {val_size}, Test: {test_size}\n\n",
        f"combined_AP50: {best_metric:.4f}, bbox_AP50: {best_bbox:.4f}, segm_AP50: {best_segm:.4f}\n\n",
        "Review results/predictions/ for qualitative outputs.\n\n",
    ]
    with open(output_dir / "review_summary_for_chatgpt.md", "w", encoding="utf-8") as f:
        f.writelines(review_lines)
    print(f"  → Reports saved: {output_dir}")


def generate_improved_report(
    results: Dict,
    output_dir: Path,
    test_metrics: Optional[Dict],
    baseline_metrics: Optional[Dict],
) -> None:
    """Generate wound_only_improved_training_report.md and review_summary_for_chatgpt_improved.md."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config = results.get("config", CONFIG)
    best_metric = results.get("best_metric", 0)
    best_epoch = results.get("best_epoch", 0)
    best_bbox = results.get("best_bbox_AP50", 0)
    best_segm = results.get("best_segm_AP50", 0)
    train_size = results.get("train_size", "?")
    val_size = results.get("val_size", "?")
    test_size = results.get("test_size", "?")
    training_time = results.get("training_time", 0)

    bl_val = baseline_metrics.get("best_validation", {}) if baseline_metrics else {}
    bl_test = baseline_metrics.get("test", {}) if baseline_metrics else {}
    imp_test = test_metrics or {}

    gap_baseline = 0.0
    if bl_val and bl_test:
        v = bl_val.get("combined_AP50", 0)
        t = bl_test.get("combined_AP50", 0)
        gap_baseline = (v - t) / v if v > 0 else 0
    gap_improved = 0.0
    if best_metric > 0 and imp_test:
        t = imp_test.get("combined_AP50", 0)
        gap_improved = (best_metric - t) / best_metric if best_metric > 0 else 0

    report_lines = [
        "# Wound-Only Improved Training Report\n\n",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        "## Changes Made\n\n",
        "- Train/val resize consistency: LongestMaxSize + PadIfNeeded (was Resize for val)\n",
        "- Image size: 768x768 (was 512x512) for better boundary detail\n",
        "- Augmentation intensity: light (was moderate) for better generalization\n",
        "- LR schedule: CosineAnnealingLR (was StepLR)\n",
        "- early_stop_min_delta: 0.005 (was 0.003)\n\n",
        "## Rationale\n\n",
        "Resize consistency reduces train/val distribution shift. Higher resolution preserves boundary pixels. ",
        "Lighter augmentation reduces overfitting to train-only artifacts. Cosine LR avoids aggressive decay.\n\n",
        f"## Best Validation Metrics (Improved)\n\n",
        f"- combined_AP50: {best_metric:.4f}\n",
        f"- bbox_AP50: {best_bbox:.4f}\n",
        f"- segm_AP50: {best_segm:.4f}\n",
        f"- Best epoch: {best_epoch}\n\n",
        "## Test Metrics (Improved)\n\n",
    ]
    for k, v in (imp_test or {}).items():
        if isinstance(v, (int, float)):
            report_lines.append(f"- {k}: {v:.4f}\n")
    report_lines.append("\n## Baseline vs Improved Comparison\n\n")
    report_lines.append("| Metric | Baseline (val) | Baseline (test) | Improved (val) | Improved (test) |\n")
    report_lines.append("|--------|----------------|-----------------|----------------|------------------|\n")
    for k in ["combined_AP50", "bbox_AP50", "segm_AP50", "segm_AP75"]:
        bv = bl_val.get(k, "-")
        bt = bl_test.get(k, "-")
        iv = best_metric if k == "combined_AP50" else (best_bbox if k == "bbox_AP50" else (best_segm if k == "segm_AP50" else "-"))
        it = imp_test.get(k, "-")
        bv_s = f"{bv:.4f}" if isinstance(bv, (int, float)) else str(bv)
        bt_s = f"{bt:.4f}" if isinstance(bt, (int, float)) else str(bt)
        iv_s = f"{iv:.4f}" if isinstance(iv, (int, float)) else str(iv)
        it_s = f"{it:.4f}" if isinstance(it, (int, float)) else str(it)
        report_lines.append(f"| {k} | {bv_s} | {bt_s} | {iv_s} | {it_s} |\n")
    report_lines.append(f"\nVal-test gap: baseline ~{gap_baseline*100:.1f}%, improved ~{gap_improved*100:.1f}%\n\n")
    report_lines.append("## Qualitative Outputs\n\n")
    report_lines.append("See results/predictions/val/, results/predictions/test/, good_boundary_*, poor_boundary_*\n\n")
    report_lines.append("## Remaining Limitations\n\n")
    report_lines.append("Segmentation quality depends on annotation consistency. AP75 may remain low with coarse GT.\n\n")
    report_lines.append("## Recommended Next Step\n\n")
    report_lines.append("If metrics improved: consider 1024px or test-time refinement. If not: tune augmentation or data.\n\n")

    with open(output_dir / "wound_only_improved_training_report.md", "w", encoding="utf-8") as f:
        f.writelines(report_lines)

    review_lines = [
        "# Wound-Only Improved — Review Summary for ChatGPT\n\n",
        "## Baseline Issues\n\n",
        "- Validation-to-test performance drop (~25% combined_AP50)\n",
        "- Low segm_AP75 (0.003) — poor fine boundary precision\n",
        "- Coarse wound masks\n\n",
        "## Changes Made\n\n",
        "Resize consistency, 768px, light augmentation, CosineAnnealingLR.\n\n",
        "## Best New Metrics\n\n",
        f"Val: combined_AP50={best_metric:.4f}, bbox_AP50={best_bbox:.4f}, segm_AP50={best_segm:.4f}\n",
        f"Test: {imp_test.get('combined_AP50', 0):.4f} combined_AP50, {imp_test.get('segm_AP75', 0):.4f} segm_AP75\n\n",
        "## Comparison (Baseline vs Improved)\n\n",
        f"Baseline test combined_AP50: {bl_test.get('combined_AP50', 0):.4f}\n",
        f"Improved test combined_AP50: {imp_test.get('combined_AP50', 0):.4f}\n",
        f"Baseline test segm_AP75: {bl_test.get('segm_AP75', 0):.4f}\n",
        f"Improved test segm_AP75: {imp_test.get('segm_AP75', 0):.4f}\n\n",
        "## Boundary Precision / Generalization\n\n",
        "Review qualitative outputs to confirm. Delta in baseline_vs_improved_comparison.json.\n\n",
        "## Unresolved Issues\n\n",
        "Dataset size and annotation quality remain limiting factors.\n\n",
        "## Recommendation for Next Stage\n\n",
        "If improved run shows better test metrics: proceed with inference pipeline. Else: data augmentation or architecture tuning.\n\n",
    ]
    with open(output_dir / "review_summary_for_chatgpt_improved.md", "w", encoding="utf-8") as f:
        f.writelines(review_lines)
    print(f"  → Improved reports saved: {output_dir}")


# ============================================================================
# Main
# ============================================================================

def main() -> Optional[Dict]:
    """Main training loop for wound-only baseline."""
    print("=" * 80)
    print("Wound-Only Segmentation Training")
    print("=" * 80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    if validate_wound_dataset() != 0:
        print("[ERROR] Dataset validation failed.")
        return None

    set_seed(CONFIG["seed"])
    device = get_device(CONFIG.get("device_prefer_cuda", True))
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    data_root = Path(CONFIG["data_root"])
    train_ann = Path(CONFIG["ann_file_train"])
    val_ann = Path(CONFIG["ann_file_val"])
    test_ann = Path(CONFIG["ann_file_test"])
    output_dir = Path(CONFIG["output_dir"])
    results_dir = Path(CONFIG["results_dir"])
    reports_dir = Path(CONFIG["reports_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # For improved config: preserve baseline metrics for comparison
    baseline_metrics: Optional[Dict] = None
    if str(reports_dir).endswith("reports_wound_only"):
        baseline_path = results_dir / "metrics_summary.json"
        if baseline_path.exists():
            try:
                with open(baseline_path, "r", encoding="utf-8") as f:
                    baseline_metrics = json.load(f)
                with open(results_dir / "baseline_metrics.json", "w", encoding="utf-8") as f:
                    json.dump(baseline_metrics, f, indent=2, default=str)
                print(f"  → Preserved baseline metrics to baseline_metrics.json")
            except Exception as e:
                print(f"  [WARNING] Could not preserve baseline: {e}")

    print("Loading datasets...")
    train_dataset = create_dataset(
        root=str(data_root),
        annotation_file=str(train_ann),
        train=True,
        image_size=CONFIG["image_size"],
        use_medical_augmentation=CONFIG["use_medical_augmentation"],
        preserve_marker=CONFIG["preserve_marker"],
        intensity=CONFIG["intensity"],
        target_classes=WOUND_ONLY_CLASSES,
    )
    val_dataset = create_dataset(
        root=str(data_root),
        annotation_file=str(val_ann),
        train=False,
        image_size=CONFIG["image_size"],
        use_medical_augmentation=CONFIG["use_medical_augmentation"],
        preserve_marker=CONFIG["preserve_marker"],
        intensity=CONFIG["intensity"],
        target_classes=WOUND_ONLY_CLASSES,
    )
    test_dataset = create_dataset(
        root=str(data_root),
        annotation_file=str(test_ann),
        train=False,
        image_size=CONFIG["image_size"],
        use_medical_augmentation=CONFIG["use_medical_augmentation"],
        preserve_marker=CONFIG["preserve_marker"],
        intensity=CONFIG["intensity"],
        target_classes=WOUND_ONLY_CLASSES,
    )
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    test_size = len(test_dataset)
    print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}\n")

    train_loader, val_loader = make_dataloaders(
        train_dataset, val_dataset,
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        collate_fn=lambda b: tuple(zip(*b)),
        pin_memory=torch.cuda.is_available(),
    )

    base_ds = train_dataset.dataset if isinstance(train_dataset, torch.utils.data.Subset) else train_dataset
    num_classes = base_ds.num_classes
    print(f"Model num_classes: {num_classes}")
    model = build_model(num_classes=num_classes, pretrained_backbone=True)
    model.to(device)

    print("\n--- Startup report ---")
    unique_labels = validate_dataset_labels(train_loader, num_classes, num_batches=5)
    print(f"  Unique labels: {unique_labels}")
    print("--------------------\n")

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=CONFIG["lr"],
        momentum=0.9,
        weight_decay=0.0005,
    )
    lr_schedule = CONFIG.get("lr_schedule", "step")
    if lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CONFIG["epochs"], eta_min=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    class_mapping = getattr(base_ds, "class_mapping", {})

    results = {
        "config": CONFIG.copy(),
        "train_losses": [],
        "val_losses": [],
        "metrics_per_epoch": [],
        "best_metric": 0.0,
        "best_epoch": 0,
        "best_bbox_AP50": 0.0,
        "best_segm_AP50": 0.0,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "training_time": 0.0,
        "device": str(device),
        "num_classes": num_classes,
    }

    best_combined_AP50 = 0.0
    best_epoch = 0
    early_stop_patience = CONFIG.get("early_stop_patience", 12)
    epochs_without_improve = 0

    print("Starting training...")
    start_time = time.time()

    try:
        for epoch in range(CONFIG["epochs"]):
            print(f"\nEpoch [{epoch + 1}/{CONFIG['epochs']}]")
            print("-" * 60)

            train_stats = train_one_epoch(
                model, optimizer, train_loader, device, epoch,
                scheduler=scheduler,
                scheduler_step_per_iter=False,
                loss_clip_max=CONFIG.get("loss_clip_max"),
                loss_skip_threshold=CONFIG.get("loss_skip_threshold"),
                skip_invalid_targets=CONFIG.get("skip_invalid_targets", True),
            )
            results["train_losses"].append(train_stats["total_loss"])

            val_stats = validate_one_epoch(
                model, val_loader, device,
                loss_clip_max=CONFIG.get("loss_clip_max"),
                loss_skip_threshold=CONFIG.get("loss_skip_threshold"),
                skip_invalid_targets=CONFIG.get("skip_invalid_targets", True),
            )
            results["val_losses"].append(val_stats["total_loss"])

            print("Evaluating metrics...")
            metrics = evaluate_metrics(model, val_loader, device)
            results["metrics_per_epoch"].append(metrics)

            combined_AP50 = metrics.get("combined_AP50", 0.0)
            bbox_AP50 = metrics.get("bbox_AP50", 0.0)
            segm_AP50 = metrics.get("segm_AP50", bbox_AP50)

            print(f"Train Loss: {train_stats['total_loss']:.4f} | Val Loss: {val_stats['total_loss']:.4f}")
            print(f"combined_AP50: {combined_AP50:.4f} | bbox_AP50: {bbox_AP50:.4f} | segm_AP50: {segm_AP50:.4f}")

            if combined_AP50 > best_combined_AP50:
                best_combined_AP50 = combined_AP50
                best_epoch = epoch + 1
                results["best_metric"] = best_combined_AP50
                results["best_epoch"] = best_epoch
                results["best_bbox_AP50"] = bbox_AP50
                results["best_segm_AP50"] = segm_AP50
                epochs_without_improve = 0
                print(f"  ✓ NEW BEST! combined_AP50: {best_combined_AP50:.4f} (Epoch {best_epoch})")
                save_best_checkpoint(
                    model, epoch=epoch + 1,
                    best_combined_AP50=best_combined_AP50,
                    bbox_AP50=bbox_AP50,
                    segm_AP50=segm_AP50,
                    config=CONFIG.copy(),
                    class_mapping=class_mapping,
                    output_dir=output_dir,
                    filename="best_model.pth",
                )
            else:
                epochs_without_improve += 1

            save_last_checkpoint(
                model, optimizer, scheduler, epoch + 1, metrics,
                output_dir=output_dir,
                filename="last_checkpoint.pth",
                scaler=None,
            )
            scheduler.step()

            if early_stop_patience > 0 and epochs_without_improve >= early_stop_patience:
                print(f"Early stopping after {epochs_without_improve} epochs without improvement.")
                break

    except KeyboardInterrupt:
        print("\n[WARNING] Training interrupted")
        results["interrupted"] = True
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        results["error"] = str(e)

    results["training_time"] = time.time() - start_time

    best_path = output_dir / "best_model.pth"
    if best_path.exists():
        load_checkpoint(model, str(best_path))

    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)
    test_metrics = None
    try:
        test_metrics = evaluate_metrics(model, test_loader, device)
        results["test_metrics"] = test_metrics
        for k, v in test_metrics.items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    except Exception as e:
        print(f"[WARNING] Test evaluation failed: {e}")
        results["test_metrics"] = {}

    metrics_summary = {
        "best_validation": {
            "combined_AP50": results["best_metric"],
            "bbox_AP50": results["best_bbox_AP50"],
            "segm_AP50": results["best_segm_AP50"],
            "best_epoch": results["best_epoch"],
        },
        "test": results.get("test_metrics", {}),
        "config": CONFIG,
    }
    with open(results_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, default=str)

    # For improved config: save improved metrics and comparison
    if baseline_metrics is not None:
        with open(results_dir / "improved_metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=2, default=str)
        bl_val = baseline_metrics.get("best_validation", {})
        bl_test = baseline_metrics.get("test", {})
        imp_val = metrics_summary["best_validation"]
        imp_test = metrics_summary.get("test", {})
        delta = {}
        for k in ["bbox_AP50", "segm_AP50", "combined_AP50", "segm_AP75", "bbox_AP75"]:
            if k in imp_test and k in bl_test:
                delta[f"test_{k}"] = round(imp_test[k] - bl_test[k], 4)
        comparison = {
            "baseline": {"val": bl_val, "test": bl_test},
            "improved": {"val": imp_val, "test": imp_test},
            "delta": delta,
        }
        with open(results_dir / "baseline_vs_improved_comparison.json", "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, default=str)
        print(f"  → Saved improved_metrics_summary.json and baseline_vs_improved_comparison.json")

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump({"train_losses": results["train_losses"], "val_losses": results["val_losses"], "metrics_per_epoch": results["metrics_per_epoch"], "config": CONFIG}, f, indent=2, default=str)

    print("\nSaving plots...")
    save_training_curves(results["train_losses"], results["val_losses"], results_dir / "training_curves.png")
    save_ap_curves(results["metrics_per_epoch"], results_dir)

    print("\nSaving qualitative predictions...")
    conf_thresh = CONFIG.get("conf_thresh_qualitative", 0.5)
    if str(reports_dir).endswith("reports_wound_only"):
        n_val_test = save_qualitative_predictions_val_test(
            model, val_loader, test_loader, device, results_dir,
            num_val=4, num_test=4, conf_thresh=conf_thresh,
        )
        n_boundary = save_boundary_quality_examples(
            model, test_loader, device, results_dir,
            num_good=3, num_poor=3, conf_thresh=conf_thresh,
        )
        print(f"  → Saved {n_val_test} to predictions/val/ and predictions/test/, {n_boundary} boundary examples")
    else:
        n_saved = save_qualitative_predictions(
            model, test_loader, device, results_dir,
            num_samples=CONFIG.get("num_qualitative_samples", 8),
            conf_thresh=conf_thresh,
        )
        print(f"  → Saved {n_saved} images to {results_dir / 'predictions'}")

    print("\nGenerating reports...")
    if baseline_metrics is not None:
        generate_improved_report(results, reports_dir, test_metrics, baseline_metrics)
    else:
        generate_wound_only_report(results, reports_dir, test_metrics)

    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    print(f"Best combined_AP50: {results['best_metric']:.4f} at epoch {best_epoch}")
    print(f"Training time: {results['training_time']:.2f}s ({results['training_time']/60:.2f} min)")
    print(f"Checkpoints: {output_dir}")
    print(f"Results: {results_dir}")
    print(f"Reports: {reports_dir}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Wound-Only Segmentation Training")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs (quick test)")
    parser.add_argument("--config", type=str, default="baseline", choices=["baseline", "improved"],
                        help="Config preset: baseline or improved")
    parser.add_argument("--validate-only", action="store_true",
                        help="Run dataset validation only, then exit")
    args = parser.parse_args()
    if args.config == "improved":
        CONFIG.clear()
        CONFIG.update(CONFIG_IMPROVED)
        print(f"[Config] Using IMPROVED preset (768px, light aug, cosine LR)")
    if args.epochs is not None:
        CONFIG["epochs"] = args.epochs
        print(f"[Override] epochs={args.epochs}")
    if args.validate_only:
        sys.exit(validate_wound_dataset())
    try:
        results = main()
        sys.exit(0 if results else 1)
    except Exception as e:
        print(f"\n[ERROR] Fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
