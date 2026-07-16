"""
YOLO11m + U-Net++ Wound Detection & Segmentation — Training Script
====================================================================

**Primary training entry point for this experiment:** ``training_pipeline.ipynb``
(run with kernel cwd = ``experiments/YOLO11m_UNetPP``). This module provides the
same functions for import and an optional CLI for automation — not the main
workflow for thesis runs.

Self-contained training, evaluation, inference and reporting for the
combined YOLO11m-seg + U-Net++ pipeline.

Stages (CLI, optional):
    python train_model.py --stage convert    # COCO -> YOLO label format
    python train_model.py --stage yolo       # Train YOLO11m-seg
    python train_model.py --stage unet       # Train U-Net++
    python train_model.py --stage combined   # Combined inference + eval
    python train_model.py --stage infection  # Train infection classifier
    python train_model.py --stage all        # Run all stages sequentially

Outputs:
    checkpoints/yolo/   — YOLO best & last weights
    checkpoints/unet/   — U-Net++ best & last weights
    results/yolo/       — YOLO metrics, curves
    results/unet/       — U-Net++ metrics, curves
    results/combined/   — Combined metrics, predictions
    reports/            — Markdown training report
"""

import argparse
import json
import math
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from pipeline_utils import (
    set_seed,
    get_device,
    load_config,
    prepare_yolo_dataset,
    validate_yolo_dataset,
    create_unet_datasets,
    make_unet_dataloaders,
    unet_collate_fn,
    WoundDataset,
    WoundROIDataset,
    get_unet_transforms,
    IMAGENET_MEAN,
    IMAGENET_STD,
    WOUND_ONLY_CLASSES,
)
from experiment_io import (
    get_combined_dirs,
    get_unet_best_checkpoint_path,
    get_unet_dirs,
    snapshot_config,
)
from experiment_provenance import (
    build_experiment_manifest,
    checkpoint_matches_config,
    save_experiment_manifest,
)

from combined.inference import combined_inference
from combined.marker import calculate_pixels_per_cm_from_marker
from combined.coco_eval import evaluate_combined_coco


VALID_RUN_MODES = {"train_from_scratch", "resume", "evaluate_only"}


def normalize_run_mode(run_mode: str) -> str:
    """Validate and normalize the explicit notebook/CLI execution mode."""
    normalized = str(run_mode).strip().lower()
    if normalized not in VALID_RUN_MODES:
        choices = ", ".join(sorted(VALID_RUN_MODES))
        raise ValueError(f"Invalid run_mode={run_mode!r}; choose one of: {choices}")
    return normalized


def should_train_component(
    component: str,
    checkpoint_path: Path,
    manifest_path: Path,
    config: dict,
    run_mode: str,
    *,
    allow_legacy_checkpoint: bool = False,
) -> bool:
    """Return whether a component should train under an explicit run mode.

    In ``evaluate_only`` mode, a checkpoint is mandatory. If a provenance
    manifest exists, a configuration mismatch is rejected. Legacy checkpoints
    without a manifest require an explicit opt-in during migration.
    """
    mode = normalize_run_mode(run_mode)
    if mode == "train_from_scratch":
        return True
    if mode == "resume":
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Cannot resume {component}; checkpoint not found: {checkpoint_path}"
            )
        return True

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Cannot evaluate {component}; checkpoint not found: {checkpoint_path}"
        )
    compatible, reason = checkpoint_matches_config(manifest_path, config)
    if not compatible:
        if allow_legacy_checkpoint and "manifest missing" in reason:
            print(
                f"[MIGRATION WARNING] {component}: {reason}. "
                "Legacy checkpoint explicitly allowed; freeze a manifest before paper use."
            )
        else:
            raise RuntimeError(f"{component} checkpoint rejected: {reason}")
    return False

# Fix Windows console encoding
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
# Stage 1: YOLO11m-seg
# ============================================================================

def build_yolo_model(weights: str = "yolo11m-seg.pt"):
    """Load YOLO11m-seg model from Ultralytics."""
    from ultralytics import YOLO
    model = YOLO(weights)
    return model


def _resolve_yolo_run_name(yolo_project: Path, base_name: str = "train") -> str:
    """
    Pick a YOLO run directory name that will not collide with locked files.

    If the previous ``train/`` folder can be archived, reuse ``train``.
    Otherwise start a timestamped run and leave the old folder untouched.
    """
    run_dir = yolo_project / base_name
    if not run_dir.exists():
        return base_name

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = yolo_project / f"{base_name}_prev_{stamp}"
    try:
        run_dir.rename(archive_dir)
        print(f"  Archived previous YOLO run -> {archive_dir.name}")
        return base_name
    except OSError:
        run_name = f"{base_name}_{stamp}"
        print(
            f"  Previous YOLO run is locked on Windows; "
            f"starting a fresh run -> {run_name}"
        )
        return run_name


def train_yolo(
    config: dict,
    script_dir: Path,
    run_mode: str = "train_from_scratch",
) -> dict:
    """
    Train YOLO11m-seg using Ultralytics API.
    Returns dict with training results summary.
    """
    yolo_cfg = config["yolo"]
    dataset_yaml = script_dir / "yolo_data" / "dataset.yaml"

    if not dataset_yaml.exists():
        print("Dataset YAML not found — running conversion first...")
        dataset_yaml = prepare_yolo_dataset(config, script_dir)

    print("\n" + "=" * 60)
    print("Stage 1: Training YOLO11m-seg")
    print("=" * 60)

    mode = normalize_run_mode(run_mode)
    if mode == "evaluate_only":
        raise ValueError("train_yolo cannot run in evaluate_only mode")
    resume_path = script_dir / "checkpoints" / "yolo" / "train" / "weights" / "last.pt"
    if not resume_path.is_file():
        resume_path = script_dir / "checkpoints" / "yolo" / "last.pt"
    if mode == "resume" and not resume_path.is_file():
        raise FileNotFoundError(f"YOLO resume checkpoint not found: {resume_path}")
    model_weights = (
        str(resume_path)
        if mode == "resume"
        else yolo_cfg.get("model", "yolo11m-seg.pt")
    )
    model = build_yolo_model(model_weights)
    yolo_project = script_dir / "checkpoints" / "yolo"
    yolo_project.mkdir(parents=True, exist_ok=True)
    run_name = _resolve_yolo_run_name(yolo_project, "train")
    train_dir = yolo_project / run_name

    train_kwargs = dict(
        data=str(dataset_yaml),
        imgsz=yolo_cfg.get("image_size", 640),
        epochs=yolo_cfg.get("epochs", 100),
        batch=yolo_cfg.get("batch_size", 8),
        lr0=yolo_cfg.get("lr0", 0.01),
        lrf=yolo_cfg.get("lrf", 0.01),
        optimizer=yolo_cfg.get("optimizer", "SGD"),
        momentum=yolo_cfg.get("momentum", 0.937),
        weight_decay=yolo_cfg.get("weight_decay", 0.0005),
        patience=yolo_cfg.get("patience", 20),
        seed=config.get("seed", 42),
        degrees=yolo_cfg.get("degrees", 10),
        perspective=yolo_cfg.get("perspective", 0.0),
        flipud=yolo_cfg.get("flipud", 0.5),
        fliplr=yolo_cfg.get("fliplr", 0.5),
        mosaic=yolo_cfg.get("mosaic", 0.5),
        mixup=yolo_cfg.get("mixup", 0.0),
        close_mosaic=yolo_cfg.get("close_mosaic", 15),
        hsv_h=yolo_cfg.get("hsv_h", 0.015),
        hsv_s=yolo_cfg.get("hsv_s", 0.7),
        hsv_v=yolo_cfg.get("hsv_v", 0.4),
        dropout=yolo_cfg.get("dropout", 0.0),
        label_smoothing=yolo_cfg.get("label_smoothing", 0.0),
        cos_lr=yolo_cfg.get("cos_lr", False),
        warmup_epochs=yolo_cfg.get("warmup_epochs", 3.0),
        project=str(yolo_project),
        name=run_name,
        exist_ok=False,
        verbose=True,
        workers=config.get("num_workers", 0),
    )
    if mode == "resume":
        train_kwargs["resume"] = True
    train_results = model.train(**train_kwargs)

    print("\nCopying YOLO results...")
    _copy_yolo_outputs(train_dir, script_dir)

    summary = _extract_yolo_metrics(train_dir)
    summary["training_completed"] = True
    summary["run_name"] = run_name
    yolo_best = script_dir / "checkpoints" / "yolo" / "best.pt"
    manifest = build_experiment_manifest(
        config,
        script_dir,
        run_mode=mode,
        checkpoint_paths=[yolo_best],
    )
    save_experiment_manifest(manifest, script_dir / "results" / "yolo")
    save_experiment_manifest(manifest, train_dir)
    return summary


def _copy_yolo_outputs(train_dir: Path, script_dir: Path) -> None:
    """Copy key YOLO training outputs to our results structure."""
    results_yolo = script_dir / "results" / "yolo"
    results_yolo.mkdir(parents=True, exist_ok=True)

    weights_dir = train_dir / "weights"
    if weights_dir.exists():
        for w in ["best.pt", "last.pt"]:
            src = weights_dir / w
            if src.exists():
                dst = script_dir / "checkpoints" / "yolo" / w
                shutil.copy2(src, dst)
                print(f"  -> {dst}")

    for pattern in ["*.png", "*.csv", "*.json"]:
        for f in train_dir.glob(pattern):
            shutil.copy2(f, results_yolo / f.name)


def _extract_yolo_metrics(train_dir: Path) -> dict:
    """Parse YOLO results.csv for final metrics."""
    results_csv = train_dir / "results.csv"
    metrics = {}
    if results_csv.exists():
        import csv
        with open(results_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if rows:
            last = rows[-1]
            for k, v in last.items():
                k = k.strip()
                try:
                    metrics[k] = float(v.strip())
                except (ValueError, AttributeError):
                    metrics[k] = v.strip() if isinstance(v, str) else v
    return metrics


def evaluate_yolo(config: dict, script_dir: Path) -> dict:
    """Run YOLO validation on the test set."""
    yolo_best = script_dir / "checkpoints" / "yolo" / "best.pt"
    if not yolo_best.exists():
        print("[WARNING] No YOLO best.pt found — skipping YOLO evaluation.")
        return {}

    dataset_yaml = script_dir / "yolo_data" / "dataset.yaml"
    print("\n" + "=" * 60)
    print("Evaluating YOLO11m-seg on test set")
    print("=" * 60)

    model = build_yolo_model(str(yolo_best))
    val_results = model.val(
        data=str(dataset_yaml),
        split="test",
        imgsz=config["yolo"].get("image_size", 640),
        batch=config["yolo"].get("batch_size", 8),
        verbose=True,
        workers=config.get("num_workers", 0),
    )

    metrics = {}
    if hasattr(val_results, "box"):
        metrics["bbox_mAP50"] = float(val_results.box.map50)
        metrics["bbox_mAP50_95"] = float(val_results.box.map)
    if hasattr(val_results, "seg"):
        metrics["segm_mAP50"] = float(val_results.seg.map50)
        metrics["segm_mAP50_95"] = float(val_results.seg.map)

    if "bbox_mAP50" in metrics and "segm_mAP50" in metrics:
        metrics["combined_AP50"] = (metrics["bbox_mAP50"] + metrics["segm_mAP50"]) / 2.0

    results_dir = script_dir / "results" / "yolo"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics


def predict_yolo(
    config: dict,
    script_dir: Path,
    num_samples: int = 8,
    conf_thresh: float = 0.5,
) -> int:
    """Save YOLO predictions on test images with area + infection info."""
    yolo_best = script_dir / "checkpoints" / "yolo" / "best.pt"
    if not yolo_best.exists():
        print("[WARNING] No YOLO best.pt — skipping predictions.")
        return 0

    model = build_yolo_model(str(yolo_best))
    dataset_yaml = script_dir / "yolo_data" / "dataset.yaml"

    with open(dataset_yaml, "r", encoding="utf-8") as f:
        ds = yaml.safe_load(f)
    base = Path(ds["path"])
    test_list = base / ds["test"]
    with open(test_list, "r", encoding="utf-8") as f:
        test_images = [l.strip() for l in f if l.strip()]

    pred_dir = script_dir / "results" / "yolo" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    combined_cfg = config.get("combined", {})
    marker_class_id = combined_cfg.get("marker_class_id", 1)
    marker_real_cm = combined_cfg.get("marker_real_cm", 3.0)

    device = get_device()
    classifier_path = script_dir / "checkpoints" / "infection" / "infection_classifier.pth"

    n = min(num_samples, len(test_images))
    for i in range(n):
        results = model(test_images[i], conf=conf_thresh, verbose=False)
        if not results or len(results) == 0:
            continue

        r = results[0]
        plot = r.plot()

        area_cm2 = None
        area_px = 0
        ppcm: Optional[float] = None
        infection_label = _parse_infection_from_filename(Path(test_images[i]).name)
        infection_prob: Optional[float] = None
        confidence: Optional[float] = None

        try:
            if r.masks is not None and r.boxes is not None:
                orig_h, orig_w = r.orig_shape
                classes = r.boxes.cls.cpu().numpy().astype(int)
                confidences = r.boxes.conf.cpu().numpy().astype(float)
                wound_confidences = [
                    float(conf)
                    for cls, conf in zip(classes, confidences)
                    if cls != marker_class_id
                ]
                if wound_confidences:
                    confidence = max(wound_confidences)
                elif len(confidences) > 0:
                    confidence = float(confidences.max())
                masks_data = r.masks.data.cpu().numpy()

                ppcm = calculate_pixels_per_cm_from_marker(
                    r, marker_class_id=marker_class_id, marker_real_cm=marker_real_cm,
                )

                wound_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                for j, cls in enumerate(classes):
                    if cls != marker_class_id:
                        m = masks_data[j]
                        m_resized = cv2.resize(
                            m, (orig_w, orig_h),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        wound_mask = np.maximum(
                            wound_mask, (m_resized > 0.5).astype(np.uint8),
                        )

                area_px = int(wound_mask.sum())
                if ppcm is not None:
                    area_cm2 = calculate_wound_area(wound_mask, ppcm)

                # Infection prediction
                if classifier_path.exists() and area_px > 0:
                    image_rgb = cv2.cvtColor(
                        cv2.imread(test_images[i]), cv2.COLOR_BGR2RGB,
                    )
                    try:
                        inf_result = predict_infection(
                            image_rgb, wound_mask, classifier_path, device,
                        )
                        infection_label = inf_result["predicted"]
                        infection_prob = inf_result["infected_prob"]
                    except Exception:
                        pass
        except Exception:
            pass

        plot = draw_info_panel(
            plot, area_cm2, area_px, ppcm,
            infection_label, infection_prob,
        )

        fname = Path(test_images[i]).stem
        out_path = pred_dir / f"pred_{fname}.png"
        cv2.imwrite(str(out_path), plot)
        with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "image": Path(test_images[i]).name,
                    "confidence": confidence,
                    "wound_area_cm2": area_cm2,
                    "wound_area_px": area_px,
                    "infection": infection_label,
                    "infection_prob": infection_prob,
                    "marker_detected": ppcm is not None,
                },
                f,
                indent=2,
            )

    print(f"  -> Saved {n} YOLO predictions to {pred_dir}")
    return n


# ============================================================================
# Stage 2: U-Net++
# ============================================================================

def build_segmentation_model(config: dict) -> nn.Module:
    """Build the configured ROI segmentation model."""
    import segmentation_models_pytorch as smp

    unet_cfg = config["unet"]
    architecture = str(unet_cfg.get("architecture", "unetplusplus")).lower()
    common_kwargs = {
        "encoder_name": unet_cfg.get("encoder", "efficientnet-b1"),
        "encoder_weights": unet_cfg.get("encoder_weights", "imagenet"),
        "in_channels": unet_cfg.get("in_channels", 3),
        "classes": unet_cfg.get("classes", 1),
        "activation": None,
    }

    if architecture == "unetplusplus":
        return smp.UnetPlusPlus(**common_kwargs)
    if architecture == "deeplabv3plus":
        return smp.DeepLabV3Plus(**common_kwargs)

    raise ValueError(
        f"Unsupported segmentation architecture '{architecture}'. "
        "Supported: unetplusplus, deeplabv3plus."
    )


def build_unet_model(config: dict) -> nn.Module:
    """Backward-compatible alias for the configured ROI segmentation model."""
    return build_segmentation_model(config)


def _mask_to_float01(t: torch.Tensor, *, like: torch.Tensor) -> torch.Tensor:
    """Binary mask for BCE/Dice: always float32 on same device as ``like``."""
    x = t.to(device=like.device, dtype=torch.float32, non_blocking=True)
    if x.numel() > 0 and x.max() > 1.0:
        x = x / 255.0
    return x.clamp(0.0, 1.0)


class DiceLoss(nn.Module):
    """Differentiable Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred.float())
        target = _mask_to_float01(target, like=pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )


class FocalLoss(nn.Module):
    """Binary focal loss for segmentation (focuses on hard boundary pixels)."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = _mask_to_float01(target, like=pred)
        bce = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p_t = torch.sigmoid(pred) * target + (1 - torch.sigmoid(pred)) * (1 - target)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class BCEDiceLoss(nn.Module):
    """Combined BCE + Dice loss (legacy, kept for checkpoint compatibility)."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = _mask_to_float01(target, like=pred)
        return self.bce_weight * self.bce(pred, target) + self.dice_weight * self.dice(pred, target)


class FocalDiceLoss(nn.Module):
    """Combined Focal + Dice loss for better boundary segmentation."""

    def __init__(
        self,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (self.focal_weight * self.focal(pred, target)
                + self.dice_weight * self.dice(pred, target))


def _edge_band_mask(target: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Return a binary band around mask boundaries using morphological gradient."""
    if kernel_size < 3:
        kernel_size = 3
    if kernel_size % 2 == 0:
        kernel_size += 1

    target = _mask_to_float01(target, like=target)
    pad = kernel_size // 2
    dilated = torch.nn.functional.max_pool2d(target, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = 1.0 - torch.nn.functional.max_pool2d(
        1.0 - target,
        kernel_size=kernel_size,
        stride=1,
        padding=pad,
    )
    return ((dilated - eroded) > 0).float()


class BoundaryAwareFocalDiceLoss(nn.Module):
    """Focal+Dice with an auxiliary boundary BCE term for sharper contours."""

    def __init__(
        self,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        boundary_weight: float = 0.15,
        alpha: float = 0.25,
        gamma: float = 2.0,
        boundary_kernel_size: int = 5,
    ):
        super().__init__()
        self.base = FocalDiceLoss(
            focal_weight=focal_weight,
            dice_weight=dice_weight,
            alpha=alpha,
            gamma=gamma,
        )
        self.boundary_weight = boundary_weight
        self.boundary_kernel_size = boundary_kernel_size

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = _mask_to_float01(target, like=pred)
        base_loss = self.base(pred, target)
        edge_mask = _edge_band_mask(target, kernel_size=self.boundary_kernel_size)
        if edge_mask.sum().item() <= 0:
            return base_loss

        bce = torch.nn.functional.binary_cross_entropy_with_logits(pred.float(), target, reduction="none")
        boundary_loss = (bce * edge_mask).sum() / edge_mask.sum().clamp_min(1.0)
        return base_loss + self.boundary_weight * boundary_loss


def train_one_epoch_unet(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    print_freq: int = 10,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> float:
    """Train U-Net++ for one epoch with optional AMP. Returns average loss."""
    model.train()
    use_amp = scaler is not None and device.type == "cuda"
    total_loss = 0.0
    n_batches = 0
    for i, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = _mask_to_float01(masks, like=images)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            preds = model(images)
            loss = criterion(preds, masks)

        if not math.isfinite(loss.item()):
            continue

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if i % print_freq == 0:
            print(f"  Epoch [{epoch}] [{i}/{len(loader)}] Loss: {loss.item():.4f}")

    return total_loss / max(1, n_batches)


@torch.no_grad()
def validate_one_epoch_unet(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Validate U-Net++. Returns average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for images, masks in loader:
        images = images.to(device)
        masks = _mask_to_float01(masks, like=images)
        preds = model(images)
        loss = criterion(preds, masks)
        if math.isfinite(loss.item()):
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate_unet_metrics(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute Dice, IoU, pixel accuracy for U-Net++."""
    model.eval()
    dice_sum, iou_sum, acc_sum = 0.0, 0.0, 0.0
    n_samples = 0
    smooth = 1e-6

    for images, masks in loader:
        images = images.to(device)
        masks = _mask_to_float01(masks, like=images)
        preds = torch.sigmoid(model(images))
        preds_bin = (preds > threshold).float()

        for j in range(preds_bin.shape[0]):
            p = preds_bin[j].view(-1)
            t = masks[j].view(-1)
            inter = (p * t).sum().item()
            union = p.sum().item() + t.sum().item()
            dice_sum += (2 * inter + smooth) / (union + smooth)
            iou_sum += (inter + smooth) / (union - inter + smooth)
            total_px = t.numel()
            correct = ((p == t).float()).sum().item()
            acc_sum += correct / total_px
            n_samples += 1

    n = max(1, n_samples)
    return {
        "dice": dice_sum / n,
        "iou": iou_sum / n,
        "pixel_accuracy": acc_sum / n,
        "n_samples": n_samples,
    }


@torch.no_grad()
def evaluate_unet_fullimage(
    model: nn.Module,
    config: dict,
    script_dir: Path,
    device: torch.device,
    threshold: float = 0.5,
    ann_key: str = "ann_test",
    use_gt_boxes: bool = True,
) -> Dict[str, float]:
    """Evaluate U-Net++ on full images (not ROI crops) for fair comparison with combined.

    For each image:
    1. Load the full image and GT masks
    2. Use GT bboxes (or YOLO-predicted bboxes) to define ROIs
    3. Run U-Net++ on each ROI
    4. Place predicted masks back on the full-image canvas
    5. Compute full-image Dice vs GT
    """
    project_root = script_dir.parent.parent
    data_root = (project_root / config["data_root"]).resolve()
    ann_path = (project_root / config[ann_key]).resolve()
    unet_cfg = config["unet"]
    image_size = tuple(unet_cfg["input_size"])
    roi_padding = unet_cfg.get("roi_padding", 0.1)

    import json as _json
    from pipeline_utils import IMAGENET_MEAN, IMAGENET_STD

    with open(ann_path, "r", encoding="utf-8") as f:
        coco = _json.load(f)

    img_lookup = {img["id"]: img for img in coco["images"]}
    anns_by_img: Dict[int, list] = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    model.eval()
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(device)

    dice_sum, iou_sum = 0.0, 0.0
    n_images = 0
    smooth = 1e-6

    for img_id, img_info in img_lookup.items():
        img_path = data_root / img_info["file_name"]
        if not img_path.exists():
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]

        anns = anns_by_img.get(img_id, [])
        if not anns:
            continue

        gt_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        for ann in anns:
            for seg in ann.get("segmentation", []):
                if len(seg) < 6:
                    continue
                poly = np.array(seg, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(gt_mask, [poly], 1)

        pred_mask = np.zeros((img_h, img_w), dtype=np.float32)
        for ann in anns:
            bbox = ann.get("bbox", [0, 0, 0, 0])
            bx, by, bw, bh = bbox
            x1, y1 = bx, by
            x2, y2 = bx + bw, by + bh

            pad_x = bw * roi_padding
            pad_y = bh * roi_padding
            cx1 = max(0, int(x1 - pad_x))
            cy1 = max(0, int(y1 - pad_y))
            cx2 = min(img_w, int(x2 + pad_x))
            cy2 = min(img_h, int(y2 + pad_y))

            crop = image[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            crop_h, crop_w = crop.shape[:2]
            uh, uw = image_size
            crop_resized = cv2.resize(crop, (uw, uh), interpolation=cv2.INTER_LINEAR)
            t = torch.from_numpy(crop_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            t = (t.to(device) - mean) / std

            probs = torch.sigmoid(model(t)).squeeze().cpu().numpy()
            prob_up = cv2.resize(probs.astype(np.float32), (crop_w, crop_h),
                                 interpolation=cv2.INTER_LINEAR)
            pred_mask[cy1:cy2, cx1:cx2] = np.maximum(pred_mask[cy1:cy2, cx1:cx2], prob_up)

        pred_bin = (pred_mask >= threshold).astype(np.float32)
        gt_f = gt_mask.astype(np.float32)
        inter = (pred_bin * gt_f).sum()
        union = pred_bin.sum() + gt_f.sum()
        dice_sum += (2 * inter + smooth) / (union + smooth)
        iou_sum += (inter + smooth) / (union - inter + smooth)
        n_images += 1

    n = max(1, n_images)
    return {
        "fullimage_dice": dice_sum / n,
        "fullimage_iou": iou_sum / n,
        "n_images": n_images,
    }


def save_unet_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    path: Path,
) -> None:
    """Save U-Net++ checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "metrics": metrics,
    }, path)


def load_unet_checkpoint(model: nn.Module, path: Path, device: torch.device) -> dict:
    """Load U-Net++ checkpoint. Returns the saved metadata dict."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


def resolve_checkpoint_path(
    checkpoint_path: Optional[str],
    script_dir: Path,
) -> Optional[Path]:
    """Resolve an optional checkpoint path from absolute or project-relative input."""
    if not checkpoint_path:
        return None
    p = Path(checkpoint_path)
    if p.is_absolute():
        return p
    candidates = [
        (script_dir / checkpoint_path).resolve(),
        (script_dir.parent.parent / checkpoint_path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_unet_criterion(unet_cfg: dict) -> nn.Module:
    """Build U-Net++ loss from ``config['unet']`` (same logic as ``train_unet`` / notebook)."""
    loss_type = str(unet_cfg.get("loss_type", "focal_dice")).lower()
    if loss_type in {"focal_dice_boundary", "focal_dice_edge"}:
        return BoundaryAwareFocalDiceLoss(
            focal_weight=unet_cfg.get("loss_bce_weight", 0.5),
            dice_weight=unet_cfg.get("loss_dice_weight", 0.5),
            boundary_weight=unet_cfg.get("loss_boundary_weight", 0.15),
            alpha=unet_cfg.get("focal_alpha", 0.25),
            gamma=unet_cfg.get("focal_gamma", 2.0),
            boundary_kernel_size=int(unet_cfg.get("boundary_kernel_size", 5)),
        )
    if loss_type == "focal_dice":
        return FocalDiceLoss(
            focal_weight=unet_cfg.get("loss_bce_weight", 0.5),
            dice_weight=unet_cfg.get("loss_dice_weight", 0.5),
            alpha=unet_cfg.get("focal_alpha", 0.25),
            gamma=unet_cfg.get("focal_gamma", 2.0),
        )
    return BCEDiceLoss(
        bce_weight=unet_cfg.get("loss_bce_weight", 0.5),
        dice_weight=unet_cfg.get("loss_dice_weight", 0.5),
    )


def train_unet(
    config: dict,
    script_dir: Path,
    run_mode: str = "train_from_scratch",
) -> dict:
    """Full U-Net++ training loop with early stopping.

    **Best checkpoint:** ``best_model.pth`` is the weights from the validation epoch
    with the **highest mean Dice** on ``val`` at the deployment threshold
    (``unet.eval_threshold`` or ``combined.unet_mask_thresh``, default 0.5).
    Test metrics are computed after reloading that checkpoint — not from
    ``last_checkpoint.pth``.
    """
    print("\n" + "=" * 60)
    print("Stage 2: Training U-Net++")
    print("=" * 60)

    mode = normalize_run_mode(run_mode)
    if mode == "evaluate_only":
        raise ValueError("train_unet cannot run in evaluate_only mode")
    unet_cfg = config["unet"]
    device = get_device()

    train_ds, val_ds, test_ds = create_unet_datasets(config, script_dir)
    print(f"  ROI samples — Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader, val_loader = make_unet_dataloaders(
        train_ds, val_ds,
        batch_size=unet_cfg.get("batch_size", 16),
        num_workers=config.get("num_workers", 0),
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=unet_cfg.get("batch_size", 16),
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
        collate_fn=unet_collate_fn,
    )

    model = build_unet_model(config)
    model.to(device)
    print(f"  U-Net++ on {device} ({sum(p.numel() for p in model.parameters()):,} params)")

    resume_path = (
        resolve_checkpoint_path(unet_cfg.get("resume_checkpoint"), script_dir)
        if mode == "resume"
        else None
    )
    if resume_path is not None:
        ckpt_meta = load_unet_checkpoint(model, resume_path, device)
        print(f"  Resumed from checkpoint: {resume_path}")
        if isinstance(ckpt_meta, dict) and "epoch" in ckpt_meta:
            print(f"  Resume epoch: {ckpt_meta['epoch']}")
    elif mode == "resume":
        print(
            "  [ERROR] Resume checkpoint not found: "
            f"{unet_cfg.get('resume_checkpoint')}"
        )
        raise FileNotFoundError(str(unet_cfg.get("resume_checkpoint")))

    if bool(unet_cfg.get("freeze_encoder", False)) and hasattr(model, "encoder"):
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("  Encoder frozen for fine-tuning.")

    criterion = build_unet_criterion(unet_cfg)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=unet_cfg.get("lr", 1e-4),
        weight_decay=unet_cfg.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=unet_cfg.get("scheduler_T_max", 50),
        eta_min=unet_cfg.get("scheduler_eta_min", 1e-6),
    )

    epochs = unet_cfg.get("epochs", 50)
    patience = unet_cfg.get("early_stop_patience", 10)
    combined_cfg = config.get("combined", {})
    eval_threshold = float(unet_cfg.get(
        "eval_threshold",
        combined_cfg.get("unet_mask_thresh", 0.5),
    ))
    unet_dirs = get_unet_dirs(script_dir, config)
    ckpt_dir = unet_dirs["checkpoints"]
    results_dir = unet_dirs["results"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = snapshot_config(
        config,
        results_dir,
        Path(config["_config_path"]) if config.get("_config_path") else None,
    )
    print(f"  Checkpoints: {ckpt_dir}")
    print(f"  Results:     {results_dir}")
    print(f"  Config:      {config_snapshot}")

    use_amp = device.type == "cuda" and bool(unet_cfg.get("use_amp", True))
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("  Mixed precision training (AMP) enabled")

    history = {
        "train_losses": [],
        "val_losses": [],
        "dice_per_epoch": [],
        "iou_per_epoch": [],
    }
    best_dice = -1.0
    best_epoch = 0
    epochs_without_improve = 0

    start_time = time.time()
    try:
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch [{epoch}/{epochs}]")
            print("-" * 40)

            train_loss = train_one_epoch_unet(
                model, train_loader, optimizer, criterion, device, epoch,
                scaler=scaler,
            )
            val_loss = validate_one_epoch_unet(model, val_loader, criterion, device)
            metrics = evaluate_unet_metrics(model, val_loader, device, threshold=eval_threshold)
            val_dice = metrics["dice"]
            if not math.isfinite(val_dice):
                print(f"  [WARNING] Non-finite val Dice — skipping best update for this epoch.")
                val_dice = best_dice

            history["train_losses"].append(train_loss)
            history["val_losses"].append(val_loss)
            history["dice_per_epoch"].append(metrics["dice"])
            history["iou_per_epoch"].append(metrics["iou"])

            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Dice: {metrics['dice']:.4f} | IoU: {metrics['iou']:.4f}")

            if val_dice > best_dice:
                best_dice = val_dice
                best_epoch = epoch
                epochs_without_improve = 0
                print(f"  -> NEW BEST val Dice: {best_dice:.4f}")
                save_unet_checkpoint(
                    model, optimizer, scheduler, epoch, metrics,
                    ckpt_dir / "best_model.pth",
                )
            else:
                epochs_without_improve += 1

            save_unet_checkpoint(
                model, optimizer, scheduler, epoch, metrics,
                ckpt_dir / "last_checkpoint.pth",
            )
            scheduler.step()

            if patience > 0 and epochs_without_improve >= patience:
                print(f"  Early stopping after {epochs_without_improve} epochs without improvement.")
                break

    except KeyboardInterrupt:
        print("\n[WARNING] Training interrupted.")

    training_time = time.time() - start_time

    # Load best model for test evaluation
    best_path = ckpt_dir / "best_model.pth"
    if best_path.exists():
        load_unet_checkpoint(model, best_path, device)

    print("\n" + "=" * 60)
    print(f"U-Net++ Test Evaluation (threshold={eval_threshold:.2f})")
    print("=" * 60)
    test_metrics = evaluate_unet_metrics(model, test_loader, device, threshold=eval_threshold)
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    # Save history and metrics
    with open(results_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    summary = {
        "best_dice": best_dice,
        "best_epoch": best_epoch,
        "training_time_s": training_time,
        "test_metrics": test_metrics,
        "config": unet_cfg,
    }
    with open(results_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    manifest = build_experiment_manifest(
        config,
        script_dir,
        run_mode=mode,
        checkpoint_paths=[best_path],
    )
    save_experiment_manifest(manifest, results_dir)

    # Save curves
    save_unet_training_curves(history, results_dir)

    return summary


def save_unet_training_curves(history: dict, results_dir: Path) -> None:
    """Plot and save U-Net++ loss and metric curves."""
    results_dir.mkdir(parents=True, exist_ok=True)
    epochs_range = range(1, len(history["train_losses"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs_range, history["train_losses"], label="Train")
    axes[0].plot(epochs_range, history["val_losses"], label="Val")
    axes[0].set_title("U-Net++ Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE + Dice Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, history["dice_per_epoch"], label="Dice", color="green")
    axes[1].set_title("U-Net++ Dice Score")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs_range, history["iou_per_epoch"], label="IoU", color="orange")
    axes[2].set_title("U-Net++ IoU Score")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("IoU")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(results_dir / "unet_training_curves.png", dpi=150)
    plt.close(fig)
    print(f"  -> Saved U-Net++ curves to {results_dir / 'unet_training_curves.png'}")


# ============================================================================
# Stage 3: Combined Pipeline
# ============================================================================
# Core logic lives in ``combined/``; symbols re-imported above for public API.

def calculate_wound_area(
    mask: np.ndarray,
    pixels_per_cm: float = 60.0,
) -> float:
    """Calculate wound area in cm² from a binary mask.

    Args:
        mask: Binary mask (H×W), nonzero pixels = wound.
        pixels_per_cm: Scale factor derived from the 3×3 cm reference marker.

    Returns:
        Wound area in cm².
    """
    wound_pixels = int(mask.sum())
    return wound_pixels / (pixels_per_cm ** 2)


def _parse_infection_from_filename(filename: str) -> str:
    """Infer infection status from filename conventions.

    Returns ``"not_infected"``, ``"infected"``, or ``"unknown"``.
    """
    name_lower = filename.lower()
    if "-not-" in name_lower or "not_infected" in name_lower:
        return "not_infected"
    if "infected" in name_lower:
        return "infected"
    return "unknown"


def _is_infected_binary(label: Optional[str]) -> Optional[bool]:
    """Map infection label to binary positive/negative, or None if unknown."""
    if not label:
        return None
    name = str(label).lower()
    if "not" in name or name in {"non_infected", "negative", "0"}:
        return False
    if "infected" in name or name in {"positive", "1"}:
        return True
    return None


def _prediction_outcome(
    metadata_label: Optional[str],
    predicted_label: Optional[str],
) -> Optional[str]:
    """Return TP/TN/FP/FN from metadata-based and predicted infection labels."""
    meta_pos = _is_infected_binary(metadata_label)
    pred_pos = _is_infected_binary(predicted_label)
    if meta_pos is None or pred_pos is None:
        return None
    if meta_pos and pred_pos:
        return "TP"
    if not meta_pos and not pred_pos:
        return "TN"
    if not meta_pos and pred_pos:
        return "FP"
    return "FN"


def _compute_mask_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> Tuple[float, float]:
    """Return Dice and IoU between binary prediction and reference masks."""
    smooth = 1e-6
    p_flat = pred_mask.flatten().astype(float)
    t_flat = gt_mask.flatten().astype(float)
    inter = (p_flat * t_flat).sum()
    union = p_flat.sum() + t_flat.sum()
    dice = float((2 * inter + smooth) / (union + smooth))
    iou = float((inter + smooth) / (union - inter + smooth))
    return dice, iou


def _bootstrap_mean_ci(
    values: List[float],
    *,
    seed: int = 42,
    n_bootstrap: int = 2000,
) -> dict:
    """Return a percentile 95% confidence interval for a sample mean."""
    if not values:
        return {}
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(n_bootstrap, len(array)))
    means = array[indices].mean(axis=1)
    return {
        "lower": round(float(np.percentile(means, 2.5)), 6),
        "upper": round(float(np.percentile(means, 97.5)), 6),
        "n_bootstrap": n_bootstrap,
    }


def _format_infection_display(label: Optional[str]) -> str:
    if not label or label == "unknown":
        return "UNKNOWN"
    if _is_infected_binary(label) is False:
        return "NOT INFECTED"
    if _is_infected_binary(label) is True:
        return "INFECTED"
    return str(label).upper()


def draw_info_panel(
    image: np.ndarray,
    area_cm2: Optional[float],
    area_px: int,
    pixels_per_cm: Optional[float],
    infection_label: Optional[str] = None,
    infection_prob: Optional[float] = None,
    *,
    dice: Optional[float] = None,
    iou: Optional[float] = None,
    metadata_infection: Optional[str] = None,
    prediction_outcome: Optional[str] = None,
) -> np.ndarray:
    """Draw a unified info panel on the top-left corner of the image.

    Three measurement display cases:
      - [MEASURED]   — Area in cm² with scale (green text)
      - [NO SCALE]   — Area in pixels only, cm² unavailable (yellow text)

    Infection status line:
      - INFECTED      — red text
      - NOT INFECTED  — green text
      - UNKNOWN       — grey text
    """
    overlay = image.copy()
    h_img, w_img = overlay.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.55, min(w_img / 900, 1.2))
    thickness = max(1, int(font_scale * 2))
    line_height = int(28 * font_scale)
    pad = 8

    lines: list = []
    colors: list = []

    # Line 1: wound area
    if area_cm2 is not None and pixels_per_cm is not None:
        lines.append(f"Wound Area: {area_cm2:.1f} cm2  [measured]")
        colors.append((100, 255, 100))  # green (BGR)
    else:
        lines.append(f"Wound Area: {area_px:,} px  [no scale ref]")
        colors.append((0, 220, 255))  # yellow (BGR)

    # Line 2: scale info
    if pixels_per_cm is not None:
        lines.append(f"Scale: {pixels_per_cm:.1f} px/cm (marker detected)")
        colors.append((180, 180, 180))  # grey
    else:
        lines.append("No reference marker detected")
        colors.append((0, 180, 255))  # orange-yellow

    # Line 3: infection status
    if infection_label and infection_label != "unknown":
        if "not" in infection_label.lower():
            lines.append("Status: NOT INFECTED")
            colors.append((100, 255, 100))  # green
        else:
            prob_str = f" ({infection_prob:.0%})" if infection_prob is not None and infection_prob >= 0 else ""
            lines.append(f"Status: INFECTED{prob_str}")
            colors.append((80, 80, 255))  # red (BGR)
    elif infection_label == "unknown":
        lines.append("Status: UNKNOWN")
        colors.append((180, 180, 180))

    if dice is not None and iou is not None:
        lines.append(f"Segm: Dice={dice:.3f} | IoU={iou:.3f}")
        colors.append((200, 200, 255))

    if metadata_infection is not None:
        meta_text = _format_infection_display(metadata_infection)
        pred_text = _format_infection_display(infection_label)
        outcome_text = f" ({prediction_outcome})" if prediction_outcome else ""
        lines.append(f"Meta: {meta_text} | Pred: {pred_text}{outcome_text}")
        colors.append((220, 220, 220))

    # Compute panel size
    max_tw = 0
    total_th = 0
    for line_text in lines:
        (tw, th), _ = cv2.getTextSize(line_text, font, font_scale, thickness)
        max_tw = max(max_tw, tw)
        total_th += line_height

    panel_w = max_tw + pad * 3
    panel_h = total_th + pad * 2

    # Semi-transparent black background
    sub = overlay[pad:pad + panel_h, pad:pad + panel_w]
    if sub.size > 0:
        black_bg = np.zeros_like(sub)
        cv2.addWeighted(sub, 0.35, black_bg, 0.65, 0, sub)

    # Draw text lines
    y_cursor = pad + pad + int(line_height * 0.75)
    for text, color in zip(lines, colors):
        cv2.putText(overlay, text, (pad + pad, y_cursor),
                    font, font_scale, color, thickness, cv2.LINE_AA)
        y_cursor += line_height

    return overlay


def evaluate_combined(config: dict, script_dir: Path) -> dict:
    """Run combined YOLO + U-Net++ evaluation on the test set."""
    print("\n" + "=" * 60)
    print("Stage 3: Combined YOLO11m + U-Net++ Evaluation")
    print("=" * 60)

    device = get_device()
    combined_cfg = config.get("combined", {})

    # Load YOLO
    yolo_best = script_dir / "checkpoints" / "yolo" / "best.pt"
    if not yolo_best.exists():
        print("[ERROR] YOLO best.pt not found. Train YOLO first.")
        return {}
    yolo_model = build_yolo_model(str(yolo_best))

    # Load U-Net++
    unet_best = get_unet_best_checkpoint_path(script_dir, config)
    if not unet_best.exists():
        print("[ERROR] U-Net++ best_model.pth not found. Train U-Net++ first.")
        return {}
    unet_model = build_unet_model(config)
    load_unet_checkpoint(unet_model, unet_best, device)
    unet_model.to(device)
    unet_model.eval()

    # Get test images
    project_root = script_dir.parent.parent
    test_ann_path = (project_root / config["ann_test"]).resolve()
    data_root = (project_root / config["data_root"]).resolve()

    with open(test_ann_path, "r", encoding="utf-8") as f:
        test_coco = json.load(f)

    img_lookup = {img["id"]: img for img in test_coco["images"]}
    cat_ids = {c["id"] for c in test_coco["categories"]}
    img_anns: Dict[int, list] = {}
    for ann in test_coco["annotations"]:
        if ann["category_id"] in cat_ids:
            img_anns.setdefault(ann["image_id"], []).append(ann)

    dice_scores, iou_scores = [], []
    wound_areas = []
    combined_dirs = get_combined_dirs(script_dir, config)
    results_combined = combined_dirs["results"]
    results_combined.mkdir(parents=True, exist_ok=True)
    pred_dir = combined_dirs["predictions"]
    pred_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = snapshot_config(
        config,
        results_combined,
        Path(config["_config_path"]) if config.get("_config_path") else None,
    )
    manifest = build_experiment_manifest(
        config,
        script_dir,
        run_mode="evaluate_only",
        checkpoint_paths=[yolo_best, unet_best],
    )
    manifest_path = save_experiment_manifest(manifest, results_combined)
    print(f"  Results dir: {results_combined}")
    print(f"  Config:      {config_snapshot}")
    print(f"  Manifest:    {manifest_path}")

    classifier_path = script_dir / "checkpoints" / "infection" / "infection_classifier.pth"
    num_qual = combined_cfg.get("num_qualitative_samples", 8)
    saved_count = 0

    n_total = 0
    n_missed = 0
    n_marker_detected = 0
    for img_id, img_info in img_lookup.items():
        img_path = str(data_root / img_info["file_name"])
        if not Path(img_path).exists():
            continue
        n_total += 1

        pred = combined_inference(yolo_model, unet_model, img_path, device, config)
        has_pred = not ("error" in pred) and bool(pred.get("masks"))

        orig_h, orig_w = img_info["height"], img_info["width"]
        gt_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for ann in img_anns.get(img_id, []):
            for seg in ann.get("segmentation", []):
                if len(seg) < 6:
                    continue
                poly = np.array(seg, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(gt_mask, [poly], 1)

        if not has_pred:
            n_missed += 1
            dice, iou = 0.0, 0.0
            dice_scores.append(dice)
            iou_scores.append(iou)
            combined_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            area_px = 0
            area_cm2 = None
            ppcm = None
            marker_detected = False
            predicted_infection = _parse_infection_from_filename(img_info["file_name"])
            infection_prob = None
        else:
            combined_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            for m in pred["masks"]:
                if m.shape == (orig_h, orig_w):
                    combined_mask = np.maximum(combined_mask, m)

            dice, iou = _compute_mask_metrics(combined_mask, gt_mask)
            dice_scores.append(dice)
            iou_scores.append(iou)

            ppcm = pred.get("pixels_per_cm")
            area_px = int(combined_mask.sum())
            area_cm2 = calculate_wound_area(combined_mask, ppcm) if ppcm else None
            marker_detected = ppcm is not None

            predicted_infection = _parse_infection_from_filename(img_info["file_name"])
            infection_prob = None
            if classifier_path.exists() and area_px > 0:
                try:
                    image_rgb = cv2.cvtColor(
                        cv2.imread(img_path), cv2.COLOR_BGR2RGB,
                    )
                    inf_result = predict_infection(
                        image_rgb, combined_mask, classifier_path, device,
                    )
                    predicted_infection = inf_result["predicted"]
                    infection_prob = inf_result["infected_prob"]
                except Exception:
                    pass

        metadata_infection = _parse_infection_from_filename(img_info["file_name"])
        outcome = _prediction_outcome(metadata_infection, predicted_infection)
        quality = "measured" if marker_detected else "unavailable"
        if marker_detected:
            n_marker_detected += 1
        wound_areas.append({
            "image": img_info["file_name"],
            "area_px": area_px,
            "area_cm2": area_cm2,
            "pixels_per_cm": ppcm,
            "marker_detected": marker_detected,
            "measurement_quality": quality,
            "dice": round(dice, 6),
            "iou": round(iou, 6),
            "metadata_infection": metadata_infection,
            "infection": predicted_infection,
            "infection_prob": infection_prob,
            "prediction_outcome": outcome,
        })

        if has_pred and saved_count < num_qual:
            img_bgr = cv2.imread(img_path)
            if img_bgr is not None:
                overlay = img_bgr.copy()
                mask_color = np.zeros_like(img_bgr)
                mask_color[:, :, 1] = 255
                overlay[combined_mask > 0] = cv2.addWeighted(
                    overlay, 0.5, mask_color, 0.5, 0
                )[combined_mask > 0]
                for box in pred["boxes"]:
                    cv2.rectangle(
                        overlay,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        (0, 255, 0),
                        2,
                    )

                overlay = draw_info_panel(
                    overlay, area_cm2, area_px, ppcm,
                    predicted_infection, infection_prob,
                    dice=dice,
                    iou=iou,
                    metadata_infection=metadata_infection,
                    prediction_outcome=outcome,
                )

                fname = Path(img_info["file_name"]).stem
                out = pred_dir / f"combined_{fname}.png"
                cv2.imwrite(str(out), overlay)
                with open(out.with_suffix(".json"), "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "image": img_info["file_name"],
                            "confidence": (
                                max(pred.get("scores", [0.0]))
                                if pred.get("scores") else None
                            ),
                            "wound_area_cm2": area_cm2,
                            "wound_area_px": area_px,
                            "dice": round(dice, 6),
                            "iou": round(iou, 6),
                            "metadata_infection": metadata_infection,
                            "infection": predicted_infection,
                            "infection_prob": infection_prob,
                            "prediction_outcome": outcome,
                            "marker_detected": marker_detected,
                        },
                        f,
                        indent=2,
                    )
                saved_count += 1

    n_full = max(1, len(dice_scores))
    n_detected = n_full - n_missed
    cond_n = max(1, n_detected)
    metrics = {
        "mean_dice": sum(dice_scores) / n_full,
        "mean_iou": sum(iou_scores) / n_full,
        "median_dice": float(np.median(dice_scores)) if dice_scores else 0.0,
        "median_iou": float(np.median(iou_scores)) if iou_scores else 0.0,
        "mean_dice_ci95": _bootstrap_mean_ci(
            dice_scores, seed=int(config.get("seed", 42))
        ),
        "mean_iou_ci95": _bootstrap_mean_ci(
            iou_scores, seed=int(config.get("seed", 42)) + 1
        ),
        "mean_dice_conditional": sum(dice_scores) / cond_n,
        "mean_iou_conditional": sum(iou_scores) / cond_n,
        "n_images_total": n_total,
        "n_images_evaluated": n_detected,
        "n_images_missed": n_missed,
        "n_marker_detected": n_marker_detected,
        "marker_detection_rate": n_marker_detected / max(n_total, 1),
        "n_dice_below_0_5": sum(score < 0.5 for score in dice_scores),
        "n_predictions_saved": saved_count,
    }

    coco_metrics = evaluate_combined_coco(
        config, script_dir, yolo_model, unet_model, device,
    )
    if coco_metrics:
        metrics.update(coco_metrics)

    with open(results_combined / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(results_combined / "wound_areas.json", "w", encoding="utf-8") as f:
        json.dump(wound_areas, f, indent=2, ensure_ascii=False)

    print(f"\n  Mean Dice: {metrics['mean_dice']:.4f}")
    print(f"  Mean IoU:  {metrics['mean_iou']:.4f}")
    if coco_metrics:
        print(f"  COCO bbox AP50:  {coco_metrics.get('coco_bbox_AP50', 0):.4f}")
        print(f"  COCO bbox AP75:  {coco_metrics.get('coco_bbox_AP75', 0):.4f}")
        print(f"  COCO segm AP50:  {coco_metrics.get('coco_segm_AP50', 0):.4f}")
        print(f"  COCO segm AP75:  {coco_metrics.get('coco_segm_AP75', 0):.4f}")
        print(f"  COCO combined AP50: {coco_metrics.get('coco_combined_AP50', 0):.4f}")
        print(f"  COCO combined AP75: {coco_metrics.get('coco_combined_AP75', 0):.4f}")
    print(f"  Images evaluated: {metrics['n_images_evaluated']}")
    print(f"  Predictions saved: {saved_count} to {pred_dir}")
    return metrics


# ============================================================================
# Stage 4: Infection Classification
# ============================================================================

class WoundInfectionClassifier(nn.Module):
    """Lightweight infection classifier on wound ROI features.

    Extracts texture/color statistics from the wound ROI and feeds them
    through a small MLP to predict infected vs non-infected.
    """

    def __init__(self, in_features: int = 15, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def extract_wound_features(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Extract texture/color features from wound region for classification.

    Returns a 15-dim feature vector: [mean_r, mean_g, mean_b, std_r, std_g,
    std_b, mean_h, mean_s, mean_v, std_h, std_s, std_v, wound_ratio,
    perimeter_ratio, compactness].
    """
    if mask.sum() == 0:
        return np.zeros(15, dtype=np.float32)

    wound_pixels = image_rgb[mask > 0]
    mean_rgb = wound_pixels.mean(axis=0) / 255.0
    std_rgb = wound_pixels.std(axis=0) / 255.0

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    wound_hsv = hsv[mask > 0]
    mean_hsv = wound_hsv.mean(axis=0) / np.array([180.0, 255.0, 255.0])
    std_hsv = wound_hsv.std(axis=0) / np.array([180.0, 255.0, 255.0])

    wound_ratio = mask.sum() / max(mask.size, 1)

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    area = max(float(mask.sum()), 1.0)
    perimeter_ratio = perimeter / max(np.sqrt(area), 1.0)
    compactness = (4 * np.pi * area) / max(perimeter ** 2, 1.0)

    return np.array([
        *mean_rgb, *std_rgb, *mean_hsv, *std_hsv,
        wound_ratio, perimeter_ratio, compactness,
    ], dtype=np.float32)


def _parse_infection_label(filename: str) -> Optional[int]:
    """Parse infection label from filename. Returns 1=infected, 0=not_infected, None=unknown."""
    name = filename.lower()
    if "not_infected" in name or "-not-" in name:
        return 0
    if "infected" in name:
        return 1
    return None


def _collect_infection_features(
    yolo_model,
    unet_model: "nn.Module",
    device: "torch.device",
    config: dict,
    ann_path: Path,
    img_root: Path,
    split_name: str,
) -> tuple:
    """Extract (features, labels) for one data split.

    Returns (features_list, labels_list, skipped_no_mask, skipped_no_label).
    """
    features_list: list = []
    labels_list: list = []
    skipped_no_label = 0
    skipped_no_mask = 0

    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    total = len(coco["images"])
    for i, img_info in enumerate(coco["images"]):
        label = _parse_infection_label(img_info["file_name"])
        if label is None:
            skipped_no_label += 1
            continue

        img_path = str(img_root / img_info["file_name"])
        if not Path(img_path).exists():
            skipped_no_mask += 1
            continue

        pred = combined_inference(
            yolo_model, unet_model, img_path, device, config, enable_tta=False
        )
        if "error" in pred or not pred["masks"]:
            skipped_no_mask += 1
            continue

        image_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        combined_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        for m in pred["masks"]:
            if m.shape == image_rgb.shape[:2]:
                combined_mask = np.maximum(combined_mask, m)

        feats = extract_wound_features(image_rgb, combined_mask)
        features_list.append(feats)
        labels_list.append(label)

        if (i + 1) % 20 == 0:
            print(
                f"  [{split_name}] {i + 1}/{total} images processed "
                f"({len(features_list)} usable)"
            )

    return features_list, labels_list, skipped_no_mask, skipped_no_label


def _binary_metrics_from_probabilities(
    probabilities: "torch.Tensor",
    y: "torch.Tensor",
    threshold: float = 0.5,
) -> dict:
    """Return binary metrics from probabilities at a locked threshold."""
    with torch.no_grad():
        preds = (probabilities >= threshold).float()
    tp = float(((preds == 1) & (y == 1)).sum())
    fp = float(((preds == 1) & (y == 0)).sum())
    fn = float(((preds == 0) & (y == 1)).sum())
    tn = float(((preds == 0) & (y == 0)).sum())
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    return {
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "specificity": round(tn / max(tn + fp, 1), 6),
        "f1_score": round(f1, 6),
        "threshold": round(float(threshold), 6),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def _compute_binary_metrics(
    model: "nn.Module",
    X_norm: "torch.Tensor",
    y: "torch.Tensor",
    threshold: float = 0.5,
) -> dict:
    """Return binary metrics for a classifier at a locked threshold."""
    with torch.no_grad():
        probabilities = torch.sigmoid(model(X_norm))
    return _binary_metrics_from_probabilities(probabilities, y, threshold)


def _select_binary_threshold(
    probabilities: "torch.Tensor",
    y: "torch.Tensor",
) -> tuple[float, dict]:
    """Select threshold on validation F1 only; never inspect test labels."""
    candidates = np.linspace(0.10, 0.90, 81)
    scored = []
    for threshold in candidates:
        metrics = _binary_metrics_from_probabilities(
            probabilities, y, float(threshold)
        )
        balanced_accuracy = 0.5 * (
            metrics["recall"] + metrics["specificity"]
        )
        scored.append((metrics["f1_score"], balanced_accuracy, threshold, metrics))
    _, _, threshold, metrics = max(
        scored,
        key=lambda item: (item[0], item[1], -abs(float(item[2]) - 0.5)),
    )
    return float(threshold), metrics


def _bootstrap_binary_confidence_intervals(
    probabilities: "torch.Tensor",
    y: "torch.Tensor",
    threshold: float,
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """Compute percentile 95% CIs on the held-out test samples."""
    probs_np = probabilities.detach().cpu().numpy().reshape(-1)
    labels_np = y.detach().cpu().numpy().reshape(-1)
    if len(labels_np) < 2:
        return {}
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in ("accuracy", "precision", "recall", "specificity", "f1_score")}
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(labels_np), size=len(labels_np))
        sampled_probs = torch.tensor(probs_np[indices], dtype=torch.float32).reshape(-1, 1)
        sampled_labels = torch.tensor(labels_np[indices], dtype=torch.float32).reshape(-1, 1)
        metrics = _binary_metrics_from_probabilities(
            sampled_probs, sampled_labels, threshold
        )
        for key in samples:
            samples[key].append(metrics[key])
    return {
        key: {
            "lower": round(float(np.percentile(values, 2.5)), 6),
            "upper": round(float(np.percentile(values, 97.5)), 6),
        }
        for key, values in samples.items()
    }


def train_infection_classifier(
    config: dict,
    script_dir: Path,
) -> dict:
    """Train on train, select epoch/threshold on val, evaluate test once.

    Infection labels remain filename-derived metadata proxies and must not be
    interpreted as clinical diagnoses.
    """
    print("\n" + "=" * 60)
    print("Stage 4: Training Infection Classifier")
    print("  Train split: ann_train only")
    print("  Val split:   ann_val (epoch + threshold selection)")
    print("  Test split:  ann_test (final held-out evaluation)")
    print("=" * 60)

    device = get_device()
    project_root = script_dir.parent.parent
    data_root = (project_root / config["data_root"]).resolve()
    data_root_train = (
        project_root / config.get("data_root_train", config["data_root"])
    ).resolve()
    infection_cfg = config.get("infection", {})
    seeds = [int(seed) for seed in infection_cfg.get("seeds", [42, 43, 44])]
    max_epochs = int(infection_cfg.get("epochs", 200))
    patience = int(infection_cfg.get("early_stop_patience", 25))
    learning_rate = float(infection_cfg.get("lr", 0.001))
    weight_decay = float(infection_cfg.get("weight_decay", 1e-4))
    n_bootstrap = int(infection_cfg.get("bootstrap_samples", 2000))

    yolo_best = script_dir / "checkpoints" / "yolo" / "best.pt"
    unet_best = get_unet_best_checkpoint_path(script_dir, config)
    if not yolo_best.exists():
        print(f"[ERROR] YOLO checkpoint not found: {yolo_best}")
        print("        Run: python train_model.py --stage yolo")
        return {}
    if not unet_best.exists():
        print(f"[ERROR] U-Net++ checkpoint not found: {unet_best}")
        print("        Run: python train_model.py --stage unet")
        return {}

    if "ann_val" not in config or "ann_test" not in config:
        print("[ERROR] ann_val and ann_test are required for independent evaluation.")
        return {}

    yolo_model = build_yolo_model(str(yolo_best))
    unet_model = build_unet_model(config)
    load_unet_checkpoint(unet_model, unet_best, device)
    unet_model.to(device)
    unet_model.eval()

    split_specs = {
        "train": (
            (project_root / config["ann_train"]).resolve(),
            data_root_train,
        ),
        "val": (
            (project_root / config["ann_val"]).resolve(),
            data_root,
        ),
        "test": (
            (project_root / config["ann_test"]).resolve(),
            data_root,
        ),
    }
    collected = {}
    for index, (split, (ann_path, root)) in enumerate(split_specs.items(), start=1):
        print(f"\n[{index}/3] Extracting features from {split.upper()} split ...")
        features, labels, skipped_mask, skipped_label = _collect_infection_features(
            yolo_model,
            unet_model,
            device,
            config,
            ann_path=ann_path,
            img_root=root,
            split_name=split,
        )
        collected[split] = {
            "features": features,
            "labels": labels,
            "skipped_no_mask": skipped_mask,
            "skipped_no_label": skipped_label,
        }
        print(
            f"  {split.title()}: {len(features)} usable "
            f"(skipped no-mask={skipped_mask}, no-label={skipped_label})"
        )

    if len(collected["train"]["features"]) < 10:
        print("[ERROR] Too few training samples.")
        return {}
    if len(collected["val"]["features"]) < 5:
        print("[ERROR] Too few validation samples for threshold selection.")
        return {}

    tensors = {}
    for split in ("train", "val", "test"):
        tensors[split] = {
            "X": torch.tensor(
                np.array(collected[split]["features"]), dtype=torch.float32
            ),
            "y": torch.tensor(
                collected[split]["labels"], dtype=torch.float32
            ).unsqueeze(1),
        }
    X_train = tensors["train"]["X"]
    y_train = tensors["train"]["y"]
    n_infected_train = int(y_train.sum())
    n_not_train = len(y_train) - n_infected_train
    print(f"  Train labels: infected={n_infected_train}, non-infected={n_not_train}")

    feat_mean = X_train.mean(dim=0)
    feat_std = X_train.std(dim=0).clamp(min=1e-6)
    for split in ("train", "val", "test"):
        tensors[split]["X_norm"] = (
            (tensors[split]["X"] - feat_mean) / feat_std
        ).to(device)
        tensors[split]["y_dev"] = tensors[split]["y"].to(device)

    pos_count = float(y_train.sum())
    neg_count = len(y_train) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    seed_runs = []
    best_run = None
    print(f"\nTraining infection classifier with seeds={seeds} ...")
    for seed in seeds:
        set_seed(seed)
        model = WoundInfectionClassifier(in_features=X_train.shape[1]).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        history = {
            "epochs": [],
            "loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_f1_at_0_5": [],
        }
        best_state = None
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_epoch = 0

        for epoch in range(1, max_epochs + 1):
            model.train()
            optimizer.zero_grad()
            logits = model(tensors["train"]["X_norm"])
            loss = criterion(logits, tensors["train"]["y_dev"])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                train_probs = torch.sigmoid(
                    model(tensors["train"]["X_norm"])
                )
                val_logits = model(tensors["val"]["X_norm"])
                val_loss = criterion(val_logits, tensors["val"]["y_dev"]).item()
                val_probs = torch.sigmoid(val_logits)
            train_metrics_epoch = _binary_metrics_from_probabilities(
                train_probs, tensors["train"]["y_dev"], 0.5
            )
            val_metrics_epoch = _binary_metrics_from_probabilities(
                val_probs, tensors["val"]["y_dev"], 0.5
            )
            history["epochs"].append(epoch)
            history["loss"].append(round(loss.item(), 6))
            history["val_loss"].append(round(val_loss, 6))
            history["train_accuracy"].append(train_metrics_epoch["accuracy"])
            history["val_f1_at_0_5"].append(val_metrics_epoch["f1_score"])

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

        if best_state is None:
            raise RuntimeError(f"No validation checkpoint selected for seed {seed}")
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            train_probs = torch.sigmoid(model(tensors["train"]["X_norm"]))
            val_probs = torch.sigmoid(model(tensors["val"]["X_norm"]))
        threshold, val_metrics = _select_binary_threshold(
            val_probs, tensors["val"]["y_dev"]
        )
        train_metrics = _binary_metrics_from_probabilities(
            train_probs, tensors["train"]["y_dev"], threshold
        )
        run = {
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 6),
            "threshold": round(threshold, 6),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "history": history,
            "model_state_dict": best_state,
        }
        seed_runs.append(run)
        print(
            f"  seed={seed}: epoch={best_epoch}, threshold={threshold:.2f}, "
            f"val_F1={val_metrics['f1_score']:.4f}"
        )
        if best_run is None or (
            val_metrics["f1_score"],
            -best_val_loss,
            -seed,
        ) > (
            best_run["val_metrics"]["f1_score"],
            -best_run["best_val_loss"],
            -best_run["seed"],
        ):
            best_run = run

    if best_run is None:
        raise RuntimeError("No infection classifier run completed")

    model = WoundInfectionClassifier(in_features=X_train.shape[1]).to(device)
    model.load_state_dict(best_run["model_state_dict"])
    model.eval()
    threshold = float(best_run["threshold"])
    with torch.no_grad():
        test_probs = torch.sigmoid(model(tensors["test"]["X_norm"]))
    test_metrics = _binary_metrics_from_probabilities(
        test_probs, tensors["test"]["y_dev"], threshold
    )
    test_ci = _bootstrap_binary_confidence_intervals(
        test_probs,
        tensors["test"]["y_dev"],
        threshold,
        n_bootstrap=n_bootstrap,
        seed=int(config.get("seed", 42)),
    )

    ckpt_dir = script_dir / "checkpoints" / "infection"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "feat_mean": feat_mean.cpu(),
        "feat_std": feat_std.cpu(),
        "in_features": X_train.shape[1],
        "threshold": threshold,
        "seed": best_run["seed"],
        "best_epoch": best_run["best_epoch"],
        "label_note": "Filename metadata proxy; not a clinical diagnosis.",
    }, ckpt_dir / "infection_classifier.pth")

    train_metrics = best_run["train_metrics"]
    val_metrics = best_run["val_metrics"]
    n_infected_val = int(tensors["val"]["y"].sum())
    n_infected_test = int(tensors["test"]["y"].sum())
    summary = {
        "evaluation_note": (
            "train_* metrics are in-sample. val_* selected epoch and threshold. "
            "test_* metrics are held out and evaluated after locking the model. "
            "Labels are derived from filename metadata (-not-), not confirmed clinical diagnosis."
        ),
        "selection_protocol": (
            "Epoch selected by validation BCE loss; decision threshold selected by "
            "validation F1; canonical seed selected by validation F1 only."
        ),
        "canonical_seed": best_run["seed"],
        "best_epoch": best_run["best_epoch"],
        "decision_threshold": threshold,
        "train_n_samples": len(X_train),
        "train_n_infected": n_infected_train,
        "train_n_non_infected": n_not_train,
        **{f"train_{key}": value for key, value in train_metrics.items()},
        "val_n_samples": len(tensors["val"]["y"]),
        "val_n_infected": n_infected_val,
        "val_n_non_infected": len(tensors["val"]["y"]) - n_infected_val,
        **{f"val_{key}": value for key, value in val_metrics.items()},
        "test_n_samples": len(tensors["test"]["y"]),
        "test_n_infected": n_infected_test,
        "test_n_non_infected": len(tensors["test"]["y"]) - n_infected_test,
        **{f"test_{key}": value for key, value in test_metrics.items()},
        "test_confidence_intervals_95": test_ci,
        "seed_runs": [
            {
                key: value
                for key, value in run.items()
                if key not in {"history", "model_state_dict"}
            }
            for run in seed_runs
        ],
    }

    results_dir = script_dir / "results" / "infection"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(best_run["history"], f, indent=2)
    with open(results_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n── Final held-out test metrics ──")
    for key in ("accuracy", "precision", "recall", "specificity", "f1_score"):
        print(f"  test_{key}: {test_metrics[key]:.4f}")
    print(f"  Saved: {results_dir / 'training_history.json'}")
    print(f"  Saved: {results_dir / 'metrics_summary.json'}")
    return summary


def predict_infection(
    image_rgb: np.ndarray,
    wound_mask: np.ndarray,
    classifier_path: Path,
    device: torch.device,
) -> Dict[str, float]:
    """Predict infection status from wound ROI features.

    The classifier is always run on **CPU** (tiny MLP). ``device`` is kept for API
    compatibility with callers; YOLO/U-Net can stay on CUDA without mixing tensors here.
    """
    if not classifier_path.exists():
        return {"infected_prob": -1.0, "predicted": "unknown"}

    ckpt = torch.load(classifier_path, map_location="cpu", weights_only=False)
    feats = extract_wound_features(image_rgb, wound_mask)
    feats_t = torch.as_tensor(feats, dtype=torch.float32).unsqueeze(0)
    feat_mean = torch.as_tensor(ckpt["feat_mean"], dtype=torch.float32)
    feat_std = torch.as_tensor(ckpt["feat_std"], dtype=torch.float32)
    feat_std = torch.clamp(feat_std, min=1e-8)
    feats_norm = (feats_t - feat_mean) / feat_std

    model = WoundInfectionClassifier(in_features=ckpt["in_features"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.cpu()

    with torch.no_grad():
        logit = model(feats_norm)
        prob = torch.sigmoid(logit).item()
    threshold = float(ckpt.get("threshold", 0.5))

    return {
        "infected_prob": prob,
        "threshold": threshold,
        "predicted": "infected" if prob >= threshold else "non_infected",
    }


# ============================================================================
# Reporting and Visualization
# ============================================================================

def save_training_curves(train_losses: list, val_losses: list, path: Path) -> None:
    """Generic loss curve plot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(train_losses) + 1), train_losses, label="Train")
    ax.plot(range(1, len(val_losses) + 1), val_losses, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def generate_report(
    yolo_results: dict,
    unet_results: dict,
    combined_results: dict,
    config: dict,
    reports_dir: Path,
    infection_results: Optional[dict] = None,
) -> Path:
    """Generate a markdown training report."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "training_report.md"

    lines = [
        "# YOLO11m + U-Net++ Training Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n---\n",
        "## Configuration\n",
    ]

    for section in ["yolo", "unet", "combined"]:
        cfg_section = config.get(section, {})
        if cfg_section:
            lines.append(f"### {section.upper()}\n")
            lines.append("| Parameter | Value |")
            lines.append("|-----------|-------|")
            for k, v in cfg_section.items():
                lines.append(f"| {k} | {v} |")
            lines.append("")

    lines.append("\n---\n")
    lines.append("## YOLO11m-seg Results\n")
    if yolo_results:
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for k, v in yolo_results.items():
            if isinstance(v, float):
                lines.append(f"| {k} | {v:.4f} |")
            elif isinstance(v, (int, bool)):
                lines.append(f"| {k} | {v} |")
    else:
        lines.append("*Not available — YOLO was not trained or evaluated.*\n")

    lines.append("\n---\n")
    lines.append("## U-Net++ Results\n")
    if unet_results:
        test_m = unet_results.get("test_metrics", {})
        lines.append(f"- **Best Dice (val):** {unet_results.get('best_dice', 'N/A'):.4f} "
                      f"at epoch {unet_results.get('best_epoch', 'N/A')}")
        lines.append(f"- **Training time:** {unet_results.get('training_time_s', 0):.0f}s")
        if test_m:
            lines.append("\n### Test Metrics\n")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for k, v in test_m.items():
                if isinstance(v, float):
                    lines.append(f"| {k} | {v:.4f} |")
    else:
        lines.append("*Not available.*\n")

    lines.append("\n---\n")
    lines.append("## Combined Pipeline Results\n")
    if combined_results:
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for k, v in combined_results.items():
            if isinstance(v, float):
                lines.append(f"| {k} | {v:.4f} |")
            elif isinstance(v, int):
                lines.append(f"| {k} | {v} |")
    else:
        lines.append("*Not available.*\n")

    lines.append("\n---\n")
    lines.append("## Infection Classification Results\n")
    if infection_results:
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for k, v in infection_results.items():
            if isinstance(v, float):
                lines.append(f"| {k} | {v:.4f} |")
            elif isinstance(v, int):
                lines.append(f"| {k} | {v} |")
    else:
        lines.append("*Not available.*\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  -> Report saved to {report_path}")
    return report_path


def _find_yolo_results_csv(script_dir: Path) -> Optional[Path]:
    """Return the first existing YOLO training results.csv path."""
    candidates = [
        script_dir / "checkpoints" / "yolo" / "train" / "results.csv",
        script_dir / "results" / "yolo" / "results.csv",
        script_dir / "checkpoints" / "yolo" / "results.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _load_json_if_exists(path: Path) -> dict:
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _display_figure_inline(path: Path, fig=None) -> None:
    """Show a saved figure inline in Jupyter; fall back to ``plt.show()``."""
    try:
        from IPython.display import Image, display
        from IPython import get_ipython
        if get_ipython() is not None:
            display(Image(filename=str(path)))
            if fig is not None:
                plt.close(fig)
            return
    except Exception:
        pass
    if fig is not None:
        plt.show()
        plt.close(fig)
    else:
        try:
            plt.imshow(plt.imread(str(path)))
            plt.axis("off")
            plt.show()
        except Exception:
            print(f"  (figure saved to {path} — open file to view)")


def display_training_curves(script_dir: Path, config: dict) -> Path:
    """Render a 2×3 dashboard of YOLO, U-Net++, and infection training curves.

    Saves ``results/figures/training_curves_dashboard.png`` and calls ``plt.show()``
    for inline notebook display.
    """
    import pandas as pd

    figures_dir = script_dir / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "training_curves_dashboard.png"

    unet_dirs = get_unet_dirs(script_dir, config)
    unet_history = _load_json_if_exists(unet_dirs["results"] / "training_history.json")
    unet_metrics = _load_json_if_exists(unet_dirs["results"] / "metrics_summary.json")
    inf_history = _load_json_if_exists(script_dir / "results" / "infection" / "training_history.json")
    inf_metrics = _load_json_if_exists(script_dir / "results" / "infection" / "metrics_summary.json")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # ── YOLO mAP curves ───────────────────────────────────────────────────────
    yolo_csv = _find_yolo_results_csv(script_dir)
    if yolo_csv is not None:
        df = pd.read_csv(yolo_csv)
        df.columns = [c.strip() for c in df.columns]
        epochs = df["epoch"] if "epoch" in df.columns else range(1, len(df) + 1)
        map_cols = {
            "bbox mAP50": "metrics/mAP50(B)",
            "bbox mAP50-95": "metrics/mAP50-95(B)",
            "segm mAP50": "metrics/mAP50(M)",
            "segm mAP50-95": "metrics/mAP50-95(M)",
        }
        for label, col in map_cols.items():
            if col in df.columns:
                axes[0].plot(epochs, df[col], label=label)
        axes[0].set_title("YOLO Metrics")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("mAP")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        loss_cols = {
            "box": "train/box_loss",
            "seg": "train/seg_loss",
            "cls": "train/cls_loss",
            "dfl": "train/dfl_loss",
        }
        for label, col in loss_cols.items():
            if col in df.columns:
                axes[1].plot(epochs, df[col], label=label)
        axes[1].set_title("YOLO Losses (train)")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
    else:
        for ax_idx in (0, 1):
            axes[ax_idx].text(
                0.5, 0.5, "YOLO results.csv not found",
                ha="center", va="center", transform=axes[ax_idx].transAxes,
            )
            axes[ax_idx].set_title("YOLO (missing)")

    # ── U-Net++ losses ────────────────────────────────────────────────────────
    if unet_history.get("train_losses"):
        epochs_u = range(1, len(unet_history["train_losses"]) + 1)
        axes[2].plot(epochs_u, unet_history["train_losses"], label="Train")
        if unet_history.get("val_losses"):
            axes[2].plot(epochs_u, unet_history["val_losses"], label="Val")
        axes[2].set_title("U-Net++ Losses")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("BCE + Dice Loss")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(
            0.5, 0.5, "U-Net++ training_history.json not found",
            ha="center", va="center", transform=axes[2].transAxes,
        )
        axes[2].set_title("U-Net++ Losses (missing)")

    # ── U-Net++ Dice / IoU ────────────────────────────────────────────────────
    if unet_history.get("dice_per_epoch"):
        epochs_u = range(1, len(unet_history["dice_per_epoch"]) + 1)
        axes[3].plot(epochs_u, unet_history["dice_per_epoch"], label="Val Dice", color="green")
        if unet_history.get("iou_per_epoch"):
            axes[3].plot(epochs_u, unet_history["iou_per_epoch"], label="Val IoU", color="orange")
        best_epoch = int(
            unet_metrics.get("best_epoch")
            or unet_history.get("best_epoch", 0)
        )
        if best_epoch > 0:
            axes[3].axvline(
                best_epoch, color="red", linestyle="--", linewidth=1,
                label=f"best epoch {best_epoch}",
            )
        axes[3].set_title("U-Net++ Dice / IoU (val)")
        axes[3].set_xlabel("Epoch")
        axes[3].set_ylabel("Score")
        axes[3].legend(fontsize=8)
        axes[3].grid(True, alpha=0.3)
    else:
        axes[3].text(
            0.5, 0.5, "U-Net++ dice history not found",
            ha="center", va="center", transform=axes[3].transAxes,
        )
        axes[3].set_title("U-Net++ Dice/IoU (missing)")

    # ── Infection loss ────────────────────────────────────────────────────────
    if inf_history.get("loss"):
        axes[4].plot(inf_history["epochs"], inf_history["loss"], color="steelblue")
        axes[4].set_title("Infection Classifier Loss (train)")
        axes[4].set_xlabel("Epoch")
        axes[4].set_ylabel("BCE Loss")
        axes[4].grid(True, alpha=0.3)
    else:
        axes[4].text(
            0.5, 0.5, "Infection training_history.json not found\n(re-run §4.4)",
            ha="center", va="center", transform=axes[4].transAxes,
        )
        axes[4].set_title("Infection Loss (missing)")

    # ── Infection accuracy ────────────────────────────────────────────────────
    if inf_history.get("train_accuracy"):
        axes[5].plot(
            inf_history["epochs"], inf_history["train_accuracy"],
            label="Train acc (in-sample)", color="steelblue",
        )
        test_acc = inf_metrics.get("test_accuracy")
        if test_acc is not None:
            axes[5].axhline(
                float(test_acc), color="crimson", linestyle=":",
                linewidth=1.5,
                label=f"test acc {float(test_acc):.3f} (held-out ref.)",
            )
        axes[5].set_title("Infection Accuracy")
        axes[5].set_xlabel("Epoch")
        axes[5].set_ylabel("Accuracy")
        axes[5].set_ylim(0.0, 1.0)
        axes[5].legend(fontsize=8)
        axes[5].grid(True, alpha=0.3)
        axes[5].text(
            0.02, 0.02,
            "Test line = final held-out reference only (not for epoch selection)",
            transform=axes[5].transAxes, fontsize=7, color="dimgray",
        )
    else:
        axes[5].text(
            0.5, 0.5, "Infection accuracy history not found\n(re-run §4.4)",
            ha="center", va="center", transform=axes[5].transAxes,
        )
        axes[5].set_title("Infection Accuracy (missing)")

    fig.suptitle("Training Curves Dashboard — YOLO + U-Net++ + Infection", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  -> Saved training dashboard: {out_path}")
    _display_figure_inline(out_path, fig)
    return out_path


def _select_confusion_cases(records: List[dict]) -> Dict[str, dict]:
    """Pick the first alphabetically sorted valid case per TP/TN/FP/FN."""
    sorted_records = sorted(records, key=lambda r: str(r.get("image", "")))
    selected: Dict[str, dict] = {}
    for outcome in ("TP", "TN", "FP", "FN"):
        for rec in sorted_records:
            if rec.get("prediction_outcome") == outcome and rec.get("image"):
                selected[outcome] = rec
                break
    return selected


def _build_gt_mask_from_coco(
    test_coco: dict,
    img_anns: Dict[int, list],
    file_name: str,
) -> np.ndarray:
    """Build a binary reference mask for one test image from COCO polygons."""
    img_info = next(
        (img for img in test_coco["images"] if img["file_name"] == file_name),
        None,
    )
    if img_info is None:
        return np.zeros((1, 1), dtype=np.uint8)
    gt_mask = np.zeros((img_info["height"], img_info["width"]), dtype=np.uint8)
    for ann in img_anns.get(img_info["id"], []):
        for seg in ann.get("segmentation", []):
            if len(seg) < 6:
                continue
            poly = np.array(seg, dtype=np.float32).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(gt_mask, [poly], 1)
    return gt_mask


def display_experiment_gallery(
    script_dir: Path,
    config: dict,
    n_total: int = 4,
    *,
    regenerate_errors: bool = True,
) -> Path:
    """Display a reproducible 2×2 gallery (TP/TN/FP/FN) with metrics table.

    Uses saved combined PNGs for TP/TN when available; regenerates FP/FN live
    against the latest checkpoints for error-case verification.
    """
    import pandas as pd

    if n_total != 4:
        raise ValueError("display_experiment_gallery supports exactly 4 images (2×2).")

    combined_dirs = get_combined_dirs(script_dir, config)
    wound_areas_path = combined_dirs["results"] / "wound_areas.json"
    if not wound_areas_path.is_file():
        print("[INFO] wound_areas.json missing — running evaluate_combined() first ...")
        evaluate_combined(config, script_dir)

    with open(wound_areas_path, "r", encoding="utf-8") as f:
        wound_areas = json.load(f)

    selected = _select_confusion_cases(wound_areas)
    missing = [o for o in ("TP", "TN", "FP", "FN") if o not in selected]
    if missing:
        print(f"[WARNING] No case found for: {', '.join(missing)}")

    project_root = script_dir.parent.parent
    test_ann_path = (project_root / config["ann_test"]).resolve()
    data_root = (project_root / config["data_root"]).resolve()
    with open(test_ann_path, "r", encoding="utf-8") as f:
        test_coco = json.load(f)
    cat_ids = {c["id"] for c in test_coco["categories"]}
    img_anns: Dict[int, list] = {}
    for ann in test_coco["annotations"]:
        if ann["category_id"] in cat_ids:
            img_anns.setdefault(ann["image_id"], []).append(ann)

    device = get_device()
    yolo_best = script_dir / "checkpoints" / "yolo" / "best.pt"
    unet_best = get_unet_best_checkpoint_path(script_dir, config)
    yolo_model = build_yolo_model(str(yolo_best))
    unet_model = build_unet_model(config)
    load_unet_checkpoint(unet_model, unet_best, device)
    unet_model.to(device)
    unet_model.eval()

    pred_dir = combined_dirs["predictions"]
    figures_dir = script_dir / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "experiment_gallery_4panel.png"

    gallery_rows = []
    panels: Dict[str, np.ndarray] = {}
    layout = [("TP", 0, 0), ("TN", 0, 1), ("FP", 1, 0), ("FN", 1, 1)]

    for outcome, _, _ in layout:
        rec = selected.get(outcome)
        if not rec:
            continue
        file_name = rec["image"]
        stem = Path(file_name).stem
        saved_png = pred_dir / f"combined_{stem}.png"
        use_saved = saved_png.is_file() and outcome in ("TP", "TN")

        if regenerate_errors and outcome in ("FP", "FN"):
            use_saved = False

        if use_saved:
            overlay_bgr = cv2.imread(str(saved_png))
            if overlay_bgr is None:
                use_saved = False
            else:
                panels[outcome] = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
                gallery_rows.append({
                    "Outcome": outcome,
                    "Image": file_name,
                    "Dice": rec.get("dice"),
                    "IoU": rec.get("iou"),
                    "Metadata": rec.get("metadata_infection"),
                    "Prediction": rec.get("infection"),
                    "Area cm²": rec.get("area_cm2"),
                    "Source": "saved PNG",
                })

        if outcome not in panels:
            img_path = str(data_root / file_name)
            gt_mask = _build_gt_mask_from_coco(test_coco, img_anns, file_name)
            overlay_bgr, info = predict_single_image(
                yolo_model, unet_model, img_path, device, config, gt_mask=gt_mask,
            )
            panels[outcome] = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
            gallery_rows.append({
                "Outcome": outcome,
                "Image": file_name,
                "Dice": info.get("dice", rec.get("dice")),
                "IoU": info.get("iou", rec.get("iou")),
                "Metadata": info.get("metadata_infection", rec.get("metadata_infection")),
                "Prediction": info.get("infection", rec.get("infection")),
                "Area cm²": info.get("wound_area_cm2", rec.get("area_cm2")),
                "Source": "live predict_single_image",
            })

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for outcome, row, col in layout:
        ax = axes[row, col]
        if outcome in panels:
            ax.imshow(panels[outcome])
            rec = selected.get(outcome, {})
            subtitle = (
                f"{outcome}: {rec.get('image', '')}\n"
                f"Dice={rec.get('dice', 'N/A')} | IoU={rec.get('iou', 'N/A')}"
            )
            ax.set_title(subtitle, fontsize=9)
        else:
            ax.text(0.5, 0.5, f"{outcome}\n(no case found)", ha="center", va="center")
            ax.set_title(outcome)
        ax.axis("off")

    fig.suptitle(
        "Experiment Gallery — TP / TN / FP / FN (metadata-derived infection labels)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  -> Saved experiment gallery: {out_path}")
    _display_figure_inline(out_path, fig)

    if gallery_rows:
        gallery_df = pd.DataFrame(gallery_rows)
        print("\n── Selected gallery cases (4-panel) ──")
        display_cols = [
            "Outcome", "Image", "Dice", "IoU",
            "Metadata", "Prediction", "Area cm²", "Source",
        ]
        print(gallery_df[display_cols].to_string(index=False))

        inf_summary_path = script_dir / "results" / "infection" / "metrics_summary.json"
        inf_metrics = _load_json_if_exists(inf_summary_path)
        if inf_metrics:
            print("\n── Full test-set infection metrics (held-out) ──")
            test_rows = {
                "Metric": [
                    "test_accuracy", "test_precision", "test_recall", "test_f1_score",
                    "test_tp", "test_fp", "test_fn", "test_tn",
                ],
                "Value": [
                    inf_metrics.get("test_accuracy"),
                    inf_metrics.get("test_precision"),
                    inf_metrics.get("test_recall"),
                    inf_metrics.get("test_f1_score"),
                    inf_metrics.get("test_tp"),
                    inf_metrics.get("test_fp"),
                    inf_metrics.get("test_fn"),
                    inf_metrics.get("test_tn"),
                ],
            }
            print(pd.DataFrame(test_rows).to_string(index=False))

    return out_path


def display_results_curves(results_dir: Path, max_show: int = 6) -> None:
    """Display saved curve images inline (for notebooks)."""
    from IPython.display import display, Image as IPImage
    shown = 0
    for subdir in ["yolo", "unet", "combined"]:
        d = results_dir / subdir
        if not d.exists():
            continue
        for img_file in sorted(d.glob("*.png")):
            if "prediction" in img_file.name.lower():
                continue
            if shown >= max_show:
                return
            print(f"\n--- {img_file.name} ---")
            try:
                display(IPImage(filename=str(img_file)))
            except Exception:
                plt.figure(figsize=(10, 6))
                plt.imshow(plt.imread(str(img_file)))
                plt.axis("off")
                plt.title(img_file.name)
                plt.show()
            shown += 1


def display_results_predictions(results_dir: Path, n_show: int = 8) -> None:
    """Display saved prediction images inline (for notebooks)."""
    def _format_caption(img_path: Path) -> str:
        meta_path = img_path.with_suffix(".json")
        meta = {}
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}

        confidence = meta.get("confidence")
        area_cm2 = meta.get("wound_area_cm2")
        infection = meta.get("infection")

        confidence_text = (
            f"{float(confidence):.2f}" if confidence is not None else "N/A"
        )
        area_text = f"{float(area_cm2):.1f} cm2" if area_cm2 is not None else "N/A"
        infection_text = str(infection).replace("_", " ") if infection else "N/A"
        return (
            f"Confidence: {confidence_text} | "
            f"Wound area: {area_text} | "
            f"Infection: {infection_text}"
        )

    pred_dirs = [results_dir / "yolo" / "predictions"]
    combined_root = results_dir / "combined"
    if (combined_root / "predictions").exists():
        pred_dirs.append(combined_root / "predictions")
    elif combined_root.exists():
        pred_dirs.extend(
            sorted(
                path for path in combined_root.glob("*/predictions")
                if path.is_dir()
            )
        )

    shown = 0
    for pred_dir in pred_dirs:
        if shown >= n_show:
            break
        if not pred_dir.exists():
            continue
        pngs = sorted(pred_dir.glob("*.png"))
        if not pngs:
            continue
        n = min(n_show - shown, len(pngs))
        if n <= 0:
            continue
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        if rows * cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for i in range(n):
            img = cv2.imread(str(pngs[i]))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i].imshow(img)
            else:
                axes[i].text(
                    0.5, 0.5,
                    f"Could not read image:\n{pngs[i].name}",
                    ha="center", va="center",
                )
            axes[i].set_title(_format_caption(pngs[i]), fontsize=10)
            axes[i].axis("off")
        for i in range(n, len(axes)):
            axes[i].axis("off")
        fig.suptitle(
            f"{pred_dir.parent.name.upper()} prediction examples",
            fontsize=13,
            fontweight="bold",
        )
        fig.tight_layout(rect=(0, 0.0, 1, 0.94))
        plt.show()
        shown += n

    if shown == 0:
        searched = "\n".join(f"- {path}" for path in pred_dirs)
        print("No prediction PNG files were found. Searched:\n" + searched)


def predict_single_image(
    yolo_model,
    unet_model: nn.Module,
    image_path: str,
    device: torch.device,
    config: dict,
    *,
    gt_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Predict on a single image and return the annotated image + info dict.
    """
    pred = combined_inference(yolo_model, unet_model, image_path, device, config)
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return np.zeros((100, 100, 3), dtype=np.uint8), {"error": "Image not found"}

    info = {
        "file_name": Path(image_path).name,
        "n_detections": len(pred.get("boxes", [])),
        "scores": pred.get("scores", []),
    }

    info["infection_filename"] = _parse_infection_from_filename(Path(image_path).name)
    info["metadata_infection"] = info["infection_filename"]
    info["confidence"] = max(pred.get("scores", [0.0]))

    classifier_path = Path(image_path).parent.parent / "checkpoints" / "infection" / "infection_classifier.pth"
    if not classifier_path.exists():
        classifier_path = SCRIPT_DIR / "checkpoints" / "infection" / "infection_classifier.pth"

    overlay = img_bgr.copy()
    combined_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    for m in pred.get("masks", []):
        if m.shape == img_bgr.shape[:2]:
            combined_mask = np.maximum(combined_mask, m)

    mask_color = np.zeros_like(img_bgr)
    mask_color[:, :, 1] = 255
    overlay[combined_mask > 0] = cv2.addWeighted(
        overlay, 0.5, mask_color, 0.5, 0
    )[combined_mask > 0]

    for box, score in zip(pred.get("boxes", []), pred.get("scores", [])):
        cv2.rectangle(
            overlay,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            (0, 255, 0),
            2,
        )
        text_org = (int(box[0]), max(0, int(box[1]) - 5))
        cv2.putText(
            overlay, f"{score:.2f}", text_org,
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
        )

    ppcm = pred.get("pixels_per_cm")
    area_px = int(combined_mask.sum())
    area_cm2 = calculate_wound_area(combined_mask, ppcm) if ppcm else None
    info["wound_area_cm2"] = round(area_cm2, 2) if area_cm2 is not None else None
    info["wound_area_px"] = area_px
    info["pixels_per_cm"] = ppcm
    info["marker_detected"] = ppcm is not None
    info["measurement_quality"] = "measured" if ppcm else "unavailable"

    image_rgb_full = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    try:
        inf_result = predict_infection(
            image_rgb_full, combined_mask, classifier_path, device,
        )
    except Exception:
        inf_result = {"infected_prob": -1.0, "predicted": "unknown"}
    infection_label = inf_result["predicted"]
    if inf_result["infected_prob"] < 0:
        infection_label = info["infection_filename"]
    info["infection"] = infection_label
    info["infection_prob"] = inf_result["infected_prob"]
    info["prediction_outcome"] = _prediction_outcome(
        info["metadata_infection"], infection_label,
    )

    if gt_mask is not None:
        dice, iou = _compute_mask_metrics(combined_mask, gt_mask)
        info["dice"] = round(dice, 6)
        info["iou"] = round(iou, 6)
    else:
        info["dice"] = None
        info["iou"] = None

    overlay = draw_info_panel(
        overlay, area_cm2, area_px, ppcm,
        infection_label, inf_result["infected_prob"],
        dice=info.get("dice"),
        iou=info.get("iou"),
        metadata_infection=info["metadata_infection"],
        prediction_outcome=info.get("prediction_outcome"),
    )

    return overlay, info


# ============================================================================
# Metrics Summary (Global)
# ============================================================================

def save_global_metrics_summary(
    yolo_metrics: dict,
    unet_metrics: dict,
    combined_metrics: dict,
    config: dict,
    results_dir: Path,
    infection_metrics: Optional[dict] = None,
) -> None:
    """Save a unified metrics_summary.json at the experiment root."""
    summary = {
        "yolo": yolo_metrics,
        "unet": unet_metrics,
        "combined": combined_metrics,
        "infection": infection_metrics or {},
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }
    path = results_dir / "metrics_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    script_dir = results_dir.parent
    manifest = build_experiment_manifest(
        config,
        script_dir,
        run_mode="evaluate_only",
        checkpoint_paths=[
            script_dir / "checkpoints" / "yolo" / "best.pt",
            get_unet_best_checkpoint_path(script_dir, config),
            script_dir / "checkpoints" / "infection" / "infection_classifier.pth",
        ],
    )
    save_experiment_manifest(manifest, results_dir)
    print(f"  -> Global metrics summary: {path}")


# ============================================================================
# Main
# ============================================================================

def main(stage: str = "all") -> dict:
    """Run the requested stage(s)."""
    config_path = SCRIPT_DIR / "config.yaml"
    config = load_config(config_path)
    set_seed(config.get("seed", 42))

    results = {"yolo": {}, "unet": {}, "combined": {}, "infection": {}}
    stages = (["convert", "yolo", "unet", "combined", "infection"]
              if stage == "all" else [stage])

    for s in stages:
        if s == "convert":
            print("\n" + "=" * 60)
            print("Converting COCO -> YOLO format")
            print("=" * 60)
            dataset_yaml = prepare_yolo_dataset(config, SCRIPT_DIR)
            validate_yolo_dataset(dataset_yaml)

        elif s == "yolo":
            results["yolo"] = train_yolo(config, SCRIPT_DIR)
            yolo_test = evaluate_yolo(config, SCRIPT_DIR)
            results["yolo"].update(yolo_test)
            predict_yolo(config, SCRIPT_DIR)

        elif s == "unet":
            results["unet"] = train_unet(config, SCRIPT_DIR)

        elif s == "combined":
            results["combined"] = evaluate_combined(config, SCRIPT_DIR)

        elif s == "infection":
            results["infection"] = train_infection_classifier(config, SCRIPT_DIR)

        else:
            print(f"[WARNING] Unknown stage: {s}")

    # Save global summary and report
    results_root = SCRIPT_DIR / "results"
    save_global_metrics_summary(
        results["yolo"], results["unet"], results["combined"],
        config, results_root, infection_metrics=results.get("infection"),
    )
    generate_report(
        results["yolo"], results["unet"], results["combined"],
        config, SCRIPT_DIR / "reports",
        infection_results=results.get("infection"),
    )

    print("\n" + "=" * 80)
    print("All stages completed.")
    print("=" * 80)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO11m + U-Net++ Training")
    parser.add_argument(
        "--stage", type=str, default="all",
        choices=["convert", "yolo", "unet", "combined", "infection", "all"],
        help="Which stage to run",
    )
    args = parser.parse_args()

    try:
        results = main(stage=args.stage)
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
