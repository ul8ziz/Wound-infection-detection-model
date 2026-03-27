"""
YOLO11m + U-Net++ Wound Detection & Segmentation — Training Script
====================================================================

Self-contained training, evaluation, inference and reporting for the
combined YOLO11m-seg + U-Net++ pipeline.

Stages (CLI):
    python train_model.py --stage convert   # COCO -> YOLO label format
    python train_model.py --stage yolo      # Train YOLO11m-seg
    python train_model.py --stage unet      # Train U-Net++
    python train_model.py --stage combined  # Combined inference + eval
    python train_model.py --stage all       # Run all stages sequentially

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
    WoundDataset,
    WoundROIDataset,
    get_unet_transforms,
    IMAGENET_MEAN,
    IMAGENET_STD,
    WOUND_ONLY_CLASSES,
)

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


def train_yolo(config: dict, script_dir: Path) -> dict:
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

    model = build_yolo_model(yolo_cfg.get("model", "yolo11m-seg.pt"))
    yolo_project = script_dir / "checkpoints" / "yolo"
    yolo_project.mkdir(parents=True, exist_ok=True)

    train_results = model.train(
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
        mosaic=yolo_cfg.get("mosaic", 1.0),
        mixup=yolo_cfg.get("mixup", 0.1),
        hsv_h=yolo_cfg.get("hsv_h", 0.015),
        hsv_s=yolo_cfg.get("hsv_s", 0.7),
        hsv_v=yolo_cfg.get("hsv_v", 0.4),
        project=str(yolo_project),
        name="train",
        exist_ok=True,
        verbose=True,
        workers=config.get("num_workers", 0),
    )

    print("\nCopying YOLO results...")
    _copy_yolo_outputs(yolo_project / "train", script_dir)

    summary = _extract_yolo_metrics(yolo_project / "train")
    summary["training_completed"] = True
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
    """Save YOLO predictions on test images."""
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

    n = min(num_samples, len(test_images))
    for i in range(n):
        results = model(test_images[i], conf=conf_thresh, verbose=False)
        if results and len(results) > 0:
            plot = results[0].plot()
            fname = Path(test_images[i]).stem
            out_path = pred_dir / f"pred_{fname}.png"
            cv2.imwrite(str(out_path), plot)

    print(f"  -> Saved {n} YOLO predictions to {pred_dir}")
    return n


# ============================================================================
# Stage 2: U-Net++
# ============================================================================

def build_unet_model(config: dict) -> nn.Module:
    """Build U-Net++ with segmentation_models_pytorch."""
    import segmentation_models_pytorch as smp
    unet_cfg = config["unet"]
    model = smp.UnetPlusPlus(
        encoder_name=unet_cfg.get("encoder", "efficientnet-b3"),
        encoder_weights=unet_cfg.get("encoder_weights", "imagenet"),
        in_channels=unet_cfg.get("in_channels", 3),
        classes=unet_cfg.get("classes", 1),
        activation=None,
    )
    return model


class DiceLoss(nn.Module):
    """Differentiable Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )


class BCEDiceLoss(nn.Module):
    """Combined BCE + Dice loss."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(pred, target) + self.dice_weight * self.dice(pred, target)


def train_one_epoch_unet(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    print_freq: int = 10,
) -> float:
    """Train U-Net++ for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for i, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        preds = model(images)
        loss = criterion(preds, masks)

        if not math.isfinite(loss.item()):
            continue

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
        masks = masks.to(device)
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
        masks = masks.to(device)
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


def train_unet(config: dict, script_dir: Path) -> dict:
    """Full U-Net++ training loop with early stopping."""
    print("\n" + "=" * 60)
    print("Stage 2: Training U-Net++")
    print("=" * 60)

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
        test_ds, batch_size=unet_cfg.get("batch_size", 16),
        shuffle=False, num_workers=config.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )

    model = build_unet_model(config)
    model.to(device)
    print(f"  U-Net++ on {device} ({sum(p.numel() for p in model.parameters()):,} params)")

    criterion = BCEDiceLoss(
        bce_weight=unet_cfg.get("loss_bce_weight", 0.5),
        dice_weight=unet_cfg.get("loss_dice_weight", 0.5),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
    ckpt_dir = script_dir / "checkpoints" / "unet"
    results_dir = script_dir / "results" / "unet"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_losses": [],
        "val_losses": [],
        "dice_per_epoch": [],
        "iou_per_epoch": [],
    }
    best_dice = 0.0
    best_epoch = 0
    epochs_without_improve = 0

    start_time = time.time()
    try:
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch [{epoch}/{epochs}]")
            print("-" * 40)

            train_loss = train_one_epoch_unet(
                model, train_loader, optimizer, criterion, device, epoch,
            )
            val_loss = validate_one_epoch_unet(model, val_loader, criterion, device)
            metrics = evaluate_unet_metrics(model, val_loader, device)

            history["train_losses"].append(train_loss)
            history["val_losses"].append(val_loss)
            history["dice_per_epoch"].append(metrics["dice"])
            history["iou_per_epoch"].append(metrics["iou"])

            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Dice: {metrics['dice']:.4f} | IoU: {metrics['iou']:.4f}")

            if metrics["dice"] > best_dice:
                best_dice = metrics["dice"]
                best_epoch = epoch
                epochs_without_improve = 0
                print(f"  -> NEW BEST Dice: {best_dice:.4f}")
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
    print("U-Net++ Test Evaluation")
    print("=" * 60)
    test_metrics = evaluate_unet_metrics(model, test_loader, device)
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

@torch.no_grad()
def combined_inference(
    yolo_model,
    unet_model: nn.Module,
    image_path: str,
    device: torch.device,
    config: dict,
) -> Dict:
    """
    Two-stage inference on a single image.

    1. YOLO11m-seg detects wound bboxes + coarse masks
    2. For each detection, crop ROI and run U-Net++ for refined mask
    3. Map refined masks back to full image coordinates

    Returns dict with boxes, masks, scores, wound area info.
    """
    combined_cfg = config.get("combined", {})
    conf_thresh = combined_cfg.get("yolo_conf_thresh", 0.5)
    mask_thresh = combined_cfg.get("unet_mask_thresh", 0.5)
    roi_padding = combined_cfg.get("roi_padding", 0.1)
    unet_size = tuple(config["unet"].get("input_size", [256, 256]))

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        return {"error": f"Could not load {image_path}"}
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = image_rgb.shape[:2]

    yolo_results = yolo_model(image_path, conf=conf_thresh, verbose=False)
    if not yolo_results or len(yolo_results) == 0:
        return {"boxes": [], "masks": [], "scores": [], "image_shape": (img_h, img_w)}

    result = yolo_results[0]
    boxes_xyxy = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.array([])
    scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.array([])

    refined_masks = []
    refined_boxes = []
    refined_scores = []

    unet_model.eval()
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(device)

    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        pad_x, pad_y = bw * roi_padding, bh * roi_padding
        cx1 = max(0, int(x1 - pad_x))
        cy1 = max(0, int(y1 - pad_y))
        cx2 = min(img_w, int(x2 + pad_x))
        cy2 = min(img_h, int(y2 + pad_y))

        crop = image_rgb[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        crop_resized = cv2.resize(crop, (unet_size[1], unet_size[0]))
        crop_tensor = torch.from_numpy(crop_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        crop_tensor = (crop_tensor.to(device) - mean) / std

        pred = torch.sigmoid(unet_model(crop_tensor))
        pred_mask = (pred > mask_thresh).squeeze().cpu().numpy().astype(np.uint8)

        full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        crop_h, crop_w = cy2 - cy1, cx2 - cx1
        if crop_h > 0 and crop_w > 0:
            upscaled = cv2.resize(pred_mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
            full_mask[cy1:cy2, cx1:cx2] = upscaled

        refined_masks.append(full_mask)
        refined_boxes.append([cx1, cy1, cx2, cy2])
        refined_scores.append(float(scores[i]))

    return {
        "boxes": refined_boxes,
        "masks": refined_masks,
        "scores": refined_scores,
        "image_shape": (img_h, img_w),
    }


def calculate_wound_area(
    mask: np.ndarray,
    pixels_per_cm: float = 26.0,
) -> float:
    """Calculate wound area in cm^2 from a binary mask."""
    wound_pixels = int(mask.sum())
    return wound_pixels / (pixels_per_cm ** 2)


def evaluate_combined(config: dict, script_dir: Path) -> dict:
    """Run combined YOLO + U-Net++ evaluation on the test set."""
    print("\n" + "=" * 60)
    print("Stage 3: Combined YOLO11m + U-Net++ Evaluation")
    print("=" * 60)

    device = get_device()
    combined_cfg = config.get("combined", {})
    pixels_per_cm = combined_cfg.get("pixels_per_cm", 26.0)

    # Load YOLO
    yolo_best = script_dir / "checkpoints" / "yolo" / "best.pt"
    if not yolo_best.exists():
        print("[ERROR] YOLO best.pt not found. Train YOLO first.")
        return {}
    yolo_model = build_yolo_model(str(yolo_best))

    # Load U-Net++
    unet_best = script_dir / "checkpoints" / "unet" / "best_model.pth"
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

    # Build GT masks for comparison
    img_lookup = {img["id"]: img for img in test_coco["images"]}
    cat_ids = {c["id"] for c in test_coco["categories"]}
    img_anns: Dict[int, list] = {}
    for ann in test_coco["annotations"]:
        if ann["category_id"] in cat_ids:
            img_anns.setdefault(ann["image_id"], []).append(ann)

    dice_scores, iou_scores = [], []
    wound_areas = []
    results_combined = script_dir / "results" / "combined"
    results_combined.mkdir(parents=True, exist_ok=True)
    pred_dir = results_combined / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    num_qual = combined_cfg.get("num_qualitative_samples", 8)
    saved_count = 0

    for img_id, img_info in img_lookup.items():
        img_path = str(data_root / img_info["file_name"])
        if not Path(img_path).exists():
            continue

        pred = combined_inference(yolo_model, unet_model, img_path, device, config)
        if "error" in pred or not pred["masks"]:
            continue

        # GT mask
        orig_h, orig_w = img_info["height"], img_info["width"]
        gt_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for ann in img_anns.get(img_id, []):
            for seg in ann.get("segmentation", []):
                if len(seg) < 6:
                    continue
                poly = np.array(seg, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(gt_mask, [poly], 1)

        # Combine all predicted masks
        combined_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for m in pred["masks"]:
            if m.shape == (orig_h, orig_w):
                combined_mask = np.maximum(combined_mask, m)

        # Metrics
        smooth = 1e-6
        p_flat = combined_mask.flatten().astype(float)
        t_flat = gt_mask.flatten().astype(float)
        inter = (p_flat * t_flat).sum()
        union = p_flat.sum() + t_flat.sum()
        dice = (2 * inter + smooth) / (union + smooth)
        iou = (inter + smooth) / (union - inter + smooth)
        dice_scores.append(dice)
        iou_scores.append(iou)

        area_cm2 = calculate_wound_area(combined_mask, pixels_per_cm)
        wound_areas.append({"image": img_info["file_name"], "area_cm2": area_cm2})

        # Save qualitative prediction
        if saved_count < num_qual:
            img_bgr = cv2.imread(img_path)
            if img_bgr is not None:
                overlay = img_bgr.copy()
                mask_color = np.zeros_like(img_bgr)
                mask_color[:, :, 1] = 255  # green
                overlay[combined_mask > 0] = cv2.addWeighted(
                    overlay, 0.5, mask_color, 0.5, 0
                )[combined_mask > 0]
                for box in pred["boxes"]:
                    cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                fname = Path(img_info["file_name"]).stem
                out = pred_dir / f"combined_{fname}.png"
                cv2.imwrite(str(out), overlay)
                saved_count += 1

    n = max(1, len(dice_scores))
    metrics = {
        "mean_dice": sum(dice_scores) / n,
        "mean_iou": sum(iou_scores) / n,
        "n_images_evaluated": len(dice_scores),
        "n_predictions_saved": saved_count,
    }

    with open(results_combined / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(results_combined / "wound_areas.json", "w", encoding="utf-8") as f:
        json.dump(wound_areas, f, indent=2, ensure_ascii=False)

    print(f"\n  Mean Dice: {metrics['mean_dice']:.4f}")
    print(f"  Mean IoU:  {metrics['mean_iou']:.4f}")
    print(f"  Images evaluated: {metrics['n_images_evaluated']}")
    print(f"  Predictions saved: {saved_count} to {pred_dir}")
    return metrics


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

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  -> Report saved to {report_path}")
    return report_path


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
    pred_dirs = [
        results_dir / "yolo" / "predictions",
        results_dir / "combined" / "predictions",
    ]
    shown = 0
    for pred_dir in pred_dirs:
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
            axes[i].set_title(pngs[i].stem, fontsize=8)
            axes[i].axis("off")
        for i in range(n, len(axes)):
            axes[i].axis("off")
        fig.suptitle(f"Predictions from {pred_dir.parent.name}", fontsize=12)
        fig.tight_layout()
        plt.show()
        shown += n


def predict_single_image(
    yolo_model,
    unet_model: nn.Module,
    image_path: str,
    device: torch.device,
    config: dict,
) -> Tuple[np.ndarray, dict]:
    """
    Predict on a single image and return the annotated image + info dict.
    """
    combined_cfg = config.get("combined", {})
    pixels_per_cm = combined_cfg.get("pixels_per_cm", 26.0)

    pred = combined_inference(yolo_model, unet_model, image_path, device, config)
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return np.zeros((100, 100, 3), dtype=np.uint8), {"error": "Image not found"}

    info = {
        "file_name": Path(image_path).name,
        "n_detections": len(pred.get("boxes", [])),
        "scores": pred.get("scores", []),
    }

    fname = Path(image_path).name.lower()
    info["infection"] = "not_infected" if "-not-" in fname or "not_infected" in fname else "infected"
    info["confidence"] = max(pred.get("scores", [0.0]))

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
        cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(overlay, f"{score:.2f}", (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    area_cm2 = calculate_wound_area(combined_mask, pixels_per_cm)
    area_px = int(combined_mask.sum())
    info["wound_area_cm2"] = round(area_cm2, 2)
    info["wound_area_px"] = area_px

    cv2.putText(overlay, f"Area: {area_cm2:.1f} cm2 | {info['infection']}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

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
) -> None:
    """Save a unified metrics_summary.json at the experiment root."""
    summary = {
        "yolo": yolo_metrics,
        "unet": unet_metrics,
        "combined": combined_metrics,
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }
    path = results_dir / "metrics_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  -> Global metrics summary: {path}")


# ============================================================================
# Main
# ============================================================================

def main(stage: str = "all") -> dict:
    """Run the requested stage(s)."""
    config_path = SCRIPT_DIR / "config.yaml"
    config = load_config(config_path)
    set_seed(config.get("seed", 42))

    results = {"yolo": {}, "unet": {}, "combined": {}}
    stages = ["convert", "yolo", "unet", "combined"] if stage == "all" else [stage]

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

        else:
            print(f"[WARNING] Unknown stage: {s}")

    # Save global summary and report
    results_root = SCRIPT_DIR / "results"
    save_global_metrics_summary(
        results["yolo"], results["unet"], results["combined"],
        config, results_root,
    )
    generate_report(
        results["yolo"], results["unet"], results["combined"],
        config, SCRIPT_DIR / "reports",
    )

    print("\n" + "=" * 80)
    print("All stages completed.")
    print("=" * 80)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO11m + U-Net++ Training")
    parser.add_argument(
        "--stage", type=str, default="all",
        choices=["convert", "yolo", "unet", "combined", "all"],
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
