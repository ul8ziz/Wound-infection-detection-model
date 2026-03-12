"""
Wound-Only Segmentation Baseline Training
==========================================

Clean baseline for wound-only segmentation using the standardized wound_focus_clean dataset.
Single class: wound.

Usage:
    cd experiments/maskrcnn
    python train_wound_only.py

Outputs:
    - checkpoints_wound_only/ (best_model.pth, last_checkpoint.pth, training_history.json)
    - results_wound_only/ (metrics, plots, predictions)
    - reports_wound_only/ (wound_only_training_report.md, review_summary_for_chatgpt.md)
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add script dir for imports
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

# Add scripts for augmentation_strategy
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
from train_model import (
    build_model,
    train_one_epoch,
    validate_one_epoch,
    evaluate_metrics,
    save_best_checkpoint,
    save_last_checkpoint,
    load_checkpoint,
    validate_dataset_labels,
)

# Import validation script
import validate_wound_only_dataset

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    "data_root": str(PROJECT_ROOT / "data" / "wound_focus_clean"),
    "ann_file_train": str(PROJECT_ROOT / "data" / "wound_focus_clean" / "train_wound_only.json"),
    "ann_file_val": str(PROJECT_ROOT / "data" / "wound_focus_clean" / "val_wound_only.json"),
    "ann_file_test": str(PROJECT_ROOT / "data" / "wound_focus_clean" / "test_wound_only.json"),
    "output_dir": str(SCRIPT_DIR / "checkpoints_wound_only"),
    "results_dir": str(SCRIPT_DIR / "results_wound_only"),
    "reports_dir": str(SCRIPT_DIR / "reports_wound_only"),
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
}

# ImageNet normalization (same as pipeline)
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


# ============================================================================
# Qualitative Predictions
# ============================================================================

def save_qualitative_predictions(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: Path,
    num_samples: int = 8,
    conf_thresh: float = 0.5,
) -> int:
    """
    Sample images, run inference, overlay predicted wound mask, save to output_dir/predictions/.
    Returns number of images saved.
    """
    model.eval()
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    dataset = data_loader.dataset
    if isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset

    images_collected = 0
    saved_count = 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            if saved_count >= num_samples:
                break
            images_dev = list(img.to(device) for img in images)
            outputs = model(images_dev)

            for idx, (img_tensor, target, output) in enumerate(zip(images, targets, outputs)):
                if saved_count >= num_samples:
                    break
                image_id = target["image_id"].item()
                img_info = dataset.images.get(image_id)
                if not img_info:
                    continue
                file_name = img_info.get("file_name", "")
                stem = Path(file_name).stem if file_name else f"img_{image_id}"

                # Denormalize image for visualization
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                img_np = img_np * STD + MEAN
                img_np = np.clip(img_np, 0, 1)
                img_np = (img_np * 255).astype(np.uint8)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                masks = output.get("masks")
                if masks is not None:
                    masks = masks.cpu().numpy()

                keep = scores >= conf_thresh
                if not keep.any():
                    max_score = float(scores.max()) if len(scores) > 0 else 0.0
                    conf_str = f"{max_score:.2f}"
                else:
                    # Merge all wound masks (single class)
                    h, w = img_np.shape[:2]
                    combined_mask = np.zeros((h, w), dtype=np.uint8)
                    for i in np.where(keep)[0]:
                        m = masks[i, 0]
                        if m.shape[0] != h or m.shape[1] != w:
                            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                        combined_mask = np.maximum(combined_mask, (m > 0.5).astype(np.uint8))
                    overlay = img_np.copy()
                    overlay[combined_mask > 0] = (
                        overlay[combined_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
                    ).astype(np.uint8)
                    img_np = overlay
                    conf_str = f"{float(scores[keep].max()):.2f}"

                out_path = pred_dir / f"pred_{stem}_conf_{conf_str}.png"
                cv2.imwrite(str(out_path), img_np)
                saved_count += 1

    return saved_count


# ============================================================================
# Plotting
# ============================================================================

def save_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    output_path: Path,
) -> None:
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


def save_ap_curves(
    metrics_per_epoch: List[Dict],
    output_dir: Path,
) -> None:
    """Save bbox AP, segm AP, and combined AP50 curves."""
    epochs = range(1, len(metrics_per_epoch) + 1)

    # Bbox AP
    bbox_keys = ["bbox_AP", "bbox_AP50", "bbox_AP75"]
    if any(k in (metrics_per_epoch[0] or {}) for k in bbox_keys):
        fig, ax = plt.subplots(figsize=(8, 5))
        for k in bbox_keys:
            vals = [m.get(k, 0) for m in metrics_per_epoch]
            ax.plot(epochs, vals, label=k, marker="o", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("AP")
        ax.set_title("Bbox AP Metrics")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "bbox_ap_curves.png", dpi=150)
        plt.close(fig)

    # Segm AP
    segm_keys = ["segm_AP", "segm_AP50", "segm_AP75"]
    if any(k in (metrics_per_epoch[0] or {}) for k in segm_keys):
        fig, ax = plt.subplots(figsize=(8, 5))
        for k in segm_keys:
            vals = [m.get(k, 0) for m in metrics_per_epoch]
            ax.plot(epochs, vals, label=k, marker="o", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("AP")
        ax.set_title("Segmentation AP Metrics")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "segm_ap_curves.png", dpi=150)
        plt.close(fig)

    # Combined AP50
    if metrics_per_epoch and "combined_AP50" in (metrics_per_epoch[0] or {}):
        fig, ax = plt.subplots(figsize=(8, 5))
        vals = [m.get("combined_AP50", 0) for m in metrics_per_epoch]
        ax.plot(epochs, vals, label="combined_AP50", marker="o", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("combined_AP50")
        ax.set_title("Combined AP50 (bbox + segm)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "combined_ap50_curve.png", dpi=150)
        plt.close(fig)


# ============================================================================
# Report Generation
# ============================================================================

def generate_wound_only_report(
    results: Dict,
    output_dir: Path,
    test_metrics: Optional[Dict] = None,
) -> None:
    """Generate wound_only_training_report.md and review_summary_for_chatgpt.md."""
    reports_dir = output_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    config = results.get("config", CONFIG)
    best_metric = results.get("best_metric", 0)
    best_epoch = results.get("best_epoch", 0)
    best_bbox = results.get("best_bbox_AP50", 0)
    best_segm = results.get("best_segm_AP50", 0)
    train_size = results.get("train_size", "?")
    val_size = results.get("val_size", "?")
    test_size = results.get("test_size", "?")
    training_time = results.get("training_time", 0)

    # wound_only_training_report.md
    report_lines = [
        "# Wound-Only Segmentation Training Report\n\n",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        "## Purpose of This Stage\n\n",
        "Establish a clean baseline for wound-only segmentation using the standardized ",
        "wound_focus_clean dataset. Single class: wound. No infection subclass segmentation.\n\n",
        "## Dataset Files Used\n\n",
        f"- Train: `{config.get('ann_file_train', '')}`\n",
        f"- Val: `{config.get('ann_file_val', '')}`\n",
        f"- Test: `{config.get('ann_file_test', '')}`\n",
        f"- Root: `{config.get('data_root', '')}`\n\n",
        "## Model/Config Used\n\n",
        f"- Model: Mask R-CNN ResNet-50-FPN\n",
        f"- num_classes: 2 (background + wound)\n",
        f"- batch_size: {config.get('batch_size', '?')}\n",
        f"- epochs: {config.get('epochs', '?')}\n",
        f"- lr: {config.get('lr', '?')}\n",
        f"- image_size: {config.get('image_size', '?')}\n",
        f"- use_medical_augmentation: {config.get('use_medical_augmentation', '?')}\n\n",
        "## Train/Val/Test Sizes\n\n",
        f"- Train: {train_size}\n",
        f"- Val: {val_size}\n",
        f"- Test: {test_size}\n\n",
        "## Training Behavior Summary\n\n",
        f"- Best epoch: {best_epoch}\n",
        f"- Training time: {training_time:.2f}s ({training_time/60:.2f} min)\n",
    ]
    if results.get("train_losses"):
        report_lines.append(f"- Final train loss: {results['train_losses'][-1]:.4f}\n")
    if results.get("val_losses"):
        report_lines.append(f"- Final val loss: {results['val_losses'][-1]:.4f}\n")
    report_lines.append("\n## Best Metrics Achieved (Validation)\n\n")
    report_lines.append(f"- combined_AP50: {best_metric:.4f}\n")
    report_lines.append(f"- bbox_AP50: {best_bbox:.4f}\n")
    report_lines.append(f"- segm_AP50: {best_segm:.4f}\n\n")
    if test_metrics:
        report_lines.append("## Test Set Metrics\n\n")
        for k, v in test_metrics.items():
            if isinstance(v, (int, float)):
                report_lines.append(f"- {k}: {v:.4f}\n")
            else:
                report_lines.append(f"- {k}: {v}\n")
        report_lines.append("\n")
    report_lines.append("## Comparison with Previous Multi-Class Attempt\n\n")
    report_lines.append(
        "Previous multi-class (8 classes) had near-zero segm_AP for subclasses due to "
        "annotation quality. This wound-only baseline focuses on the single well-annotated "
        "class (wound). Compare segm_AP50 above with prior results.\n\n"
    )
    report_lines.append("## Qualitative Prediction Observations\n\n")
    report_lines.append(
        "See `results_wound_only/predictions/` for example predictions. "
        "Review overlay quality and confidence scores.\n\n"
    )
    report_lines.append("## Issues Found\n\n")
    report_lines.append(results.get("issues_found", "None noted.\n\n"))
    report_lines.append("## Recommended Next Step\n\n")
    report_lines.append(
        "1. If segm_AP50 improved vs multi-class: proceed with wound-only + infection "
        "classification pipeline.\n"
        "2. If still weak: consider data augmentation, longer training, or architecture tuning.\n"
        "3. Add infected vs non-infected image-level classification using labels_infection.json.\n\n"
    )

    report_path = reports_dir / "wound_only_training_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    print(f"  → Report saved: {report_path}")

    # review_summary_for_chatgpt.md
    review_lines = [
        "# Wound-Only Segmentation Baseline — Review Summary for ChatGPT\n\n",
        "## What Was Implemented\n\n",
        "A clean wound-only segmentation baseline using Mask R-CNN on the standardized ",
        "wound_focus_clean dataset. Single class: wound. Training, validation, test evaluation, ",
        "plots, qualitative predictions, and reports.\n\n",
        "## Which Files Were Used\n\n",
        f"- Train: train_wound_only.json ({train_size} images)\n",
        f"- Val: val_wound_only.json ({val_size} images)\n",
        f"- Test: test_wound_only.json ({test_size} images)\n",
        "- Images: data/wound_focus_clean/images/\n\n",
        "## Best Key Metrics\n\n",
        f"- Validation combined_AP50: {best_metric:.4f}\n",
        f"- Validation bbox_AP50: {best_bbox:.4f}\n",
        f"- Validation segm_AP50: {best_segm:.4f}\n",
    ]
    if test_metrics:
        review_lines.append("\nTest set:\n")
        for k in ["bbox_AP50", "segm_AP50", "combined_AP50"]:
            if k in test_metrics:
                review_lines.append(f"- {k}: {test_metrics[k]:.4f}\n")
    review_lines.append("\n## Whether Wound-Only Direction Appears More Viable\n\n")
    review_lines.append(
        "Compare segm_AP50 with prior multi-class results. Wound-only removes noisy subclass "
        "annotations and focuses on the single well-defined wound region.\n\n"
    )
    review_lines.append("## Any Unresolved Issues\n\n")
    review_lines.append(results.get("unresolved_issues", "None.\n\n"))
    review_lines.append("## Recommended Next Action\n\n")
    review_lines.append(
        "1. Review qualitative predictions in results_wound_only/predictions/\n"
        "2. If metrics acceptable: add infected vs non-infected classification.\n"
        "3. If not: tune hyperparameters or augment data.\n\n"
    )

    review_path = reports_dir / "review_summary_for_chatgpt.md"
    with open(review_path, "w", encoding="utf-8") as f:
        f.writelines(review_lines)
    print(f"  → Review summary saved: {review_path}")


# ============================================================================
# Main
# ============================================================================

def main() -> Optional[Dict]:
    """Main training function for wound-only baseline."""
    print("=" * 80)
    print("Wound-Only Segmentation Baseline Training")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Pre-training validation
    print("Running pre-training validation...")
    if validate_wound_only_dataset.main() != 0:
        print("[ERROR] Dataset validation failed. Fix dataset before training.")
        return None
    print()

    set_seed(CONFIG["seed"])
    device = get_device(CONFIG.get("device_prefer_cuda", True))
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

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

    # Create datasets
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
        use_medical_augmentation=False,
        preserve_marker=CONFIG["preserve_marker"],
        intensity=CONFIG["intensity"],
        target_classes=WOUND_ONLY_CLASSES,
    )
    test_dataset = create_dataset(
        root=str(data_root),
        annotation_file=str(test_ann),
        train=False,
        image_size=CONFIG["image_size"],
        use_medical_augmentation=False,
        preserve_marker=CONFIG["preserve_marker"],
        intensity=CONFIG["intensity"],
        target_classes=WOUND_ONLY_CLASSES,
    )

    train_size = len(train_dataset)
    val_size = len(val_dataset)
    test_size = len(test_dataset)
    print(f"  Train: {train_size}, Val: {val_size}, Test: {test_size}")
    print()

    train_loader, val_loader = make_dataloaders(
        train_dataset,
        val_dataset,
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

    # Build model
    base_ds = train_dataset.dataset if isinstance(train_dataset, torch.utils.data.Subset) else train_dataset
    num_classes = base_ds.num_classes
    print(f"Model num_classes: {num_classes}")
    model = build_model(num_classes=num_classes, pretrained_backbone=True)
    model.to(device)

    # Startup report
    print("\n--- Startup report ---")
    unique_labels = validate_dataset_labels(train_loader, num_classes, num_batches=5)
    print(f"  Unique labels in batches: {unique_labels}")
    print("------------------------\n")

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=CONFIG["lr"],
        momentum=0.9,
        weight_decay=0.0005,
    )
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
        "training_start": datetime.now().isoformat(),
        "training_time": 0.0,
        "device": str(device),
        "num_classes": num_classes,
        "issues_found": "None noted.",
        "unresolved_issues": "None.",
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
                model,
                optimizer,
                train_loader,
                device,
                epoch,
                scheduler=scheduler,
                scheduler_step_per_iter=False,
                loss_clip_max=CONFIG.get("loss_clip_max"),
                loss_skip_threshold=CONFIG.get("loss_skip_threshold"),
                skip_invalid_targets=CONFIG.get("skip_invalid_targets", True),
            )
            results["train_losses"].append(train_stats["total_loss"])

            val_stats = validate_one_epoch(
                model,
                val_loader,
                device,
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

            is_best = combined_AP50 > best_combined_AP50
            if is_best:
                best_combined_AP50 = combined_AP50
                best_epoch = epoch + 1
                results["best_metric"] = best_combined_AP50
                results["best_epoch"] = best_epoch
                results["best_bbox_AP50"] = bbox_AP50
                results["best_segm_AP50"] = segm_AP50
                epochs_without_improve = 0
                print(f"  ✓ NEW BEST! combined_AP50: {best_combined_AP50:.4f} (Epoch {best_epoch})")
                save_best_checkpoint(
                    model,
                    epoch=epoch + 1,
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
                model,
                optimizer,
                scheduler,
                epoch=epoch + 1,
                metrics=metrics,
                output_dir=output_dir,
                filename="last_checkpoint.pth",
                scaler=None,
            )

            scheduler.step()

            if early_stop_patience > 0 and epochs_without_improve >= early_stop_patience:
                print(f"Early stopping after {epochs_without_improve} epochs without improvement.")
                break

    except KeyboardInterrupt:
        print("\n[WARNING] Training interrupted by user")
        results["interrupted"] = True
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        results["error"] = str(e)

    results["training_time"] = time.time() - start_time
    results["training_end"] = datetime.now().isoformat()

    # Load best model for test eval and qualitative preds
    best_path = output_dir / "best_model.pth"
    if best_path.exists():
        load_checkpoint(model, str(best_path))

    # Test evaluation
    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)
    test_metrics = None
    try:
        test_metrics = evaluate_metrics(model, test_loader, device)
        results["test_metrics"] = test_metrics
        print("Test metrics:")
        for k, v in test_metrics.items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    except Exception as e:
        print(f"[WARNING] Test evaluation failed: {e}")
        results["test_metrics"] = {}

    # Save metrics summary
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
    print(f"\n  → Metrics saved: {results_dir / 'metrics_summary.json'}")

    # Save training history
    history = {
        "train_losses": results["train_losses"],
        "val_losses": results["val_losses"],
        "metrics_per_epoch": results["metrics_per_epoch"],
        "config": CONFIG,
    }
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=str)
    print(f"  → History saved: {output_dir / 'training_history.json'}")

    # Plots
    print("\nSaving plots...")
    save_training_curves(
        results["train_losses"],
        results["val_losses"],
        results_dir / "training_curves.png",
    )
    save_ap_curves(results["metrics_per_epoch"], results_dir)
    print(f"  → Plots saved to {results_dir}")

    # Qualitative predictions
    print("\nSaving qualitative predictions...")
    n_saved = save_qualitative_predictions(
        model,
        test_loader,
        device,
        results_dir,
        num_samples=CONFIG.get("num_qualitative_samples", 8),
        conf_thresh=CONFIG.get("conf_thresh_qualitative", 0.5),
    )
    print(f"  → Saved {n_saved} prediction images to {results_dir / 'predictions'}")

    # Reports
    print("\nGenerating reports...")
    generate_wound_only_report(results, reports_dir, test_metrics)

    # Summary
    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    print(f"Best combined_AP50: {results['best_metric']:.4f} at epoch {best_epoch}")
    print(f"Training time: {results['training_time']:.2f}s ({results['training_time']/60:.2f} min)")
    print(f"Checkpoints: {output_dir}")
    print(f"Results: {results_dir}")
    print(f"Reports: {reports_dir}")
    print("=" * 80)
    print("[OK] Wound-only baseline training complete!")
    print("=" * 80)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Wound-Only Segmentation Baseline")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs (for quick test)")
    args = parser.parse_args()
    if args.epochs is not None:
        CONFIG["epochs"] = args.epochs
        print(f"[Override] epochs={args.epochs}")
    try:
        results = main()
        sys.exit(0 if results else 1)
    except Exception as e:
        print(f"\n[ERROR] Fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
