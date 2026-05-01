"""
Evaluate 5-fold ensemble on the combined pipeline (test set).

Loads all K-fold U-Net++ checkpoints, passes them as a list to
combined_inference (which averages their probability maps), and
computes COCO + Dice/IoU metrics.

Usage:
    python scripts/eval_ensemble.py [--config config.yaml]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from pipeline_utils import get_device, load_config
from train_model import (
    build_segmentation_model,
    build_yolo_model,
    load_unet_checkpoint,
    calculate_wound_area,
)
from combined.inference import combined_inference
from combined.coco_eval import evaluate_combined_coco
from experiment_io import snapshot_config


def load_ensemble_models(
    config: dict,
    fold_dirs: List[Path],
    device: torch.device,
) -> List[torch.nn.Module]:
    """Load multiple U-Net++ checkpoints into a list of eval-mode models."""
    models = []
    for fold_dir in fold_dirs:
        ckpt_path = fold_dir / "best_model.pth"
        if not ckpt_path.exists():
            print(f"  [WARNING] Missing checkpoint: {ckpt_path}")
            continue
        model = build_segmentation_model(config)
        load_unet_checkpoint(model, ckpt_path, device)
        model.to(device)
        model.eval()
        models.append(model)
        print(f"  Loaded fold model: {ckpt_path.parent.name}")
    return models


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate K-fold ensemble on combined pipeline")
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--also-single", action="store_true",
                        help="Also evaluate the single best Phase 4 model for comparison")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()

    print("=" * 60)
    print("Ensemble Evaluation: K-Fold U-Net++ on Combined Pipeline")
    print("=" * 60)

    # Load YOLO
    yolo_best = SCRIPT_DIR / "checkpoints" / "yolo" / "best.pt"
    if not yolo_best.exists():
        print("[ERROR] YOLO best.pt not found.")
        return
    yolo_model = build_yolo_model(str(yolo_best))

    # Load ensemble U-Net++ models
    kfold_base = SCRIPT_DIR / "checkpoints" / "unet" / "kfold"
    fold_dirs = sorted(kfold_base.iterdir()) if kfold_base.exists() else []
    fold_dirs = [d for d in fold_dirs if d.is_dir() and (d / "best_model.pth").exists()]

    if len(fold_dirs) < 2:
        print(f"[ERROR] Need at least 2 fold checkpoints, found {len(fold_dirs)}")
        return

    print(f"\n  Loading {len(fold_dirs)} fold models...")
    ensemble_models = load_ensemble_models(config, fold_dirs, device)
    print(f"  Ensemble size: {len(ensemble_models)} models")

    # Load test data
    test_ann_path = (PROJECT_ROOT / config["ann_test"]).resolve()
    data_root = (PROJECT_ROOT / config["data_root"]).resolve()

    with open(test_ann_path, "r", encoding="utf-8") as f:
        test_coco = json.load(f)

    img_lookup = {img["id"]: img for img in test_coco["images"]}
    cat_ids = {c["id"] for c in test_coco["categories"]}
    img_anns: Dict[int, list] = {}
    for ann in test_coco["annotations"]:
        if ann["category_id"] in cat_ids:
            img_anns.setdefault(ann["image_id"], []).append(ann)

    combined_cfg = config.get("combined", {})
    pixels_per_cm = combined_cfg.get("pixels_per_cm", 60.0)

    # Results directory
    results_dir = SCRIPT_DIR / "results" / "combined" / "ensemble_kfold"
    results_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = results_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    snapshot_config(
        config, results_dir,
        Path(config["_config_path"]) if config.get("_config_path") else None,
    )

    # Run evaluation
    dice_scores, iou_scores = [], []
    wound_areas = []
    saved_count = 0
    num_qual = combined_cfg.get("num_qualitative_samples", 8)

    n_total = 0
    n_missed = 0

    print(f"\n  Evaluating on {len(img_lookup)} test images...")
    for img_id, img_info in img_lookup.items():
        img_path = str(data_root / img_info["file_name"])
        if not Path(img_path).exists():
            continue
        n_total += 1

        pred = combined_inference(
            yolo_model, ensemble_models, img_path, device, config,
        )
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
            dice_scores.append(0.0)
            iou_scores.append(0.0)
            continue

        combined_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for m in pred["masks"]:
            if m.shape == (orig_h, orig_w):
                combined_mask = np.maximum(combined_mask, m)

        smooth = 1e-6
        p_flat = combined_mask.flatten().astype(float)
        t_flat = gt_mask.flatten().astype(float)
        inter = (p_flat * t_flat).sum()
        union = p_flat.sum() + t_flat.sum()
        dice = (2 * inter + smooth) / (union + smooth)
        iou = (inter + smooth) / (union - inter + smooth)
        dice_scores.append(dice)
        iou_scores.append(iou)

        effective_ppcm = pred.get("pixels_per_cm") or pixels_per_cm
        area_cm2 = calculate_wound_area(combined_mask, effective_ppcm)
        wound_areas.append({
            "image": img_info["file_name"],
            "area_cm2": area_cm2,
            "pixels_per_cm": effective_ppcm,
            "marker_detected": pred.get("pixels_per_cm") is not None,
        })

        if saved_count < num_qual:
            img_bgr = cv2.imread(img_path)
            if img_bgr is not None:
                overlay = img_bgr.copy()
                mask_color = np.zeros_like(img_bgr)
                mask_color[:, :, 1] = 255
                overlay[combined_mask > 0] = cv2.addWeighted(
                    overlay, 0.5, mask_color, 0.5, 0
                )[combined_mask > 0]
                for box in pred["boxes"]:
                    cv2.rectangle(overlay,
                                  (int(box[0]), int(box[1])),
                                  (int(box[2]), int(box[3])),
                                  (0, 255, 0), 2)
                fname = Path(img_info["file_name"]).stem
                cv2.imwrite(str(pred_dir / f"ensemble_{fname}.png"), overlay)
                saved_count += 1

    # Compute metrics
    n_full = max(1, len(dice_scores))
    n_detected = n_full - n_missed

    metrics = {
        "mean_dice": sum(dice_scores) / n_full,
        "mean_iou": sum(iou_scores) / n_full,
        "mean_dice_conditional": sum(dice_scores) / max(1, n_detected),
        "mean_iou_conditional": sum(iou_scores) / max(1, n_detected),
        "n_images_total": n_total,
        "n_images_evaluated": n_detected,
        "n_images_missed": n_missed,
        "n_predictions_saved": saved_count,
        "ensemble_size": len(ensemble_models),
    }

    # COCO evaluation
    print("\n  Running COCO evaluation...")
    coco_metrics = evaluate_combined_coco(
        config, SCRIPT_DIR, yolo_model, ensemble_models, device,
    )
    if coco_metrics:
        metrics.update(coco_metrics)

    # Save results
    with open(results_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(results_dir / "wound_areas.json", "w", encoding="utf-8") as f:
        json.dump(wound_areas, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print("Ensemble Results")
    print(f"{'=' * 60}")
    print(f"  Ensemble size:    {len(ensemble_models)} models")
    print(f"  Mean Dice:        {metrics['mean_dice']:.4f}")
    print(f"  Mean IoU:         {metrics['mean_iou']:.4f}")
    if coco_metrics:
        print(f"  COCO bbox AP50:   {coco_metrics.get('coco_bbox_AP50', 0):.4f}")
        print(f"  COCO bbox AP75:   {coco_metrics.get('coco_bbox_AP75', 0):.4f}")
        print(f"  COCO segm AP50:   {coco_metrics.get('coco_segm_AP50', 0):.4f}")
        print(f"  COCO segm AP75:   {coco_metrics.get('coco_segm_AP75', 0):.4f}")
        print(f"  COCO combined AP50: {coco_metrics.get('coco_combined_AP50', 0):.4f}")
        print(f"  COCO combined AP75: {coco_metrics.get('coco_combined_AP75', 0):.4f}")
    print(f"  Images evaluated: {metrics['n_images_evaluated']}")
    print(f"  Predictions saved: {saved_count}")


if __name__ == "__main__":
    main()
