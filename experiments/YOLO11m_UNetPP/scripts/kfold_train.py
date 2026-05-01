"""
5-Fold Cross-Validation training for U-Net++.

Splits the wound-only training annotations into 5 stratified folds
(stratified by infected/non-infected labels from filenames),
trains one U-Net++ model per fold using the current config,
and saves 5 best checkpoints for ensemble inference.

Usage:
    python scripts/kfold_train.py [--n-folds 5] [--config config.yaml]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from pipeline_utils import load_config, set_seed


def _infection_label(file_name: str) -> int:
    """0 = non-infected, 1 = infected, based on filename convention."""
    name = file_name.lower()
    if "not_infected" in name or "-not-" in name:
        return 0
    if "infected" in name:
        return 1
    return -1


def stratified_kfold_image_ids(
    coco: dict,
    n_folds: int,
    seed: int = 42,
) -> List[List[int]]:
    """Stratified K-fold split by infection label at image level."""
    rng = np.random.RandomState(seed)

    img_labels: Dict[int, int] = {}
    for img in coco["images"]:
        img_labels[img["id"]] = _infection_label(img["file_name"])

    groups: Dict[int, List[int]] = defaultdict(list)
    for img_id, label in img_labels.items():
        groups[label].append(img_id)

    for group_ids in groups.values():
        rng.shuffle(group_ids)

    folds: List[List[int]] = [[] for _ in range(n_folds)]
    for label, ids in groups.items():
        for i, img_id in enumerate(ids):
            folds[i % n_folds].append(img_id)

    for fold in folds:
        rng.shuffle(fold)

    return folds


def create_fold_json(
    coco: dict,
    image_ids: set,
    output_path: Path,
) -> Path:
    """Write a COCO JSON containing only images in image_ids."""
    fold_images = [img for img in coco["images"] if img["id"] in image_ids]
    fold_anns = [ann for ann in coco["annotations"] if ann["image_id"] in image_ids]

    fold_coco = {
        "images": fold_images,
        "annotations": fold_anns,
        "categories": coco["categories"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fold_coco, f, indent=2, ensure_ascii=False)
    return output_path


def train_single_fold(
    fold_idx: int,
    config: dict,
    train_ann_path: Path,
    val_ann_path: Path,
    script_dir: Path,
) -> dict:
    """Train one U-Net++ fold and return metrics summary."""
    from train_model import (
        build_unet_model,
        build_unet_criterion,
        train_one_epoch_unet,
        validate_one_epoch_unet,
        evaluate_unet_metrics,
        save_unet_checkpoint,
        load_unet_checkpoint,
    )
    from pipeline_utils import (
        get_device,
        WoundROIDataset,
        MixedWoundROIDataset,
        get_unet_transforms,
        make_unet_dataloaders,
        unet_collate_fn,
    )
    import math
    import torch

    unet_cfg = config["unet"]
    device = get_device()
    image_size = tuple(unet_cfg["input_size"])
    roi_padding = unet_cfg.get("roi_padding", 0.1)
    train_crop_mode = str(unet_cfg.get("roi_crop_mode", "gt_only"))
    eval_crop_mode = str(unet_cfg.get("eval_roi_crop_mode", "gt_only"))
    crop_mix_weights = dict(unet_cfg.get("roi_mix_weights") or {})
    crop_jitter = dict(unet_cfg.get("roi_jitter") or {})
    yolo_roi_cache_path = unet_cfg.get("yolo_roi_cache_path")
    eval_yolo_roi_cache_path = unet_cfg.get("eval_yolo_roi_cache_path", yolo_roi_cache_path)
    yolo_match_iou_min = float(unet_cfg.get("yolo_match_iou_min", 0.0))

    project_root = script_dir.parent.parent
    data_root_train = (project_root / config.get("data_root_train", config["data_root"])).resolve()
    data_root = (project_root / config["data_root"]).resolve()

    train_dataset_cls = MixedWoundROIDataset if train_crop_mode == "mixed" else WoundROIDataset
    train_ds = train_dataset_cls(
        root=data_root_train,
        annotation_file=str(train_ann_path),
        transforms=get_unet_transforms(train=True, image_size=image_size),
        roi_padding=roi_padding,
        crop_mode=train_crop_mode,
        crop_mix_weights=crop_mix_weights,
        crop_jitter=crop_jitter,
        yolo_roi_cache_path=(
            str((project_root / yolo_roi_cache_path).resolve())
            if yolo_roi_cache_path and not Path(yolo_roi_cache_path).is_absolute()
            else yolo_roi_cache_path
        ),
        yolo_match_iou_min=yolo_match_iou_min,
    )
    val_ds = WoundROIDataset(
        root=data_root_train,
        annotation_file=str(val_ann_path),
        transforms=get_unet_transforms(train=False, image_size=image_size),
        roi_padding=roi_padding,
        crop_mode=eval_crop_mode,
        crop_mix_weights=crop_mix_weights,
        crop_jitter=crop_jitter,
        yolo_roi_cache_path=(
            str((project_root / eval_yolo_roi_cache_path).resolve())
            if eval_yolo_roi_cache_path and not Path(eval_yolo_roi_cache_path).is_absolute()
            else eval_yolo_roi_cache_path
        ),
        yolo_match_iou_min=yolo_match_iou_min,
    )

    print(f"\n  Fold {fold_idx}: Train {len(train_ds)} samples, Val {len(val_ds)} samples")

    train_loader, val_loader = make_unet_dataloaders(
        train_ds, val_ds,
        batch_size=unet_cfg.get("batch_size", 4),
        num_workers=config.get("num_workers", 0),
    )

    model = build_unet_model(config)
    model.to(device)

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

    ckpt_dir = script_dir / "checkpoints" / "unet" / "kfold" / f"fold_{fold_idx}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    use_amp = device.type == "cuda" and bool(unet_cfg.get("use_amp", True))
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_dice = -1.0
    best_epoch = 0
    epochs_without_improve = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch_unet(
            model, train_loader, optimizer, criterion, device, epoch,
            scaler=scaler, print_freq=50,
        )
        val_loss = validate_one_epoch_unet(model, val_loader, criterion, device)
        metrics = evaluate_unet_metrics(model, val_loader, device, threshold=eval_threshold)
        val_dice = metrics["dice"]
        if not math.isfinite(val_dice):
            val_dice = best_dice

        print(f"  Fold {fold_idx} Epoch [{epoch}/{epochs}] "
              f"Train: {train_loss:.4f} Val: {val_loss:.4f} "
              f"Dice: {metrics['dice']:.4f} IoU: {metrics['iou']:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            epochs_without_improve = 0
            save_unet_checkpoint(
                model, optimizer, scheduler, epoch, metrics,
                ckpt_dir / "best_model.pth",
            )
            print(f"    -> NEW BEST Fold {fold_idx} Dice: {best_dice:.4f}")
        else:
            epochs_without_improve += 1

        scheduler.step()

        if patience > 0 and epochs_without_improve >= patience:
            print(f"  Fold {fold_idx} early stopping at epoch {epoch}")
            break

    training_time = time.time() - start_time
    return {
        "fold": fold_idx,
        "best_dice": best_dice,
        "best_epoch": best_epoch,
        "training_time_s": training_time,
        "checkpoint": str(ckpt_dir / "best_model.pth"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="K-Fold CV training for U-Net++")
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.get("seed", 42)
    set_seed(seed)
    n_folds = args.n_folds

    ann_train_path = (PROJECT_ROOT / config["ann_train"]).resolve()
    with open(ann_train_path, "r", encoding="utf-8") as f:
        coco_train = json.load(f)

    ann_val_path = (PROJECT_ROOT / config["ann_val"]).resolve()

    print("=" * 60)
    print(f"K-Fold Cross-Validation Training ({n_folds} folds)")
    print("=" * 60)
    print(f"  Training annotations: {ann_train_path}")
    print(f"  Total images: {len(coco_train['images'])}")
    print(f"  Total annotations: {len(coco_train['annotations'])}")

    folds = stratified_kfold_image_ids(coco_train, n_folds, seed=seed)
    for i, fold_ids in enumerate(folds):
        n_inf = sum(1 for iid in fold_ids
                    for img in coco_train["images"]
                    if img["id"] == iid and _infection_label(img["file_name"]) == 1)
        print(f"  Fold {i}: {len(fold_ids)} images ({n_inf} infected)")

    fold_json_dir = SCRIPT_DIR / "results" / "kfold" / "fold_jsons"
    fold_json_dir.mkdir(parents=True, exist_ok=True)

    all_image_ids = set()
    for fold_ids in folds:
        all_image_ids.update(fold_ids)

    fold_results = []
    for fold_idx in range(n_folds):
        print(f"\n{'=' * 60}")
        print(f"  Training Fold {fold_idx}/{n_folds - 1}")
        print(f"{'=' * 60}")

        val_ids = set(folds[fold_idx])
        train_ids = all_image_ids - val_ids

        train_json = fold_json_dir / f"train_fold{fold_idx}.json"
        val_json = fold_json_dir / f"val_fold{fold_idx}.json"
        create_fold_json(coco_train, train_ids, train_json)
        create_fold_json(coco_train, val_ids, val_json)
        print(f"  Train: {len(train_ids)} images, Val: {len(val_ids)} images")

        fold_config = copy.deepcopy(config)
        fold_config["ann_train"] = str(train_json)

        result = train_single_fold(
            fold_idx=fold_idx,
            config=fold_config,
            train_ann_path=train_json,
            val_ann_path=val_json,
            script_dir=SCRIPT_DIR,
        )
        fold_results.append(result)

    print("\n" + "=" * 60)
    print("K-Fold Results Summary")
    print("=" * 60)

    dices = []
    for r in fold_results:
        dices.append(r["best_dice"])
        print(f"  Fold {r['fold']}: Dice={r['best_dice']:.4f} "
              f"(epoch {r['best_epoch']}, {r['training_time_s']:.0f}s)")

    print(f"\n  Mean Dice: {np.mean(dices):.4f} +/- {np.std(dices):.4f}")

    results_dir = SCRIPT_DIR / "results" / "kfold"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_folds": n_folds,
        "mean_dice": float(np.mean(dices)),
        "std_dice": float(np.std(dices)),
        "fold_results": fold_results,
    }
    with open(results_dir / "kfold_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {results_dir / 'kfold_summary.json'}")


if __name__ == "__main__":
    main()
