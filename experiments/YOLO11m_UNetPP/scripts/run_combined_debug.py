#!/usr/bin/env python3
"""Save combined-pipeline debug visualizations for a subset of val/test images."""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import cv2
import numpy as np

from combined.config import combined_config_from_dict
from combined.debug_viz import save_combined_debug_steps
from experiment_io import get_unet_best_checkpoint_path
from pipeline_utils import get_device, load_config
from train_model import build_unet_model, build_yolo_model, load_unet_checkpoint


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--max-images", type=int, default=16)
    args = p.parse_args()

    config = load_config(SCRIPT_DIR / "config.yaml")
    cfg_inf = combined_config_from_dict(config)
    out_root = SCRIPT_DIR / cfg_inf.debug_output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    device = get_device()
    yolo_best = SCRIPT_DIR / "checkpoints" / "yolo" / "best.pt"
    unet_best = get_unet_best_checkpoint_path(SCRIPT_DIR, config)
    if not yolo_best.exists() or not unet_best.exists():
        print("ERROR: Train YOLO and U-Net++ first (checkpoints missing).")
        sys.exit(1)

    yolo_model = build_yolo_model(str(yolo_best))
    unet_model = build_unet_model(config)
    load_unet_checkpoint(unet_model, unet_best, device)
    unet_model.to(device)
    unet_model.eval()

    project_root = SCRIPT_DIR.parent.parent
    ann_key = "ann_val" if args.split == "val" else "ann_test"
    ann_path = (project_root / config[ann_key]).resolve()
    data_root = (project_root / config["data_root"]).resolve()

    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    img_lookup = {img["id"]: img for img in coco["images"]}
    cat_ids = {c["id"] for c in coco["categories"]}
    img_anns: dict = {}
    for ann in coco["annotations"]:
        if ann["category_id"] in cat_ids:
            img_anns.setdefault(ann["image_id"], []).append(ann)

    n = 0
    for _img_id, img_info in img_lookup.items():
        if n >= args.max_images:
            break
        img_path = str(data_root / img_info["file_name"])
        if not Path(img_path).exists():
            continue

        orig_h, orig_w = img_info["height"], img_info["width"]
        gt_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for ann in img_anns.get(img_info["id"], []):
            for seg in ann.get("segmentation", []):
                if len(seg) < 6:
                    continue
                poly = np.array(seg, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(gt_mask, [poly], 1)

        stem = Path(img_info["file_name"]).stem
        save_combined_debug_steps(
            yolo_model,
            unet_model,
            img_path,
            device,
            config,
            out_root,
            stem,
            cfg=cfg_inf,
            gt_mask=gt_mask,
        )
        n += 1

    print(f"Wrote debug panels for {n} images to {out_root}")


if __name__ == "__main__":
    main()
