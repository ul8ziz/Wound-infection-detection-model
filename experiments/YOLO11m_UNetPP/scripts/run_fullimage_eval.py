#!/usr/bin/env python3
"""Quick script to run full-image U-Net evaluation."""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import torch
from pipeline_utils import load_config, get_device
from train_model import build_unet_model, load_unet_checkpoint, evaluate_unet_fullimage

config = load_config(SCRIPT_DIR / "config.yaml")
device = get_device()

model = build_unet_model(config)
model.to(device)
load_unet_checkpoint(model, SCRIPT_DIR / "checkpoints/unet/best_model.pth", device)

r = evaluate_unet_fullimage(model, config, SCRIPT_DIR, device, threshold=0.40, ann_key="ann_test")
print(f"Full-image Dice (GT boxes): {r['fullimage_dice']:.4f}")
print(f"Full-image IoU  (GT boxes): {r['fullimage_iou']:.4f}")
print(f"Images evaluated: {r['n_images']}")
