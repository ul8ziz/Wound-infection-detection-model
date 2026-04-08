#!/usr/bin/env python3
"""
Grid search over combined-pipeline hyperparameters on val (default) or test.

Uses one YOLO forward per image at low conf + Python score filtering (via ``yolo_conf_thresh``).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from combined.coco_eval import balanced_score, evaluate_combined_coco, pixel_metrics_on_split
from combined.config import CombinedInferenceConfig, combined_config_from_dict
from pipeline_utils import get_device, load_config
from train_model import build_unet_model, build_yolo_model, load_unet_checkpoint


YOLO_CONFS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
UNET_THRESH = [0.30, 0.40, 0.50, 0.60, 0.70]
ROI_PADS = [0.00, 0.05, 0.10, 0.15, 0.20]
PRESETS = ["none", "remove_small_cc", "binary_closing", "open_close", "fill_holes", "close_fill"]
MIN_AREAS = [0, 50, 100, 200]


def project_config_with_combined(base: Dict[str, Any], c: CombinedInferenceConfig) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    merged = c.to_dict()
    out["combined"] = {**(out.get("combined") or {}), **merged}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--max-configs", type=int, default=0, help="0 = all combinations")
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    config = load_config(SCRIPT_DIR / "config.yaml")
    base_ci = combined_config_from_dict(config)
    ann_key = "ann_val" if args.split == "val" else "ann_test"

    device = get_device()
    yolo_best = SCRIPT_DIR / "checkpoints" / "yolo" / "best.pt"
    unet_best = SCRIPT_DIR / "checkpoints" / "unet" / "best_model.pth"
    if not yolo_best.exists() or not unet_best.exists():
        print("ERROR: Missing checkpoints.")
        sys.exit(1)

    yolo_model = build_yolo_model(str(yolo_best))
    unet_model = build_unet_model(config)
    load_unet_checkpoint(unet_model, unet_best, device)
    unet_model.to(device)
    unet_model.eval()

    out_dir = Path(args.out_dir) if args.out_dir else SCRIPT_DIR / "results" / "combined" / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = list(product(YOLO_CONFS, UNET_THRESH, ROI_PADS, PRESETS, MIN_AREAS))
    if args.max_configs > 0:
        grid = grid[: args.max_configs]

    rows: List[Dict[str, Any]] = []
    if not grid:
        print("Empty grid.")
        return
    for i, (yc, ut, rp, preset, ma) in enumerate(grid):
        ci = replace(
            base_ci,
            yolo_conf_thresh=yc,
            unet_mask_thresh=ut,
            roi_padding=rp,
            postprocess_preset=preset,
            min_mask_area=ma,
            postprocess=[],
        )
        pconf = project_config_with_combined(config, ci)

        coco = evaluate_combined_coco(
            pconf,
            SCRIPT_DIR,
            yolo_model,
            unet_model,
            device,
            cfg=ci,
            ann_key=ann_key,
            write_json=False,
        )
        pix = pixel_metrics_on_split(
            pconf, SCRIPT_DIR, yolo_model, unet_model, device, cfg=ci, ann_key=ann_key,
        )
        bs = balanced_score({**coco, **pix}, ci.balanced_score_weights)

        row = {
            "yolo_conf_thresh": yc,
            "unet_mask_thresh": ut,
            "roi_padding": rp,
            "postprocess_preset": preset,
            "min_mask_area": ma,
            "balanced_score": bs,
            **coco,
            **pix,
        }
        rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(grid)}] last balanced_score={bs:.4f}")

    keys = list(rows[0].keys()) if rows else []
    csv_path = out_dir / f"tuning_results_{args.split}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    def sort_key(k: str) -> float:
        return float(rows[0].get(k, 0) or 0)

    rankings = {
        "best_combined_AP50": max(rows, key=lambda r: r.get("coco_combined_AP50", 0) or 0),
        "best_segm_AP75": max(rows, key=lambda r: r.get("coco_segm_AP75", 0) or 0),
        "best_balanced_score": max(rows, key=lambda r: r.get("balanced_score", 0) or 0),
    }
    with open(out_dir / f"tuning_rankings_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump(rankings, f, indent=2, default=str)

    best = rankings["best_balanced_score"]
    with open(out_dir / f"best_config_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, default=str)

    print(f"Wrote {csv_path} ({len(rows)} rows)")
    print("Best balanced:", json.dumps(best, indent=2, default=str))


if __name__ == "__main__":
    main()
