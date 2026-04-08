#!/usr/bin/env python3
"""
Staged grid search over combined-pipeline hyperparameters.

Stage A (coarse): core params -- conf, thresh, pad, bbox_mode, upscale.
Stage B (refine): narrow around best from A + postprocess, min_area, strategy.
Stage C (final):  fine-tune threshold in 0.05 steps near optimum.

Uses one YOLO forward per image at low conf + Python score filtering.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from combined.coco_eval import balanced_score, evaluate_combined_coco, pixel_metrics_on_split
from combined.config import CombinedInferenceConfig, combined_config_from_dict
from pipeline_utils import get_device, load_config
from train_model import build_unet_model, build_yolo_model, load_unet_checkpoint


# ── Stage A (coarse) defaults ─────────────────────────────────────────────────
STAGE_A_YOLO_CONFS = [0.15, 0.20, 0.25, 0.30]
STAGE_A_UNET_THRESH = [0.25, 0.35, 0.45, 0.55]
STAGE_A_ROI_PADS = [0.00, 0.05, 0.10]
STAGE_A_BBOX_MODES = ["yolo_xyxy", "mask_tight"]
STAGE_A_UPSCALES = ["linear_probs"]

# ── Stage B (refine) defaults ─────────────────────────────────────────────────
STAGE_B_PRESETS = [
    "none", "close_fill", "largest_then_fill", "keep_largest_component",
]
STAGE_B_MIN_AREAS = [0, 100]
STAGE_B_STRATEGIES = ["all_above_thresh", "highest_conf_single"]


def project_config_with_combined(base: Dict[str, Any], c: CombinedInferenceConfig) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    merged = c.to_dict()
    out["combined"] = {**(out.get("combined") or {}), **merged}
    return out


def precompute_yolo_cache(
    yolo_model,
    config: Dict[str, Any],
    ann_key: str,
    min_conf: float = 0.001,
) -> Dict[str, Any]:
    """Run YOLO once on every image in the split and cache results."""
    import json as _json
    project_root = SCRIPT_DIR.parent.parent
    ann_path = (project_root / config[ann_key]).resolve()
    data_root = (project_root / config["data_root"]).resolve()

    with open(ann_path, "r", encoding="utf-8") as f:
        coco = _json.load(f)

    cache: Dict[str, Any] = {}
    for img_info in coco["images"]:
        img_path = str(data_root / img_info["file_name"])
        if not Path(img_path).exists():
            continue
        results = yolo_model(img_path, conf=min_conf, verbose=False)
        cache[img_path] = results[0] if results else None
    print(f"  YOLO cache: {len(cache)} images pre-computed")
    return cache


def run_one_config(
    config: Dict[str, Any],
    ci: CombinedInferenceConfig,
    yolo_model,
    unet_model,
    device,
    ann_key: str,
    compute_pixel: bool = True,
    yolo_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Evaluate a single config and return merged metrics dict."""
    pconf = project_config_with_combined(config, ci)

    coco = evaluate_combined_coco(
        pconf, SCRIPT_DIR, yolo_model, unet_model, device,
        cfg=ci, ann_key=ann_key, write_json=False,
        yolo_cache=yolo_cache,
    )
    pix: Dict[str, float] = {}
    if compute_pixel:
        pix = pixel_metrics_on_split(
            pconf, SCRIPT_DIR, yolo_model, unet_model, device,
            cfg=ci, ann_key=ann_key, yolo_cache=yolo_cache,
        )
    merged = {**coco, **pix}
    merged["balanced_score"] = balanced_score(merged, ci.balanced_score_weights)
    return merged


def evaluate_grid(
    grid: List[Dict[str, Any]],
    config: Dict[str, Any],
    base_ci: CombinedInferenceConfig,
    yolo_model,
    unet_model,
    device,
    ann_key: str,
    stage_label: str = "",
    compute_pixel: bool = True,
    yolo_cache: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Run each config in *grid* and return list of result rows."""
    rows: List[Dict[str, Any]] = []
    total = len(grid)
    t0 = time.time()
    for i, params in enumerate(grid):
        ci = replace(base_ci, postprocess=[], **params)
        metrics = run_one_config(
            config, ci, yolo_model, unet_model, device, ann_key,
            compute_pixel, yolo_cache=yolo_cache,
        )
        row = {**params, **metrics}
        rows.append(row)
        if (i + 1) % 5 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (total - i - 1)
            bs = row.get("balanced_score", 0)
            print(f"  [{stage_label}] {i+1}/{total}  bs={bs:.4f}  ETA={eta:.0f}s")
    return rows


def write_results(rows: List[Dict[str, Any]], out_dir: Path, split: str, stage: str) -> None:
    if not rows:
        return
    csv_path = out_dir / f"tuning_{stage}_{split}.csv"
    keys = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {csv_path} ({len(rows)} rows)")


def best_from_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return max(rows, key=lambda r: r.get("balanced_score", 0))


def refine_around(best: Dict[str, Any], step_map: Dict[str, List]) -> List:
    """Generate +-1 step neighbours for continuous params."""
    neighbours = {}
    for k, options in step_map.items():
        val = best.get(k)
        if val in options:
            idx = options.index(val)
            lo = max(0, idx - 1)
            hi = min(len(options) - 1, idx + 1)
            neighbours[k] = sorted(set(options[lo: hi + 1]))
        else:
            neighbours[k] = [val]
    return neighbours


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--stage", choices=("A", "B", "C", "all"), default="all")
    ap.add_argument("--max-configs", type=int, default=0, help="0 = all")
    ap.add_argument("--out-dir", type=str, default="")
    ap.add_argument("--skip-pixel", action="store_true", help="Skip pixel metrics for speed on coarse stage")
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

    print("\nPre-computing YOLO results (one pass)...")
    yolo_cache = precompute_yolo_cache(yolo_model, config, ann_key)

    all_rows: List[Dict[str, Any]] = []
    best_A: Optional[Dict[str, Any]] = None
    best_B: Optional[Dict[str, Any]] = None

    # ── Stage A ───────────────────────────────────────────────────────────────
    if args.stage in ("A", "all"):
        print("\n=== Stage A: Coarse grid ===")
        grid_A = [
            {
                "yolo_conf_thresh": yc,
                "unet_mask_thresh": ut,
                "roi_padding": rp,
                "coco_bbox_mode": bm,
                "mask_upscale": up,
                "postprocess_preset": "none",
                "min_mask_area": 0,
                "bbox_selection_strategy": "all_above_thresh",
            }
            for yc, ut, rp, bm, up in product(
                STAGE_A_YOLO_CONFS, STAGE_A_UNET_THRESH, STAGE_A_ROI_PADS,
                STAGE_A_BBOX_MODES, STAGE_A_UPSCALES,
            )
        ]
        if args.max_configs > 0:
            grid_A = grid_A[: args.max_configs]
        print(f"  {len(grid_A)} configurations")
        rows_A = evaluate_grid(
            grid_A, config, base_ci, yolo_model, unet_model, device, ann_key,
            stage_label="A", compute_pixel=not args.skip_pixel,
            yolo_cache=yolo_cache,
        )
        write_results(rows_A, out_dir, args.split, "stageA")
        all_rows.extend(rows_A)
        best_A = best_from_rows(rows_A)
        print(f"\n  Best A balanced_score = {best_A['balanced_score']:.4f}")
        print(f"    conf={best_A['yolo_conf_thresh']}, thresh={best_A['unet_mask_thresh']}, "
              f"pad={best_A['roi_padding']}, bbox_mode={best_A['coco_bbox_mode']}, "
              f"upscale={best_A['mask_upscale']}")
    elif args.stage in ("B", "C"):
        best_json = out_dir / f"best_stageA_{args.split}.json"
        if best_json.exists():
            with open(best_json, "r") as f:
                best_A = json.load(f)

    if best_A is not None:
        with open(out_dir / f"best_stageA_{args.split}.json", "w") as f:
            json.dump(best_A, f, indent=2, default=str)

    # ── Stage B ───────────────────────────────────────────────────────────────
    if args.stage in ("B", "all") and best_A is not None:
        print("\n=== Stage B: Refine around best A ===")
        step_map = {
            "yolo_conf_thresh": STAGE_A_YOLO_CONFS,
            "unet_mask_thresh": STAGE_A_UNET_THRESH,
            "roi_padding": STAGE_A_ROI_PADS,
        }
        narrow = refine_around(best_A, step_map)
        if 0.10 not in narrow.get("roi_padding", []):
            narrow["roi_padding"] = sorted(set(narrow["roi_padding"] + [0.10]))

        # Phase B1: core sweep (thresh x pad x presets, fixed conf/strategy/area)
        grid_B1 = [
            {
                "yolo_conf_thresh": best_A["yolo_conf_thresh"],
                "unet_mask_thresh": ut,
                "roi_padding": rp,
                "coco_bbox_mode": best_A["coco_bbox_mode"],
                "mask_upscale": best_A["mask_upscale"],
                "postprocess_preset": pp,
                "min_mask_area": 0,
                "bbox_selection_strategy": "all_above_thresh",
            }
            for ut, rp, pp in product(
                narrow["unet_mask_thresh"],
                narrow["roi_padding"],
                STAGE_B_PRESETS,
            )
        ]
        # Phase B2: strategy + min_area sweep on best 2 (pad,thresh) combos
        best_combos = [(0.0, best_A["unet_mask_thresh"]), (0.10, 0.25)]
        grid_B2 = [
            {
                "yolo_conf_thresh": best_A["yolo_conf_thresh"],
                "unet_mask_thresh": ut,
                "roi_padding": rp,
                "coco_bbox_mode": best_A["coco_bbox_mode"],
                "mask_upscale": best_A["mask_upscale"],
                "postprocess_preset": "none",
                "min_mask_area": ma,
                "bbox_selection_strategy": strat,
            }
            for (rp, ut), ma, strat in product(
                best_combos, STAGE_B_MIN_AREAS, STAGE_B_STRATEGIES,
            )
        ]
        grid_B = grid_B1 + grid_B2
        if args.max_configs > 0:
            grid_B = grid_B[: args.max_configs]
        print(f"  {len(grid_B)} configurations")
        rows_B = evaluate_grid(
            grid_B, config, base_ci, yolo_model, unet_model, device, ann_key,
            stage_label="B", compute_pixel=True,
            yolo_cache=yolo_cache,
        )
        write_results(rows_B, out_dir, args.split, "stageB")
        all_rows.extend(rows_B)
        best_B = best_from_rows(rows_B)
        print(f"\n  Best B balanced_score = {best_B['balanced_score']:.4f}")
    elif args.stage == "C":
        best_json = out_dir / f"best_stageB_{args.split}.json"
        if best_json.exists():
            with open(best_json, "r") as f:
                best_B = json.load(f)

    if best_B is not None:
        with open(out_dir / f"best_stageB_{args.split}.json", "w") as f:
            json.dump(best_B, f, indent=2, default=str)

    # ── Stage C ───────────────────────────────────────────────────────────────
    if args.stage in ("C", "all") and best_B is not None:
        print("\n=== Stage C: Final threshold refinement ===")
        best_thresh = best_B["unet_mask_thresh"]
        fine_thresholds = sorted(set([
            round(best_thresh - 0.10, 2),
            round(best_thresh - 0.05, 2),
            best_thresh,
            round(best_thresh + 0.05, 2),
            round(best_thresh + 0.10, 2),
        ]))
        fine_thresholds = [t for t in fine_thresholds if 0.05 <= t <= 0.95]

        best_conf = best_B["yolo_conf_thresh"]
        fine_confs = sorted(set([
            round(best_conf - 0.05, 2),
            best_conf,
            round(best_conf + 0.05, 2),
        ]))
        fine_confs = [c for c in fine_confs if 0.05 <= c <= 0.80]

        grid_C = [
            {
                "yolo_conf_thresh": yc,
                "unet_mask_thresh": ut,
                "roi_padding": best_B["roi_padding"],
                "coco_bbox_mode": best_B["coco_bbox_mode"],
                "mask_upscale": best_B["mask_upscale"],
                "postprocess_preset": best_B["postprocess_preset"],
                "min_mask_area": best_B["min_mask_area"],
                "bbox_selection_strategy": best_B["bbox_selection_strategy"],
            }
            for yc, ut in product(fine_confs, fine_thresholds)
        ]
        print(f"  {len(grid_C)} configurations")
        rows_C = evaluate_grid(
            grid_C, config, base_ci, yolo_model, unet_model, device, ann_key,
            stage_label="C", compute_pixel=True,
            yolo_cache=yolo_cache,
        )
        write_results(rows_C, out_dir, args.split, "stageC")
        all_rows.extend(rows_C)

    # ── Final summary ─────────────────────────────────────────────────────────
    if all_rows:
        write_results(all_rows, out_dir, args.split, "all_stages")
        best_final = best_from_rows(all_rows)
        with open(out_dir / f"best_config_{args.split}.json", "w") as f:
            json.dump(best_final, f, indent=2, default=str)

        rankings = {
            "best_balanced_score": best_final,
            "best_combined_AP50": max(all_rows, key=lambda r: r.get("coco_combined_AP50", 0) or 0),
            "best_segm_AP75": max(all_rows, key=lambda r: r.get("coco_segm_AP75", 0) or 0),
            "best_bbox_AP75": max(all_rows, key=lambda r: r.get("coco_bbox_AP75", 0) or 0),
        }
        with open(out_dir / f"tuning_rankings_{args.split}.json", "w") as f:
            json.dump(rankings, f, indent=2, default=str)

        print(f"\n{'='*60}")
        print(f"BEST FINAL CONFIG (balanced_score = {best_final['balanced_score']:.4f}):")
        print(json.dumps(best_final, indent=2, default=str))
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
