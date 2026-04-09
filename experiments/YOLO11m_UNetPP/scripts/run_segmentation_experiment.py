from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from combined.coco_eval import evaluate_combined_coco, pixel_metrics_on_split
from combined.error_analysis import run_error_analysis
from experiment_io import get_combined_dirs, get_experiment_name, get_unet_best_checkpoint_path
from pipeline_utils import get_device, load_config, set_seed
from train_model import (
    build_unet_model,
    build_yolo_model,
    evaluate_combined,
    load_unet_checkpoint,
    train_unet,
)


def deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_experiment_config(base_config_path: Path, override_path: Path) -> Dict[str, Any]:
    base_cfg = load_config(base_config_path, validate_combined=True)
    overrides = load_yaml(override_path)
    config = deep_update(base_cfg, overrides)
    config["_config_path"] = str(override_path.resolve())
    return config


def run_val_combined_metrics(config: Dict[str, Any]) -> Dict[str, Any]:
    device = get_device()
    yolo_best = SCRIPT_DIR / "checkpoints" / "yolo" / "best.pt"
    unet_best = get_unet_best_checkpoint_path(SCRIPT_DIR, config)
    if not yolo_best.exists() or not unet_best.exists():
        return {}

    yolo_model = build_yolo_model(str(yolo_best))
    unet_model = build_unet_model(config)
    load_unet_checkpoint(unet_model, unet_best, device)
    unet_model.to(device)
    unet_model.eval()

    metrics = {
        "val_pixel": pixel_metrics_on_split(
            config,
            SCRIPT_DIR,
            yolo_model,
            unet_model,
            device,
            ann_key="ann_val",
        ),
        "val_coco": evaluate_combined_coco(
            config,
            SCRIPT_DIR,
            yolo_model,
            unet_model,
            device,
            ann_key="ann_val",
            write_json=False,
        ),
    }
    combined_dir = get_combined_dirs(SCRIPT_DIR, config)["results"]
    combined_dir.mkdir(parents=True, exist_ok=True)
    with open(combined_dir / "validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def candidate_row(
    config: Dict[str, Any],
    unet_summary: Dict[str, Any],
    combined_summary: Dict[str, Any],
    val_summary: Dict[str, Any],
) -> Dict[str, Any]:
    unet_cfg = config.get("unet", {})
    experiment_name = get_experiment_name(config) or "baseline"
    input_size = unet_cfg.get("input_size", [256, 256])
    val_pixel = val_summary.get("val_pixel", {})
    val_coco = val_summary.get("val_coco", {})
    test_metrics = unet_summary.get("test_metrics", {})
    combined_dir = get_combined_dirs(SCRIPT_DIR, config)["results"]
    return {
        "experiment_name": experiment_name,
        "architecture": unet_cfg.get("architecture", "unetplusplus"),
        "input_size": "x".join(str(x) for x in input_size),
        "loss_type": unet_cfg.get("loss_type", "focal_dice"),
        "roi_crop_mode": unet_cfg.get("roi_crop_mode", "gt_only"),
        "multi_scale_refinement": bool(config.get("combined", {}).get("multi_scale_refinement", False)),
        "refinement_postprocess": config.get("combined", {}).get("refinement_postprocess", "none"),
        "val_best_dice": unet_summary.get("best_dice"),
        "val_best_epoch": unet_summary.get("best_epoch"),
        "unet_test_dice": test_metrics.get("dice"),
        "unet_test_iou": test_metrics.get("iou"),
        "val_combined_dice": val_pixel.get("mean_dice"),
        "val_combined_iou": val_pixel.get("mean_iou"),
        "val_segm_AP50": val_coco.get("coco_segm_AP50"),
        "val_segm_AP75": val_coco.get("coco_segm_AP75"),
        "test_combined_dice": combined_summary.get("mean_dice"),
        "test_combined_iou": combined_summary.get("mean_iou"),
        "test_segm_AP50": combined_summary.get("coco_segm_AP50"),
        "test_segm_AP75": combined_summary.get("coco_segm_AP75"),
        "test_combined_AP50": combined_summary.get("coco_combined_AP50"),
        "test_combined_AP75": combined_summary.get("coco_combined_AP75"),
        "results_dir": str(combined_dir),
    }


def regenerate_report(rows: List[Dict[str, Any]], report_path: Path, csv_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        fieldnames = []

    lines = [
        "# Segmentation improvement experiments",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Candidate table",
        "",
    ]
    if rows:
        headers = [
            "experiment_name",
            "architecture",
            "input_size",
            "loss_type",
            "roi_crop_mode",
            "multi_scale_refinement",
            "refinement_postprocess",
            "val_best_dice",
            "val_combined_dice",
            "val_segm_AP75",
            "test_combined_dice",
            "test_segm_AP75",
            "test_combined_AP75",
        ]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for row in rows:
            values = []
            for key in headers:
                value = row.get(key, "")
                if isinstance(value, float):
                    values.append(f"{value:.4f}")
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        lines.extend([
            "",
            f"CSV: `{csv_path}`",
            "",
            "## Experiment notes",
            "",
        ])
        for row in rows:
            lines.extend([
                f"### {row['experiment_name']}",
                "",
                f"- Results dir: `{row['results_dir']}`",
                f"- Architecture: `{row['architecture']}`",
                f"- Input size: `{row['input_size']}`",
                f"- ROI mode: `{row['roi_crop_mode']}`",
                f"- Loss: `{row['loss_type']}`",
                f"- Multi-scale: `{row['multi_scale_refinement']}`",
                f"- Refinement postprocess: `{row['refinement_postprocess']}`",
                "",
            ])
    else:
        lines.append("No experiment summaries collected yet.")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one segmentation experiment group and update comparison report.")
    parser.add_argument("--group", required=True, help="Experiment group label, e.g. group_B_yolo_like_crops")
    parser.add_argument("--config", required=True, help="Experiment override YAML path")
    parser.add_argument("--base-config", default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-combined", action="store_true")
    parser.add_argument("--skip-val-metrics", action="store_true")
    parser.add_argument("--run-error-analysis", action="store_true")
    parser.add_argument("--error-split", default="val", choices=["val", "test", "both"])
    args = parser.parse_args()

    base_config_path = Path(args.base_config).resolve()
    override_path = Path(args.config).resolve()
    config = build_experiment_config(base_config_path, override_path)
    config["experiment_name"] = args.group
    set_seed(config.get("seed", 42))

    unet_summary: Dict[str, Any] = {}
    combined_summary: Dict[str, Any] = {}
    val_summary: Dict[str, Any] = {}

    if not args.skip_train:
        unet_summary = train_unet(config, SCRIPT_DIR)
    else:
        combined_dir = get_combined_dirs(SCRIPT_DIR, config)["results"]
        unet_summary_path = SCRIPT_DIR / "results" / "unet" / args.group / "metrics_summary.json"
        if unet_summary_path.is_file():
            with open(unet_summary_path, "r", encoding="utf-8") as f:
                unet_summary = json.load(f)
        combined_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_combined:
        combined_summary = evaluate_combined(config, SCRIPT_DIR)
    else:
        combined_summary_path = get_combined_dirs(SCRIPT_DIR, config)["results"] / "metrics_summary.json"
        if combined_summary_path.is_file():
            with open(combined_summary_path, "r", encoding="utf-8") as f:
                combined_summary = json.load(f)

    if not args.skip_val_metrics:
        val_summary = run_val_combined_metrics(config)

    if args.run_error_analysis:
        splits = ["val", "test"] if args.error_split == "both" else [args.error_split]
        for split in splits:
            run_error_analysis(config, SCRIPT_DIR, split=split)

    row = candidate_row(config, unet_summary, combined_summary, val_summary)
    summaries_dir = SCRIPT_DIR / "results" / "experiments" / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    with open(summaries_dir / f"{args.group}.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "group": args.group,
                "config_path": str(override_path),
                "config": config,
                "unet_summary": unet_summary,
                "combined_summary": combined_summary,
                "val_summary": val_summary,
                "candidate_row": row,
            },
            f,
            indent=2,
            default=str,
        )

    rows: List[Dict[str, Any]] = []
    for summary_path in sorted(summaries_dir.glob("*.json")):
        with open(summary_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if "candidate_row" in payload:
            rows.append(payload["candidate_row"])

    regenerate_report(
        rows=rows,
        report_path=SCRIPT_DIR / "reports" / "segmentation_improvement_experiments.md",
        csv_path=SCRIPT_DIR / "reports" / "segmentation_candidate_table.csv",
    )
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
