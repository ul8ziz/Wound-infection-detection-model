"""Verify gallery + training curves pipeline (headless)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from pipeline_utils import load_config, set_seed
from train_model import (
    display_experiment_gallery,
    display_training_curves,
    evaluate_combined,
)


def main() -> None:
    config = load_config(SCRIPT_DIR / "config.yaml", validate_combined=True)
    config["_config_path"] = str(SCRIPT_DIR / "config.yaml")
    set_seed(config.get("seed", 42))

    inf_hist = SCRIPT_DIR / "results" / "infection" / "training_history.json"
    if not inf_hist.is_file():
        print("[WARN] infection/training_history.json missing — curves panel 4–5 empty until §4.4 re-run.")

    print("\n[1/3] evaluate_combined ...")
    metrics = evaluate_combined(config, SCRIPT_DIR)
    assert metrics, "evaluate_combined returned empty metrics"
    wa_path = (
        SCRIPT_DIR / "results" / "combined"
        / "best_phase7_finetune_roi20" / "wound_areas.json"
    )
    if not wa_path.is_file():
        from experiment_io import get_combined_dirs
        wa_path = get_combined_dirs(SCRIPT_DIR, config)["results"] / "wound_areas.json"
    with open(wa_path, "r", encoding="utf-8") as f:
        wound_areas = json.load(f)
    required = {"dice", "iou", "metadata_infection", "prediction_outcome"}
    sample = wound_areas[0] if wound_areas else {}
    missing = required - set(sample.keys())
    assert not missing, f"wound_areas missing fields: {missing}"
    print(f"  wound_areas.json OK ({len(wound_areas)} records)")

    print("\n[2/3] display_training_curves ...")
    curves_path = display_training_curves(SCRIPT_DIR, config)
    assert curves_path.is_file(), f"Missing {curves_path}"
    print(f"  OK: {curves_path} ({curves_path.stat().st_size} bytes)")

    print("\n[3/3] display_experiment_gallery ...")
    gallery_path = display_experiment_gallery(SCRIPT_DIR, config, n_total=4)
    assert gallery_path.is_file(), f"Missing {gallery_path}"
    print(f"  OK: {gallery_path} ({gallery_path.stat().st_size} bytes)")

    print("\nAll verification checks passed.")


if __name__ == "__main__":
    main()
