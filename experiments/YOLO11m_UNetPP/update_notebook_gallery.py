"""Update training_pipeline.ipynb for gallery + training curves plan."""
import json
from pathlib import Path

NOTEBOOK = Path("training_pipeline.ipynb")


def to_source(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    return lines if lines else [text]


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    # Cell 2: add imports
    src2 = "".join(nb["cells"][2]["source"])
    if "display_training_curves" not in src2:
        src2 = src2.replace(
            "    display_results_predictions,\n)",
            "    display_results_predictions,\n    display_training_curves,\n    display_experiment_gallery,\n)",
        )
        nb["cells"][2]["source"] = to_source(src2)

    # Cell 8: fix SKIP_UNET best_dice from metrics_summary.json
    skip_unet_old = """if SKIP_UNET:
    print(f"[SKIP] U-Net++ checkpoint found — reusing {unet_best}")
    results_unet = get_unet_dirs(SCRIPT_DIR, CONFIG)["results"]
    history_path = results_unet / "training_history.json"
    if history_path.is_file():
        with open(history_path, "r", encoding="utf-8") as f:
            unet_history = json.load(f)
        best_dice = float(unet_history.get("best_dice", 0.0))
        best_epoch = int(unet_history.get("best_epoch", 0))
    else:
        unet_history = {}
        best_dice = 0.0
        best_epoch = 0
    unet_results_summary = {"best_dice": best_dice, "best_epoch": best_epoch, "training_time_s": 0}
    unet_test_metrics = {}"""

    skip_unet_new = """if SKIP_UNET:
    print(f"[SKIP] U-Net++ checkpoint found — reusing {unet_best}")
    results_unet = get_unet_dirs(SCRIPT_DIR, CONFIG)["results"]
    metrics_path = results_unet / "metrics_summary.json"
    history_path = results_unet / "training_history.json"
    if metrics_path.is_file():
        with open(metrics_path, "r", encoding="utf-8") as f:
            unet_results_summary = json.load(f)
        best_dice = float(unet_results_summary.get("best_dice", 0.0))
        best_epoch = int(unet_results_summary.get("best_epoch", 0))
        unet_test_metrics = dict(unet_results_summary.get("test_metrics", {}))
    else:
        unet_results_summary = {"best_dice": 0.0, "best_epoch": 0, "training_time_s": 0}
        best_dice = 0.0
        best_epoch = 0
        unet_test_metrics = {}
    if history_path.is_file():
        with open(history_path, "r", encoding="utf-8") as f:
            unet_history = json.load(f)
    else:
        unet_history = {}"""

    src8 = "".join(nb["cells"][8]["source"])
    if skip_unet_old in src8:
        src8 = src8.replace(skip_unet_old, skip_unet_new)
        nb["cells"][8]["source"] = to_source(src8)

    # Cell 11: replace §5.5 inline plotting
    src11 = "".join(nb["cells"][11]["source"])
    marker = "# ============================================================================\n# 5.5 Training curves"
    if marker in src11:
        head = src11.split(marker)[0]
        src11_new = head + """# ============================================================================
# 5.5 Training curves dashboard (YOLO + U-Net++ + Infection)
# ============================================================================
# Panels (2x3):
#   [0] YOLO mAP curves   [1] YOLO losses
#   [2] U-Net++ losses    [3] U-Net++ val Dice/IoU (+ best epoch)
#   [4] Infection loss    [5] Infection train acc (+ held-out test reference line)
#
# Saved to: results/figures/training_curves_dashboard.png
# Re-run §4.4 first if infection history panels are empty.

from train_model import display_training_curves

print("\\nRendering training curves dashboard ...")
training_curves_path = display_training_curves(SCRIPT_DIR, CONFIG)
print(f"Dashboard saved: {training_curves_path}")
"""
        nb["cells"][11]["source"] = to_source(src11_new)

    # Cell 12: expand markdown
    nb["cells"][12]["source"] = to_source(
        """## 6: Qualitative Analysis and Summary

### 6.1 Prediction overlay elements

Each combined prediction image shows:

- **Green bbox** — YOLO wound detection
- **Green mask overlay** — U-Net++ refined segmentation
- **Info panel** — wound area (cm² / px), marker scale, segmentation Dice/IoU vs COCO reference mask
- **Metadata label vs Prediction** — filename-derived infection metadata (`-not-`) compared to classifier output (TP/TN/FP/FN)

Training curves for all stages are in **§5.5** (not repeated here).

### 6.2 Experiment gallery (4 images, 2×2)

`display_experiment_gallery(n_total=4)` shows one reproducible case per confusion-matrix cell:

| Panel | Meaning |
|-------|---------|
| TP | Metadata infected, predicted infected |
| TN | Metadata not infected, predicted not infected |
| FP | Metadata not infected, predicted infected |
| FN | Metadata infected, predicted not infected |

Selection is deterministic: alphabetically first valid test image per category.
Saved to `results/figures/experiment_gallery_4panel.png`.

### 6.3 Metrics tables

- Full held-out infection metrics from `results/infection/metrics_summary.json`
- Per-image table for the four selected gallery cases
"""
    )

    # Cell 13: replace with gallery cell
    nb["cells"][13]["source"] = to_source(
        """%matplotlib inline

import sys
import json
import importlib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ============================================================================
# Bootstrap — standalone without §1–§5
# ============================================================================

if "SCRIPT_DIR" not in globals():
    SCRIPT_DIR = Path.cwd().resolve()
if not (SCRIPT_DIR / "config.yaml").is_file() or not (SCRIPT_DIR / "train_model.py").is_file():
    raise RuntimeError(
        "Kernel cwd must be experiments/YOLO11m_UNetPP "
        f"(config.yaml + train_model.py not found in {SCRIPT_DIR})"
    )
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if "CONFIG" not in globals():
    from pipeline_utils import load_config, set_seed, get_device
    CONFIG = load_config(SCRIPT_DIR / "config.yaml", validate_combined=True)
    set_seed(CONFIG.get("seed", 42))
    device = get_device()

import train_model as _train_model
importlib.reload(_train_model)
from train_model import display_experiment_gallery
from experiment_io import get_combined_dirs

# ============================================================================
# 6.2 Experiment gallery — TP / TN / FP / FN (2×2)
# ============================================================================

print("=" * 60)
print("6.2  EXPERIMENT GALLERY — 4 diagnostic cases (2×2)")
print("=" * 60)
print(
    "Uses saved combined PNGs for TP/TN when available; "
    "regenerates FP/FN live against latest checkpoints."
)

gallery_path = display_experiment_gallery(SCRIPT_DIR, CONFIG, n_total=4)
print(f"\\nGallery figure: {gallery_path}")

# ============================================================================
# 6.3 Per-image metrics from wound_areas.json (full test set summary)
# ============================================================================

comb_dirs = get_combined_dirs(SCRIPT_DIR, CONFIG)
wound_areas_path = comb_dirs["results"] / "wound_areas.json"
if wound_areas_path.is_file():
    with open(wound_areas_path, "r", encoding="utf-8") as f:
        wound_areas = json.load(f)
    df_all = pd.DataFrame(wound_areas)
    if not df_all.empty and "prediction_outcome" in df_all.columns:
        print("\\n── Test-set infection confusion counts (metadata vs prediction) ──")
        print(df_all["prediction_outcome"].value_counts(dropna=False).to_string())
    if {"dice", "iou"}.issubset(df_all.columns):
        print(
            f"\\nSegmentation on test set: "
            f"mean Dice={df_all['dice'].mean():.4f}, mean IoU={df_all['iou'].mean():.4f}"
        )
else:
    print("[WARNING] wound_areas.json not found — run §5 combined evaluation first.")
"""
    )
    nb["cells"][13]["outputs"] = []
    nb["cells"][13]["execution_count"] = None

    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Notebook updated successfully.")


if __name__ == "__main__":
    main()
