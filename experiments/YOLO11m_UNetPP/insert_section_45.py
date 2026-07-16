"""Insert §4.5 visual summary cell and update training completion message."""
import json
from pathlib import Path

NOTEBOOK = Path("training_pipeline.ipynb")

CELL_45_MD = """### 4.5 Visual summary — training curves & 4-panel gallery

**Run this cell after §4.1–4.4** (not only §4.1). Training alone does not render the new dashboards.

| Output | File |
|--------|------|
| 2×3 training curves (YOLO + U-Net++ + Infection) | `results/figures/training_curves_dashboard.png` |
| 2×2 experiment gallery (TP/TN/FP/FN) | `results/figures/experiment_gallery_4panel.png` |
"""

CELL_45_CODE = """%matplotlib inline

import importlib
import train_model as _tm
importlib.reload(_tm)

from train_model import (
    evaluate_combined,
    display_training_curves,
    display_experiment_gallery,
)

print("=" * 60)
print("4.5  VISUAL SUMMARY — curves + experiment gallery")
print("=" * 60)

print("\\n[1/3] Refreshing combined evaluation (Dice/IoU + infection fields in JSON/PNGs) ...")
evaluate_combined(CONFIG, SCRIPT_DIR)

print("\\n[2/3] Training curves dashboard (same as §5.5) ...")
curves_path = display_training_curves(SCRIPT_DIR, CONFIG)
print(f"Saved: {curves_path}")

print("\\n[3/3] Experiment gallery — TP / TN / FP / FN (same as §6.2) ...")
gallery_path = display_experiment_gallery(SCRIPT_DIR, CONFIG, n_total=4)
print(f"Saved: {gallery_path}")
"""


def to_source(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    return lines if lines else [text]


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    # Update §4 completion message
    src8 = "".join(nb["cells"][8]["source"])
    old_tail = 'print("\\nTraining complete. Continue with §4.4 (infection), §5 (metrics), and §6 (visuals).")'
    new_tail = (
        'print("\\nTraining complete.")\n'
        'print("  -> Run §4.4 (infection) if not done yet.")\n'
        'print("  -> Then run §4.5 below for training curves + 4-panel gallery (required for new visuals).")\n'
        'print("  -> §5–§6 repeat the same metrics/gallery with full evaluation tables.")'
    )
    if old_tail in src8:
        nb["cells"][8]["source"] = to_source(src8.replace(old_tail, new_tail))

    # Insert §4.5 after cell 9 (§4.4) if not already present
    already = any("4.5  VISUAL SUMMARY" in "".join(c.get("source", [])) for c in nb["cells"])
    if not already:
        md_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": to_source(CELL_45_MD),
        }
        code_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": to_source(CELL_45_CODE),
        }
        nb["cells"].insert(10, md_cell)
        nb["cells"].insert(11, code_cell)

    # Ensure §5 reloads train_model before display_training_curves
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        if "display_training_curves(SCRIPT_DIR, CONFIG)" in src and "importlib.reload" in src:
            if "from train_model import display_training_curves" in src:
                src = src.replace(
                    "from train_model import display_training_curves\n",
                    "from train_model import display_training_curves  # reloaded above\n",
                )
                nb["cells"][i]["source"] = to_source(src)

    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Inserted §4.5 and updated messages.")


if __name__ == "__main__":
    main()
