"""Execute training cells in training_pipeline.ipynb and persist outputs in-place.

Runs §1 (setup), §4.1–4.3 (YOLO/U-Net), and §4.4 (infection classifier) so that
printed metrics and logs are saved inside the notebook file.

Usage (from experiments/YOLO11m_UNetPP):
    python scripts/run_notebook_training.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError

SCRIPT_DIR = Path(__file__).resolve().parent.parent
NB_PATH = SCRIPT_DIR / "training_pipeline.ipynb"
TRAINING_CELLS = (2, 8, 9)  # §1 setup, §4 training, §4.4 infection


def main() -> int:
    if not NB_PATH.is_file():
        print(f"[ERROR] Notebook not found: {NB_PATH}")
        return 1

    with open(NB_PATH, encoding="utf-8") as f:
        full_nb = nbformat.read(f, as_version=4)

    mini_nb = nbformat.v4.new_notebook(metadata=copy.deepcopy(getattr(full_nb, "metadata", {})))
    for idx in TRAINING_CELLS:
        mini_nb.cells.append(copy.deepcopy(full_nb.cells[idx]))

    ep = ExecutePreprocessor(timeout=-1, kernel_name="venv_cuda")
    print(f"Notebook: {NB_PATH}")
    print(f"Working dir: {SCRIPT_DIR}")
    print(f"Cells to run: {TRAINING_CELLS}")
    print("=" * 60)

    try:
        ep.preprocess(mini_nb, {"metadata": {"path": str(SCRIPT_DIR)}})
    except CellExecutionError as exc:
        err_text = str(exc).encode("utf-8", errors="replace").decode("utf-8")
        print(f"\n[ERROR] Cell execution failed:\n{err_text}")
        for i, idx in enumerate(TRAINING_CELLS):
            if i < len(mini_nb.cells):
                full_nb.cells[idx].outputs = mini_nb.cells[i].outputs
                full_nb.cells[idx].execution_count = mini_nb.cells[i].execution_count
        with open(NB_PATH, "w", encoding="utf-8") as f:
            nbformat.write(full_nb, f)
        print(f"Partial outputs saved to: {NB_PATH}")
        return 1

    for i, idx in enumerate(TRAINING_CELLS):
        full_nb.cells[idx].outputs = mini_nb.cells[i].outputs
        full_nb.cells[idx].execution_count = mini_nb.cells[i].execution_count

    with open(NB_PATH, "w", encoding="utf-8") as f:
        nbformat.write(full_nb, f)

    print("\n" + "=" * 60)
    print(f"Notebook saved with outputs: {NB_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
