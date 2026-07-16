"""Clear stale notebook outputs and embed final locked paper artifacts."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import nbformat
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = SCRIPT_DIR / "training_pipeline.ipynb"


def _stream(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_output("stream", name="stdout", text=text)


def _image_output(path: Path) -> nbformat.NotebookNode:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return nbformat.v4.new_output(
        "display_data",
        data={"image/png": payload, "text/plain": f"<Image: {path.name}>"},
        metadata={},
    )


def _find_cell(notebook: nbformat.NotebookNode, marker: str) -> nbformat.NotebookNode:
    for cell in notebook.cells:
        if marker in cell.get("source", ""):
            return cell
    raise KeyError(f"Notebook cell marker not found: {marker}")


def main() -> int:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    config = yaml.safe_load((SCRIPT_DIR / "config.yaml").read_text(encoding="utf-8"))
    metrics = json.loads(
        (SCRIPT_DIR / "results" / "metrics_summary.json").read_text(encoding="utf-8")
    )
    combined = metrics["combined"]
    infection = metrics["infection"]
    figures = SCRIPT_DIR / "results" / "figures"
    curves = figures / "training_curves_dashboard.png"
    gallery = figures / "experiment_gallery_4panel.png"

    config_cell = _find_cell(notebook, "CONFIG = load_config(SCRIPT_DIR")
    config_cell.outputs = [
        _stream(
            "Locked experiment loaded\n"
            f"  experiment: {config['experiment_name']}\n"
            f"  run mode: {config['run']['mode']}\n"
            f"  grouped annotations: {config['ann_train']}, "
            f"{config['ann_val']}, {config['ann_test']}\n"
            f"  YOLO image size: {config['yolo']['image_size']}\n"
        )
    ]

    infection_cell = _find_cell(notebook, "INFECTION METADATA CLASSIFIER")
    infection_cell.outputs = [
        _stream(
            "Final locked infection metadata result\n"
            f"  seed/epoch/threshold: {infection['canonical_seed']}/"
            f"{infection['best_epoch']}/{infection['decision_threshold']:.2f}\n"
            f"  test accuracy: {infection['test_accuracy']:.4f}\n"
            f"  test sensitivity/specificity: {infection['test_recall']:.4f}/"
            f"{infection['test_specificity']:.4f}\n"
            f"  test F1: {infection['test_f1_score']:.4f}\n"
            "  Labels are filename metadata, not clinical diagnoses.\n"
        )
    ]

    visual_cell = _find_cell(notebook, "VISUAL SUMMARY — curves + experiment gallery")
    visual_cell.outputs = [
        _stream(
            "Final locked visual artifacts\n"
            f"  mean Dice (95% CI): {combined['mean_dice']:.4f} "
            f"({combined['mean_dice_ci95']['lower']:.4f}–"
            f"{combined['mean_dice_ci95']['upper']:.4f})\n"
            f"  marker detection: {combined['n_marker_detected']}/"
            f"{combined['n_images_total']} "
            f"({combined['marker_detection_rate']:.1%})\n"
        ),
        _image_output(curves),
        _image_output(gallery),
    ]

    quantitative_cell = _find_cell(notebook, "# 5.2 Save reports")
    quantitative_cell.outputs = [
        _stream(
            "Final locked test summary\n"
            f"  YOLO bbox/segm AP50: {metrics['yolo']['bbox_mAP50']:.4f}/"
            f"{metrics['yolo']['segm_mAP50']:.4f}\n"
            f"  Combined bbox/segm AP50: {combined['coco_bbox_AP50']:.4f}/"
            f"{combined['coco_segm_AP50']:.4f}\n"
            f"  Combined Dice/IoU: {combined['mean_dice']:.4f}/"
            f"{combined['mean_iou']:.4f}\n"
        ),
        _image_output(curves),
    ]

    gallery_cells = [
        cell for cell in notebook.cells
        if "display_experiment_gallery(" in cell.get("source", "")
    ]
    gallery_cell = gallery_cells[-1] if gallery_cells else visual_cell
    if gallery_cell is not visual_cell:
        gallery_cell.outputs = [
            _stream(
                "Four deterministic diagnostic cases (TP/TN/FP/FN); "
                "not a statistical sample.\n"
            ),
            _image_output(gallery),
        ]

    nbformat.write(notebook, NOTEBOOK_PATH)
    print(f"Refreshed notebook outputs: {NOTEBOOK_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
