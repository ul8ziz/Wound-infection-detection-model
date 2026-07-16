"""Build deterministic task-grouped wound+marker splits for the paper run."""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_SOURCE = (
    PROJECT_ROOT / "data" / "wound_focus_clean" / "annotations_wound_marker.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "wound_focus_clean" / "paper_split"
TASK_PATTERN = re.compile(r"(task_\d+)", re.IGNORECASE)
SPLITS = ("train", "val", "test")
TARGET_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def _task_key(file_name: str) -> str:
    match = TASK_PATTERN.search(str(file_name).replace("\\", "/"))
    if not match:
        raise ValueError(f"Cannot derive CVAT task group from {file_name!r}")
    return match.group(1).lower()


def _infection_label(file_name: str) -> int:
    name = str(file_name).lower()
    return 0 if ("-not-" in name or "not_infected" in name) else 1


def _assignment_score(
    split: str,
    group_images: list[dict[str, Any]],
    counts: dict[str, dict[str, int]],
    targets: dict[str, dict[str, float]],
) -> float:
    added_total = len(group_images)
    added_positive = sum(_infection_label(img["file_name"]) for img in group_images)
    score = 0.0
    for metric, added in (("total", added_total), ("positive", added_positive)):
        target = max(targets[split][metric], 1.0)
        projected = counts[split][metric] + added
        score += ((projected - target) / target) ** 2
    # Strongly discourage assigning a group after a split is already over capacity.
    if counts[split]["total"] >= targets[split]["total"]:
        score += 2.0
    return score


def build_splits(
    source_path: Path,
    output_dir: Path,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    coco = json.loads(source_path.read_text(encoding="utf-8"))
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for image in coco.get("images", []):
        groups[_task_key(image["file_name"])].append(image)

    total_images = len(coco.get("images", []))
    total_positive = sum(
        _infection_label(image["file_name"]) for image in coco.get("images", [])
    )
    targets = {
        split: {
            "total": total_images * ratio,
            "positive": total_positive * ratio,
        }
        for split, ratio in TARGET_RATIOS.items()
    }
    counts = {
        split: {"total": 0, "positive": 0}
        for split in SPLITS
    }
    assignments: dict[str, str] = {}

    rng = random.Random(seed)
    group_items = list(groups.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    for task, images in group_items:
        best_split = min(
            SPLITS,
            key=lambda split: (
                _assignment_score(split, images, counts, targets),
                SPLITS.index(split),
            ),
        )
        assignments[task] = best_split
        counts[best_split]["total"] += len(images)
        counts[best_split]["positive"] += sum(
            _infection_label(image["file_name"]) for image in images
        )

    image_split = {
        image["id"]: assignments[_task_key(image["file_name"])]
        for image in coco.get("images", [])
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    split_reports = {}

    for split in SPLITS:
        images = [
            image for image in coco.get("images", [])
            if image_split[image["id"]] == split
        ]
        image_ids = {image["id"] for image in images}
        annotations = [
            annotation for annotation in coco.get("annotations", [])
            if annotation.get("image_id") in image_ids
        ]
        payload = {
            "images": images,
            "annotations": annotations,
            "categories": coco.get("categories", []),
            "info": {
                "split_strategy": "CVAT-task grouped deterministic split",
                "seed": seed,
                "source": str(source_path),
            },
        }
        destination = output_dir / f"{split}_wound_marker_grouped.json"
        destination.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        positives = sum(_infection_label(image["file_name"]) for image in images)
        split_reports[split] = {
            "path": str(destination),
            "n_images": len(images),
            "n_infected_metadata": positives,
            "n_not_infected_metadata": len(images) - positives,
            "n_tasks": len(
                {_task_key(image["file_name"]) for image in images}
            ),
            "n_annotations": len(annotations),
        }

    task_sets = {
        split: {task for task, assigned in assignments.items() if assigned == split}
        for split in SPLITS
    }
    assert not (task_sets["train"] & task_sets["val"])
    assert not (task_sets["train"] & task_sets["test"])
    assert not (task_sets["val"] & task_sets["test"])

    report = {
        "seed": seed,
        "strategy": "CVAT-task grouped deterministic greedy stratification",
        "source": str(source_path),
        "ratios": TARGET_RATIOS,
        "splits": split_reports,
        "task_assignment": assignments,
        "limitations": [
            "CVAT task is a source-group proxy; verified patient identifiers are unavailable.",
            "Filename infection labels are metadata proxies, not clinical diagnoses.",
        ],
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = build_splits(
        args.source.resolve(),
        args.output_dir.resolve(),
        seed=args.seed,
    )
    for split, summary in report["splits"].items():
        print(
            f"{split}: images={summary['n_images']}, "
            f"tasks={summary['n_tasks']}, infected={summary['n_infected_metadata']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
