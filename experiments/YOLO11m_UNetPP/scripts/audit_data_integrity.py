"""Audit split leakage, augmentation lineage, and annotation schema consistency."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCRIPT_DIR.parent.parent
AUGMENTED_SUFFIX = re.compile(r"_aug\d+(?=\.[^.]+$)", re.IGNORECASE)
TASK_PATTERN = re.compile(r"(task_\d+)", re.IGNORECASE)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalized_name(file_name: str) -> str:
    return str(file_name).replace("\\", "/")


def _source_key(file_name: str) -> str:
    return AUGMENTED_SUFFIX.sub("", Path(_normalized_name(file_name)).name)


def _task_key(file_name: str) -> str | None:
    match = TASK_PATTERN.search(_normalized_name(file_name))
    return match.group(1).lower() if match else None


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_summary(
    split: str,
    annotation_path: Path,
    image_root: Path,
) -> tuple[dict[str, Any], set[str], set[str], dict[str, list[str]]]:
    coco = _load_json(annotation_path)
    names = [_normalized_name(image["file_name"]) for image in coco.get("images", [])]
    source_keys = {_source_key(name) for name in names}
    task_keys = {task for name in names if (task := _task_key(name))}
    category_counts = Counter(
        str(category.get("name", category.get("id")))
        for category in coco.get("categories", [])
    )
    annotation_category_counts = Counter(
        int(annotation.get("category_id", -1))
        for annotation in coco.get("annotations", [])
    )

    original_hashes: dict[str, list[str]] = {}
    missing = []
    for name in names:
        if AUGMENTED_SUFFIX.search(Path(name).name):
            continue
        path = image_root / name
        digest = _sha256(path)
        if digest is None:
            missing.append(name)
        else:
            original_hashes.setdefault(digest, []).append(name)

    summary = {
        "split": split,
        "annotation_path": str(annotation_path),
        "image_root": str(image_root),
        "n_images": len(names),
        "n_unique_sources": len(source_keys),
        "n_tasks": len(task_keys),
        "categories": sorted(category_counts),
        "annotation_count_by_category_id": dict(annotation_category_counts),
        "n_original_images_hashed": sum(len(v) for v in original_hashes.values()),
        "n_missing_images": len(missing),
        "missing_images": missing[:20],
    }
    return summary, source_keys, task_keys, original_hashes


def run_audit(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_root = (PROJECT_ROOT / config["data_root"]).resolve()
    train_root = (
        PROJECT_ROOT / config.get("data_root_train", config["data_root"])
    ).resolve()

    summaries = {}
    sources = {}
    tasks = {}
    hashes = {}
    for split in ("train", "val", "test"):
        ann_path = (PROJECT_ROOT / config[f"ann_{split}"]).resolve()
        image_root = train_root if split == "train" else data_root
        summary, source_keys, task_keys, original_hashes = _split_summary(
            split, ann_path, image_root
        )
        summaries[split] = summary
        sources[split] = source_keys
        tasks[split] = task_keys
        hashes[split] = original_hashes

    pairwise = {}
    critical = []
    warnings = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        source_overlap = sorted(sources[left] & sources[right])
        task_overlap = sorted(tasks[left] & tasks[right])
        hash_overlap = sorted(set(hashes[left]) & set(hashes[right]))
        key = f"{left}_vs_{right}"
        pairwise[key] = {
            "source_overlap_count": len(source_overlap),
            "source_overlap_examples": source_overlap[:20],
            "task_overlap_count": len(task_overlap),
            "task_overlap_examples": task_overlap[:20],
            "content_hash_overlap_count": len(hash_overlap),
        }
        if source_overlap or hash_overlap:
            critical.append(
                f"{key}: exact source/content leakage detected "
                f"(sources={len(source_overlap)}, hashes={len(hash_overlap)})"
            )
        if task_overlap:
            warnings.append(
                f"{key}: {len(task_overlap)} CVAT task groups span both splits; "
                "patient independence is not established."
            )

    category_sets = {
        split: tuple(summary["categories"]) for split, summary in summaries.items()
    }
    if len(set(category_sets.values())) > 1:
        critical.append(
            "Annotation categories differ across splits: "
            + ", ".join(f"{k}={v}" for k, v in category_sets.items())
        )

    status = "fail" if critical else ("warning" if warnings else "pass")
    report = {
        "status": status,
        "config_path": str(config_path),
        "splits": summaries,
        "pairwise": pairwise,
        "critical_issues": critical,
        "warnings": warnings,
        "patient_grouping": {
            "available": False,
            "note": (
                "No verified patient identifier is present in COCO metadata. "
                "CVAT task is audited as a conservative source-group proxy."
            ),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_integrity.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# Data Integrity Audit",
        "",
        f"**Status:** {status.upper()}",
        "",
        "## Split summary",
        "",
        "| Split | Images | Original sources | Tasks | Categories |",
        "|---|---:|---:|---:|---|",
    ]
    for split, summary in summaries.items():
        lines.append(
            f"| {split} | {summary['n_images']} | {summary['n_unique_sources']} | "
            f"{summary['n_tasks']} | {', '.join(summary['categories'])} |"
        )
    lines.extend(["", "## Critical issues", ""])
    lines.extend(f"- {item}" for item in critical or ["None"])
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {item}" for item in warnings or ["None"])
    (output_dir / "data_integrity.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=SCRIPT_DIR / "config.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "results" / "audits",
    )
    args = parser.parse_args()
    report = run_audit(args.config.resolve(), args.output_dir.resolve())
    print(f"Data integrity status: {report['status'].upper()}")
    for issue in report["critical_issues"]:
        print(f"[CRITICAL] {issue}")
    for warning in report["warnings"]:
        print(f"[WARNING] {warning}")
    return 1 if report["critical_issues"] else 0


if __name__ == "__main__":
    sys.exit(main())
