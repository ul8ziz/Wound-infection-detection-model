"""Freeze one immutable experiment package for paper traceability."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from experiment_io import get_unet_best_checkpoint_path  # noqa: E402
from experiment_provenance import (  # noqa: E402
    build_experiment_manifest,
    save_experiment_manifest,
)


def _copy_if_exists(source: Path, destination: Path) -> None:
    if not source.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def freeze(name: str, *, copy_checkpoints: bool = True) -> Path:
    config_path = SCRIPT_DIR / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    archive_root = SCRIPT_DIR / "paper_artifacts" / name
    if archive_root.exists():
        raise FileExistsError(
            f"Immutable archive already exists: {archive_root}. Choose another name."
        )
    archive_root.mkdir(parents=True)

    yolo_checkpoint = SCRIPT_DIR / "checkpoints" / "yolo" / "best.pt"
    unet_checkpoint = get_unet_best_checkpoint_path(SCRIPT_DIR, config)
    infection_checkpoint = (
        SCRIPT_DIR / "checkpoints" / "infection" / "infection_classifier.pth"
    )
    checkpoint_paths = [yolo_checkpoint, unet_checkpoint, infection_checkpoint]

    manifest = build_experiment_manifest(
        config,
        SCRIPT_DIR,
        run_mode="evaluate_only",
        checkpoint_paths=checkpoint_paths,
    )
    manifest["archive_name"] = name
    manifest["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["immutable_intent"] = True
    save_experiment_manifest(manifest, archive_root)

    _copy_if_exists(config_path, archive_root / "config.yaml")
    artifact_files = [
        SCRIPT_DIR / "results" / "metrics_summary.json",
        SCRIPT_DIR / "results" / "yolo" / "test_metrics.json",
        SCRIPT_DIR / "results" / "infection" / "metrics_summary.json",
        SCRIPT_DIR / "results" / "infection" / "training_history.json",
        SCRIPT_DIR / "reports" / "training_report.md",
        SCRIPT_DIR / "results" / "figures" / "training_curves_dashboard.png",
        SCRIPT_DIR / "results" / "figures" / "experiment_gallery_4panel.png",
    ]
    for source in artifact_files:
        relative = source.relative_to(SCRIPT_DIR)
        _copy_if_exists(source, archive_root / relative)

    if copy_checkpoints:
        for checkpoint in checkpoint_paths:
            if checkpoint.is_file():
                destination = archive_root / "checkpoints" / checkpoint.name
                if "unet" in checkpoint.parts:
                    destination = archive_root / "checkpoints" / "unet_best_model.pth"
                elif "infection" in checkpoint.parts:
                    destination = (
                        archive_root / "checkpoints" / "infection_classifier.pth"
                    )
                elif "yolo" in checkpoint.parts:
                    destination = archive_root / "checkpoints" / "yolo_best.pt"
                _copy_if_exists(checkpoint, destination)

    inventory = {
        "archive": str(archive_root),
        "copied_checkpoints": copy_checkpoints,
        "files": sorted(
            str(path.relative_to(archive_root)).replace("\\", "/")
            for path in archive_root.rglob("*")
            if path.is_file()
        ),
    }
    (archive_root / "inventory.json").write_text(
        json.dumps(inventory, indent=2), encoding="utf-8"
    )
    return archive_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name",
        default=f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument(
        "--no-copy-checkpoints",
        action="store_true",
        help="Record hashes only instead of copying model binaries.",
    )
    args = parser.parse_args()
    archive = freeze(args.name, copy_checkpoints=not args.no_copy_checkpoints)
    print(f"Frozen experiment archive: {archive}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
