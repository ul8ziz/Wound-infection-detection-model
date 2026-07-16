"""Evaluate a specified YOLO checkpoint and save a traceable audit record."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from experiment_provenance import sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=SCRIPT_DIR / "yolo_data" / "dataset.yaml",
    )
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    from ultralytics import YOLO

    checkpoint = args.checkpoint.resolve()
    output_dir = SCRIPT_DIR / "results" / "checkpoint_audit" / args.label
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(checkpoint))
    result = model.val(
        data=str(args.data.resolve()),
        split="test",
        imgsz=args.image_size,
        batch=args.batch_size,
        workers=2,
        project=str(output_dir),
        name="ultralytics",
        exist_ok=True,
        verbose=False,
    )
    metrics = {
        "label": args.label,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "data": str(args.data.resolve()),
        "image_size": args.image_size,
        "bbox_mAP50": float(result.box.map50),
        "bbox_mAP50_95": float(result.box.map),
        "segm_mAP50": float(result.seg.map50),
        "segm_mAP50_95": float(result.seg.map),
        "note": (
            "Checkpoint gate only. This result is not paper-citable unless the "
            "data-integrity audit for the evaluated split passes."
        ),
    }
    destination = output_dir / "metrics.json"
    destination.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
