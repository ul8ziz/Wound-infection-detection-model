"""
Build Wound Focus Clean Dataset
==============================

Safe, non-destructive image inventory and renaming pipeline for the raw dataset.
Scans data/task_*/data/, uses manifest.jsonl for source identity and infection status,
produces mappings and reports. Does NOT modify raw data.

Usage:
    cd scripts
    python build_wound_focus_dataset.py --data-root ../data --output-dir ../data/wound_focus_clean
    python build_wound_focus_dataset.py --copy   # After validating mapping, copy images

Output:
    data/wound_focus_clean/
    ├── images/           # Copied images (with --copy)
    ├── mappings/
    │   ├── image_mapping.csv
    │   ├── image_mapping.json
    │   ├── skipped_images.csv
    │   └── ambiguous_cases.csv
    └── reports/
        └── RENAMING_REPORT.md
"""

import argparse
import csv
import json
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Resolve project root
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# Image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Clinical patterns for determinate infection status (MK/МК + digits, -day-, -hosp-, -inf-)
CLINICAL_PATTERN = re.compile(
    r"(-not-|-inf-|-day-|-hosp-|mk\d|мк\d|MK\d|МК\d)",
    re.IGNORECASE
)
# Generic/placeholder patterns -> ambiguous
GENERIC_PATTERNS = [
    re.compile(r"^whatsapp\s+image", re.IGNORECASE),
    re.compile(r"^img_\d", re.IGNORECASE),
    re.compile(r"^\+\d{10,}"),  # Phone number prefix
    re.compile(r"^\d{10,}$"),   # Long timestamp (purely numeric)
    re.compile(r"^\d+$"),       # Short numeric (1, 2, 3)
]


def _is_ambiguous_name(name: str) -> Tuple[bool, Optional[str]]:
    """
    Check if manifest name is ambiguous for infection status.
    Returns (is_ambiguous, reason).
    """
    if not name or not name.strip():
        return True, "empty_name"
    name_lower = name.strip().lower()
    # Purely numeric
    if name_lower.isdigit():
        return True, "purely_numeric"
    # Generic patterns
    for pat in GENERIC_PATTERNS:
        if pat.search(name_lower):
            return True, "generic_pattern"
    # No clinical markers at all
    if not CLINICAL_PATTERN.search(name):
        return True, "no_clinical_markers"
    return False, None


def _infer_infection_status(name: str) -> Tuple[Optional[int], str]:
    """
    Infer infection status from manifest name.
    Returns (status, label) where status is 0/1 or None for ambiguous.
    """
    is_amb, reason = _is_ambiguous_name(name)
    if is_amb:
        return None, "ambiguous"
    if "-not-" in name.lower():
        return 0, "not_infected"
    return 1, "infected"


def _load_manifest_images(manifest_path: Path, data_dir: Path, data_root: Path) -> List[Dict[str, Any]]:
    """Load image entries from manifest.jsonl."""
    if not manifest_path.exists():
        return []
    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or '"type"' in line or '"version"' in line:
                continue
            try:
                obj = json.loads(line)
                name = obj.get("name", "")
                ext = obj.get("extension", ".jpg")
                if not ext.startswith("."):
                    ext = "." + ext
                w = int(obj.get("width", 0))
                h = int(obj.get("height", 0))
                if not name or w <= 0 or h <= 0:
                    continue
                rel_path = data_dir.relative_to(data_root) / (name + ext)
                entries.append({
                    "name": name,
                    "extension": ext,
                    "width": w,
                    "height": h,
                    "rel_path": str(rel_path).replace("\\", "/"),
                    "filename": name + ext,
                })
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return entries


def _load_annotations_file_names(data_root: Path) -> set:
    """Load set of file_name paths from annotations_cleaned.json if present."""
    # Check original_data/ first (new structure), then data root (backward compat)
    ann_path = data_root / "original_data" / "annotations_cleaned.json"
    if not ann_path.exists():
        ann_path = data_root / "annotations_cleaned.json"
    if not ann_path.exists():
        return set()
    try:
        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        images = data.get("images", [])
        return {img.get("file_name", "") for img in images if img.get("file_name")}
    except (json.JSONDecodeError, KeyError, IOError):
        return set()


def run_inventory(data_root: Path) -> Tuple[List[Dict], List[Dict], List[Dict], set]:
    """
    Stage 1: Scan all tasks, build per-image records.
    Returns (valid_records, skipped_records, ambiguous_records, annotated_paths).
    """
    annotated_paths = _load_annotations_file_names(data_root)
    # Support new structure: data/original_data/task_* or legacy: data/task_*
    raw_root = data_root / "original_data"
    if raw_root.exists():
        task_folders = sorted(
            [f for f in raw_root.iterdir() if f.is_dir() and f.name.startswith("task_")]
        )
    else:
        task_folders = sorted(
            [f for f in data_root.iterdir() if f.is_dir() and f.name.startswith("task_")]
        )
        raw_root = data_root
    valid: List[Dict] = []
    skipped: List[Dict] = []
    ambiguous: List[Dict] = []

    for task_folder in task_folders:
        m = re.match(r"task_(\d+)$", task_folder.name)
        if not m:
            continue
        task_id = int(m.group(1))
        data_dir = task_folder / "data"
        manifest_path = data_dir / "manifest.jsonl"

        if not manifest_path.exists():
            skipped.append({
                "task_id": task_id,
                "original_local_path": str(data_dir.relative_to(data_root)).replace("\\", "/"),
                "skip_reason": "missing_manifest",
                "notes": f"Task {task_id} has no manifest.jsonl",
            })
            continue

        entries = _load_manifest_images(manifest_path, data_dir, data_root)
        if not entries:
            continue

        for ent in entries:
            rel_path = ent["rel_path"]
            full_path = data_root / rel_path
            orig_filename = ent["filename"]
            source_name = ent["name"]
            manifest_path_str = str(manifest_path.relative_to(data_root)).replace("\\", "/")

            # Check file exists
            if not full_path.exists():
                skipped.append({
                    "task_id": task_id,
                    "original_local_path": rel_path,
                    "original_local_filename": orig_filename,
                    "manifest_path": manifest_path_str,
                    "source_name_from_manifest": source_name,
                    "skip_reason": "missing_file",
                    "annotation_available": rel_path in annotated_paths,
                    "status": "skipped",
                    "notes": "Image file not found",
                })
                continue

            # Non-image check
            ext = Path(orig_filename).suffix.lower()
            if ext not in IMAGE_EXTENSIONS:
                skipped.append({
                    "task_id": task_id,
                    "original_local_path": rel_path,
                    "original_local_filename": orig_filename,
                    "manifest_path": manifest_path_str,
                    "source_name_from_manifest": source_name,
                    "skip_reason": "non_image",
                    "annotation_available": rel_path in annotated_paths,
                    "status": "skipped",
                    "notes": f"Extension {ext} not in {IMAGE_EXTENSIONS}",
                })
                continue

            inf_status, inf_label = _infer_infection_status(source_name)
            base_record = {
                "task_id": task_id,
                "original_local_path": rel_path,
                "original_local_filename": orig_filename,
                "manifest_path": manifest_path_str,
                "source_name_from_manifest": source_name,
                "annotation_available": rel_path in annotated_paths,
            }

            if inf_status is None:
                is_amb, amb_reason = _is_ambiguous_name(source_name)
                ambiguous.append({
                    **base_record,
                    "infection_status": "ambiguous",
                    "ambiguity_reason": amb_reason or "unknown",
                    "status": "ambiguous",
                    "notes": f"Could not infer infection status: {amb_reason}",
                })
            else:
                valid.append({
                    **base_record,
                    "infection_status": inf_status,
                    "infection_label": inf_label,
                    "status": "ok",
                    "notes": "",
                })

    return valid, skipped, ambiguous, annotated_paths


def run_mapping(
    valid: List[Dict],
    skipped: List[Dict],
    ambiguous: List[Dict],
    output_dir: Path,
) -> Tuple[List[Dict], List[str]]:
    """
    Stage 2: Assign global_id, generate new filenames, write CSV/JSON.
    Returns (mapped_valid_with_new_names, new_filenames_list).
    """
    mappings_dir = output_dir / "mappings"
    mappings_dir.mkdir(parents=True, exist_ok=True)

    global_id = 1
    mapped_valid: List[Dict] = []
    new_filenames: List[str] = []

    for rec in valid:
        task_id = rec["task_id"]
        inf_label = rec["infection_label"]
        new_name = f"task_{task_id:03d}_img_{global_id:06d}_{inf_label}.jpg"
        new_filenames.append(new_name)
        mapped_rec = {
            **rec,
            "global_id": global_id,
            "new_filename": new_name,
        }
        mapped_valid.append(mapped_rec)
        global_id += 1

    # Schema fields for CSV
    base_fields = [
        "global_id", "task_id", "original_local_path", "original_local_filename",
        "manifest_path", "source_name_from_manifest", "new_filename",
        "infection_status", "annotation_available", "status", "notes",
    ]
    skip_extra = ["skip_reason"]
    amb_extra = ["ambiguity_reason"]

    # image_mapping.csv (valid only)
    mapping_path = mappings_dir / "image_mapping.csv"
    with open(mapping_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=base_fields, extrasaction="ignore")
        writer.writeheader()
        for r in mapped_valid:
            row = {k: r.get(k, "") for k in base_fields}
            row["infection_status"] = r.get("infection_status", "")
            writer.writerow(row)

    # image_mapping.json (full structure)
    mapping_json = {
        "valid": mapped_valid,
        "skipped": skipped,
        "ambiguous": ambiguous,
        "summary": {
            "total_valid": len(mapped_valid),
            "total_skipped": len(skipped),
            "total_ambiguous": len(ambiguous),
        },
    }
    with open(mappings_dir / "image_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping_json, f, indent=2, ensure_ascii=False)

    # skipped_images.csv
    skip_fields = base_fields + skip_extra
    with open(mappings_dir / "skipped_images.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=skip_fields, extrasaction="ignore")
        writer.writeheader()
        for r in skipped:
            row = {k: r.get(k, "") for k in skip_fields}
            writer.writerow(row)

    # ambiguous_cases.csv
    amb_fields = base_fields + amb_extra
    with open(mappings_dir / "ambiguous_cases.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=amb_fields, extrasaction="ignore")
        writer.writeheader()
        for r in ambiguous:
            row = {k: r.get(k, "") for k in amb_fields}
            row["global_id"] = ""
            row["new_filename"] = ""
            row["infection_status"] = "ambiguous"
            writer.writerow(row)

    return mapped_valid, new_filenames


def run_validation(
    mapped_valid: List[Dict],
    new_filenames: List[str],
    skipped: List[Dict],
    ambiguous: List[Dict],
    data_root: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Stage 3: Run validation checks, return report dict.
    """
    errors: List[str] = []
    # Duplicate new filenames
    seen = set()
    for n in new_filenames:
        if n in seen:
            errors.append(f"Duplicate new filename: {n}")
        seen.add(n)

    # Missing image file (re-check)
    for r in mapped_valid:
        p = data_root / r["original_local_path"]
        if not p.exists():
            errors.append(f"Valid record references missing file: {r['original_local_path']}")

    if errors:
        return {"valid": False, "errors": errors}

    tasks_with_images = len({r["task_id"] for r in mapped_valid})
    tasks_with_multiple = sum(
        1 for tid in {r["task_id"] for r in mapped_valid}
        if sum(1 for r in mapped_valid if r["task_id"] == tid) > 1
    )

    report = {
        "valid": True,
        "total_valid": len(mapped_valid),
        "total_skipped": len(skipped),
        "total_ambiguous": len(ambiguous),
        "tasks_with_images": tasks_with_images,
        "tasks_with_multiple_images": tasks_with_multiple,
        "errors": [],
    }
    return report


def write_report(
    report: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Write RENAMING_REPORT.md."""
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "RENAMING_REPORT.md"

    lines = [
        "# Wound Focus Clean Dataset - Renaming Report",
        "",
        "## How Filenames Are Derived",
        "",
        "New filenames follow: `task_{task_id:03d}_img_{global_id:06d}_{infection_label}.jpg`",
        "- `task_id`: from folder name (e.g. task_14 → 014)",
        "- `global_id`: sequential 1-based index across valid images",
        "- `infection_label`: `not_infected` or `infected` (only for determinate cases)",
        "",
        "## How Infection Status Is Inferred",
        "",
        "- **not_infected (0)**: manifest `name` contains `-not-`",
        "- **infected (1)**: manifest `name` has clinical pattern (MK/МК, -day-, -inf-, -hosp-) and no `-not-`",
        "- **ambiguous**: name empty, purely numeric, generic (WhatsApp, IMG_, etc.), or no clinical markers",
        "",
        "## Assumptions",
        "",
        "- Raw dataset is immutable; only copies are created in wound_focus_clean/images/",
        "- Manifest `name` is the original source filename (authoritative for infection)",
        "- Extension normalized to .jpg on copy",
        "- annotation_available = whether image appears in annotations_cleaned.json",
        "",
        "## Summary",
        "",
        f"- **Valid (mapped)**: {report.get('total_valid', 0)}",
        f"- **Skipped**: {report.get('total_skipped', 0)}",
        f"- **Ambiguous**: {report.get('total_ambiguous', 0)}",
        f"- **Tasks with images**: {report.get('tasks_with_images', 0)}",
        f"- **Tasks with multiple images**: {report.get('tasks_with_multiple_images', 0)}",
        "",
    ]
    if report.get("errors"):
        lines.extend(["## Validation Errors", ""] + [f"- {e}" for e in report["errors"]] + [""])
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logging.info("Report written to %s", report_path)


def run_copy(
    mapped_valid: List[Dict],
    data_root: Path,
    output_dir: Path,
) -> Tuple[int, List[str]]:
    """
    Stage 4: Copy images to wound_focus_clean/images/ with new filenames.
    Returns (copied_count, errors).
    """
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    errors: List[str] = []
    copied = 0
    for r in mapped_valid:
        src = data_root / r["original_local_path"]
        dst = images_dir / r["new_filename"]
        if not src.exists():
            errors.append(f"Source missing: {r['original_local_path']}")
            continue
        try:
            shutil.copy2(src, dst)
            copied += 1
        except OSError as e:
            errors.append(f"Copy failed {src} -> {dst}: {e}")
    return copied, errors


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Build wound focus clean dataset (safe renaming pipeline)"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(PROJECT_ROOT / "data"),
        help="Data root containing task_* folders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "wound_focus_clean"),
        help="Output directory for wound_focus_clean",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy images to output/images/ after mapping (run only after validating)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not data_root.exists():
        logging.error("Data root not found: %s", data_root)
        return 1

    # Stage 1: Inventory
    logging.info("Stage 1: Inventory...")
    valid, skipped, ambiguous, _ = run_inventory(data_root)
    logging.info("  Valid: %d, Skipped: %d, Ambiguous: %d", len(valid), len(skipped), len(ambiguous))

    # Stage 2: Mapping
    logging.info("Stage 2: Mapping...")
    mapped_valid, new_filenames = run_mapping(valid, skipped, ambiguous, output_dir)
    logging.info("  Mappings written to %s", output_dir / "mappings")

    # Stage 3: Validation
    logging.info("Stage 3: Validation...")
    report = run_validation(
        mapped_valid, new_filenames, skipped, ambiguous, data_root, output_dir
    )
    write_report(report, output_dir)

    if not report.get("valid", True):
        for e in report.get("errors", []):
            logging.error("Validation error: %s", e)
        return 1

    # Stage 4: Copy (optional)
    if args.copy:
        logging.info("Stage 4: Copying images...")
        copied, copy_errors = run_copy(mapped_valid, data_root, output_dir)
        if copy_errors:
            for e in copy_errors:
                logging.error("Copy error: %s", e)
        logging.info("  Copied %d images to %s", copied, output_dir / "images")

    logging.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
