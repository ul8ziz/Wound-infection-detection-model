"""Reproducibility and checkpoint provenance helpers for paper experiments."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


PAPER_CONFIG_KEYS = ("seed", "experiment_name", "classes", "yolo", "unet", "combined")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str | None:
    """Return a file SHA-256 digest, or ``None`` when the file is absent."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    """Hash JSON-compatible data using canonical key ordering."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def paper_config_payload(config: dict[str, Any]) -> dict[str, Any]:
    """Return the scientific configuration subset used for compatibility checks."""
    return {key: config.get(key) for key in PAPER_CONFIG_KEYS}


def config_fingerprint(config: dict[str, Any]) -> str:
    """Return a stable fingerprint for paper-relevant configuration."""
    return stable_hash(paper_config_payload(config))


def annotation_manifest(path: Path) -> dict[str, Any]:
    """Summarize one COCO annotation file and fingerprint image membership."""
    if not path.is_file():
        return {"path": str(path), "exists": False}
    with path.open("r", encoding="utf-8") as stream:
        coco = json.load(stream)
    images = sorted(
        str(image.get("file_name", "")).replace("\\", "/")
        for image in coco.get("images", [])
    )
    categories = sorted(
        str(category.get("name", category.get("id", "")))
        for category in coco.get("categories", [])
    )
    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "n_images": len(images),
        "n_annotations": len(coco.get("annotations", [])),
        "categories": categories,
        "image_membership_sha256": stable_hash(images),
    }


def git_revision(project_root: Path) -> dict[str, Any]:
    """Return current Git revision and dirty-state metadata without mutating Git."""
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def build_experiment_manifest(
    config: dict[str, Any],
    script_dir: Path,
    *,
    run_mode: str,
    checkpoint_paths: Iterable[Path] = (),
) -> dict[str, Any]:
    """Build an immutable manifest for one training/evaluation invocation."""
    project_root = script_dir.parent.parent
    annotations = {}
    for split in ("train", "val", "test"):
        key = f"ann_{split}"
        value = config.get(key)
        if value:
            annotations[split] = annotation_manifest((project_root / value).resolve())

    checkpoints = []
    for path in checkpoint_paths:
        resolved = path.resolve()
        checkpoints.append(
            {
                "path": str(resolved),
                "exists": resolved.is_file(),
                "size_bytes": resolved.stat().st_size if resolved.is_file() else None,
                "sha256": sha256_file(resolved),
            }
        )

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_mode": run_mode,
        "config_fingerprint": config_fingerprint(config),
        "config": paper_config_payload(config),
        "annotations": annotations,
        "checkpoints": checkpoints,
        "git": git_revision(project_root),
    }


def save_experiment_manifest(manifest: dict[str, Any], output_dir: Path) -> Path:
    """Save manifest as JSON and a human-readable YAML copy."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "experiment_manifest.json"
    yaml_path = output_dir / "experiment_manifest.yaml"
    json_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    yaml_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return json_path


def load_manifest(path: Path) -> dict[str, Any] | None:
    """Load an experiment manifest when present."""
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_matches_config(
    manifest_path: Path,
    config: dict[str, Any],
) -> tuple[bool, str]:
    """Check whether an existing artifact manifest matches the active config."""
    manifest = load_manifest(manifest_path)
    if manifest is None:
        return False, f"manifest missing: {manifest_path}"
    expected = config_fingerprint(config)
    actual = manifest.get("config_fingerprint")
    if actual != expected:
        return False, f"config fingerprint mismatch: expected {expected}, found {actual}"
    return True, "config fingerprint matches"
