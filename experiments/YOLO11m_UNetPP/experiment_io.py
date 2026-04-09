from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def get_experiment_name(config: Dict[str, Any]) -> Optional[str]:
    """Return optional experiment name from top-level or U-Net config."""
    for key in ("experiment_name",):
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    unet_cfg = config.get("unet") or {}
    value = unet_cfg.get("experiment_name")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def experiment_slug(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    slug = slug.strip("._-")
    return slug or None


def get_unet_dirs(script_dir: Path, config: Dict[str, Any]) -> Dict[str, Path]:
    base_ckpt = script_dir / "checkpoints" / "unet"
    base_results = script_dir / "results" / "unet"
    slug = experiment_slug(get_experiment_name(config))
    if slug:
        return {
            "checkpoints": base_ckpt / slug,
            "results": base_results / slug,
            "base_checkpoints": base_ckpt,
            "base_results": base_results,
        }
    return {
        "checkpoints": base_ckpt,
        "results": base_results,
        "base_checkpoints": base_ckpt,
        "base_results": base_results,
    }


def get_combined_dirs(script_dir: Path, config: Dict[str, Any]) -> Dict[str, Path]:
    base_results = script_dir / "results" / "combined"
    slug = experiment_slug(get_experiment_name(config))
    if slug:
        results_dir = base_results / slug
    else:
        results_dir = base_results
    return {
        "results": results_dir,
        "predictions": results_dir / "predictions",
        "error_analysis": results_dir / "error_analysis",
        "base_results": base_results,
    }


def get_unet_best_checkpoint_path(script_dir: Path, config: Dict[str, Any]) -> Path:
    dirs = get_unet_dirs(script_dir, config)
    primary = dirs["checkpoints"] / "best_model.pth"
    if primary.exists():
        return primary

    unet_cfg = config.get("unet") or {}
    resume_checkpoint = unet_cfg.get("resume_checkpoint")
    if isinstance(resume_checkpoint, str) and resume_checkpoint.strip():
        resume_path = Path(resume_checkpoint)
        if resume_path.is_absolute():
            return resume_path
        for candidate in (
            (script_dir / resume_checkpoint).resolve(),
            (script_dir.parent.parent / resume_checkpoint).resolve(),
        ):
            if candidate.exists():
                return candidate
        return (script_dir / resume_checkpoint).resolve()

    fallback = dirs["base_checkpoints"] / "best_model.pth"
    return fallback


def snapshot_config(
    config: Dict[str, Any],
    destination_dir: Path,
    source_config_path: Optional[Path] = None,
) -> Path:
    """Save a reproducible YAML snapshot beside experiment outputs."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = destination_dir / "config_snapshot.yaml"
    if source_config_path is not None and source_config_path.is_file():
        shutil.copy2(source_config_path, snapshot_path)
        return snapshot_path

    with open(snapshot_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    return snapshot_path
