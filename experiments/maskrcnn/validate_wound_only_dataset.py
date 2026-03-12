"""
Validate Wound-Only Dataset
===========================

Pre-training validation for the wound-only dataset. Verifies:
- COCO files load correctly
- Wound class is the only class
- Image paths resolve correctly
- Masks are non-empty
- Dataset size matches expectations from the build report

Usage:
    cd experiments/maskrcnn
    python validate_wound_only_dataset.py

Exit: 0 on success, 1 on any failure.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Expected paths (relative to project root)
DATA_ROOT = PROJECT_ROOT / "data" / "wound_focus_clean"
TRAIN_ANN = DATA_ROOT / "train_wound_only.json"
VAL_ANN = DATA_ROOT / "val_wound_only.json"
TEST_ANN = DATA_ROOT / "test_wound_only.json"

# Expected counts from dataset_build_report.md
EXPECTED_TRAIN = 257
EXPECTED_VAL = 57
EXPECTED_TEST = 55

EXPECTED_CATEGORIES = [{"id": 1, "name": "wound"}]


def _normalize_path(p: str) -> str:
    return str(p).replace("\\", "/")


def load_coco(path: Path) -> dict:
    """Load COCO JSON with UTF-8."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_categories(coco: dict, split_name: str) -> bool:
    """Verify categories are exactly one wound class."""
    cats = coco.get("categories", [])
    if len(cats) != 1:
        print(f"  [FAIL] {split_name}: expected 1 category, got {len(cats)}")
        return False
    if cats[0] != EXPECTED_CATEGORIES[0]:
        print(f"  [FAIL] {split_name}: expected {EXPECTED_CATEGORIES[0]}, got {cats[0]}")
        return False
    print(f"  [OK] {split_name}: categories = {cats}")
    return True


def validate_image_paths(coco: dict, root: Path, split_name: str) -> bool:
    """Verify all image paths resolve and files exist."""
    images = coco.get("images", [])
    missing = []
    for img in images:
        fn = _normalize_path(img.get("file_name", ""))
        if not fn:
            missing.append((img.get("id"), "empty file_name"))
            continue
        full_path = root / fn
        if not full_path.exists():
            missing.append((img.get("id"), str(full_path)))
    if missing:
        for img_id, path in missing[:5]:
            print(f"  [FAIL] {split_name}: image {img_id} not found: {path}")
        if len(missing) > 5:
            print(f"  [FAIL] {split_name}: ... and {len(missing) - 5} more")
        return False
    print(f"  [OK] {split_name}: all {len(images)} images found")
    return True


def validate_annotations(coco: dict, split_name: str) -> bool:
    """Verify annotations have non-empty segmentation and valid bbox."""
    anns = coco.get("annotations", [])
    invalid = []
    for ann in anns:
        seg = ann.get("segmentation", [])
        if not seg:
            invalid.append((ann.get("id"), "empty segmentation"))
            continue
        bbox = ann.get("bbox", [])
        if len(bbox) < 4 or bbox[2] <= 0 or bbox[3] <= 0:
            invalid.append((ann.get("id"), "invalid bbox"))
    if invalid:
        for ann_id, reason in invalid[:5]:
            print(f"  [FAIL] {split_name}: annotation {ann_id}: {reason}")
        if len(invalid) > 5:
            print(f"  [FAIL] {split_name}: ... and {len(invalid) - 5} more")
        return False
    print(f"  [OK] {split_name}: all {len(anns)} annotations valid")
    return True


def validate_counts(coco: dict, split_name: str, expected: int) -> bool:
    """Verify image count matches build report."""
    n = len(coco.get("images", []))
    if n != expected:
        print(f"  [FAIL] {split_name}: expected {expected} images, got {n}")
        return False
    print(f"  [OK] {split_name}: {n} images (matches expected)")
    return True


def validate_dataset_sample(root: Path, ann_file: Path) -> bool:
    """Sample a few images via WoundDataset and verify masks."""
    try:
        sys.path.insert(0, str(SCRIPT_DIR))
        from pipeline_utils import create_dataset, WOUND_ONLY_CLASSES
    except ImportError as e:
        print(f"  [WARN] Could not import pipeline_utils: {e}. Skipping dataset sample check.")
        return True

    try:
        dataset = create_dataset(
            root=str(root),
            annotation_file=str(ann_file),
            train=False,
            image_size=(512, 512),
            use_medical_augmentation=False,
            target_classes=WOUND_ONLY_CLASSES,
        )
        n_samples = min(5, len(dataset))
        for i in range(n_samples):
            img, target = dataset[i]
            masks = target.get("masks")
            if masks is None or masks.numel() == 0:
                print(f"  [FAIL] Sample {i}: no masks in target")
                return False
            if masks.sum().item() <= 0:
                print(f"  [FAIL] Sample {i}: mask is empty")
                return False
        print(f"  [OK] Sampled {n_samples} images; masks non-empty")
        return True
    except Exception as e:
        print(f"  [FAIL] Dataset sample error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> int:
    print("=" * 60)
    print("Wound-Only Dataset Validation")
    print("=" * 60)
    print(f"Data root: {DATA_ROOT}")
    print()

    if not DATA_ROOT.exists():
        print(f"[FAIL] Data root does not exist: {DATA_ROOT}")
        return 1

    all_ok = True

    for split_name, ann_path, expected in [
        ("train", TRAIN_ANN, EXPECTED_TRAIN),
        ("val", VAL_ANN, EXPECTED_VAL),
        ("test", TEST_ANN, EXPECTED_TEST),
    ]:
        print(f"--- {split_name} ---")
        if not ann_path.exists():
            print(f"  [FAIL] Annotation file not found: {ann_path}")
            all_ok = False
            continue

        coco = load_coco(ann_path)
        if not validate_categories(coco, split_name):
            all_ok = False
        if not validate_image_paths(coco, DATA_ROOT, split_name):
            all_ok = False
        if not validate_annotations(coco, split_name):
            all_ok = False
        if not validate_counts(coco, split_name, expected):
            all_ok = False
        print()

    print("--- Dataset sample (WoundDataset) ---")
    if not validate_dataset_sample(DATA_ROOT, TRAIN_ANN):
        all_ok = False
    print()

    print("=" * 60)
    if all_ok:
        print("PASS: All validation checks passed.")
        return 0
    else:
        print("FAIL: One or more validation checks failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
