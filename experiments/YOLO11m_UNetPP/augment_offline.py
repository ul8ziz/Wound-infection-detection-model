"""
Offline Augmentation for YOLO11m + U-Net++ Experiment
======================================================

Generates 3 augmented variants per training image, expanding the training
set from ~257 to ~1028 images.  Outputs a new COCO JSON alongside the
augmented images so the same pipeline (convert -> yolo -> unet) works
unchanged.

Augmentations are medically safe: no strong perspective warps, no
extreme elastic transforms that would destroy wound/marker geometry.

Usage:
    python augment_offline.py                      # defaults
    python augment_offline.py --num-augments 4     # 4 variants per image
    python augment_offline.py --dry-run             # preview without writing
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent


def _get_augmentation_pipeline(image_size: Tuple[int, int] = (0, 0)) -> A.Compose:
    """Medically-safe augmentation pipeline for wound images with polygon masks."""
    transforms = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.08, scale_limit=0.15, rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT_101, p=0.6,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.04, p=0.4),
        A.GaussNoise(p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.15),
    ]
    if image_size[0] > 0 and image_size[1] > 0:
        transforms.insert(0, A.Resize(height=image_size[0], width=image_size[1]))

    return A.Compose(
        transforms,
        keypoint_params=None,
    )


def _polygon_to_mask(
    segmentation: List[List[float]], h: int, w: int,
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for seg in segmentation:
        if len(seg) < 6:
            continue
        poly = np.array(seg, dtype=np.float32).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [poly], 255)
    return mask


def _mask_to_polygons(mask: np.ndarray, min_area: int = 25) -> List[List[float]]:
    """Convert binary mask back to COCO polygon segmentation."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1,
    )
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        cnt = cnt.flatten().tolist()
        if len(cnt) >= 6:
            polygons.append(cnt)
    return polygons


def augment_dataset(
    ann_json: Path,
    data_root: Path,
    output_dir: Path,
    num_augments: int = 3,
    seed: int = 42,
    dry_run: bool = False,
) -> Path:
    """
    Augment a COCO JSON split, writing new images and an updated annotation file.

    Returns path to the generated annotation JSON.
    """
    random.seed(seed)
    np.random.seed(seed)

    with open(ann_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    img_lookup = {img["id"]: img for img in coco["images"]}
    img_anns: Dict[int, list] = {}
    for ann in coco["annotations"]:
        img_anns.setdefault(ann["image_id"], []).append(ann)

    output_images_dir = output_dir / "images"
    if not dry_run:
        output_images_dir.mkdir(parents=True, exist_ok=True)

    pipeline = _get_augmentation_pipeline()

    new_images = list(coco["images"])
    new_annotations = list(coco["annotations"])
    next_img_id = max(img["id"] for img in coco["images"]) + 1
    next_ann_id = max(ann["id"] for ann in coco["annotations"]) + 1

    copied_originals = 0
    generated = 0

    for img_info in coco["images"]:
        img_id = img_info["id"]
        img_path = data_root / img_info["file_name"]
        if not img_path.exists():
            print(f"  [SKIP] {img_path} not found")
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        if not dry_run:
            dst = output_images_dir / img_path.name
            if not dst.exists():
                cv2.imwrite(str(dst), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            copied_originals += 1

        anns = img_anns.get(img_id, [])
        if not anns:
            continue

        masks_per_ann = []
        for ann in anns:
            m = _polygon_to_mask(ann.get("segmentation", []), h, w)
            masks_per_ann.append(m)

        for aug_idx in range(num_augments):
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            for m in masks_per_ann:
                combined_mask = np.maximum(combined_mask, m)

            result = pipeline(image=image, mask=combined_mask)
            aug_img = result["image"]
            aug_mask = result["mask"]

            stem = img_path.stem
            aug_fname = f"{stem}_aug{aug_idx + 1}.jpg"

            if not dry_run:
                out_path = output_images_dir / aug_fname
                cv2.imwrite(str(out_path), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

            new_h, new_w = aug_img.shape[:2]
            new_img_entry = {
                "id": next_img_id,
                "file_name": f"images/{aug_fname}",
                "width": new_w,
                "height": new_h,
            }
            new_images.append(new_img_entry)

            new_polys = _mask_to_polygons(aug_mask)
            if new_polys:
                bbox_mask = (aug_mask > 0).astype(np.uint8)
                ys, xs = np.where(bbox_mask > 0)
                if len(xs) > 0:
                    bx, by = int(xs.min()), int(ys.min())
                    bw, bh = int(xs.max() - xs.min()), int(ys.max() - ys.min())
                else:
                    bx, by, bw, bh = 0, 0, 0, 0

                new_ann = {
                    "id": next_ann_id,
                    "image_id": next_img_id,
                    "category_id": anns[0]["category_id"],
                    "segmentation": new_polys,
                    "bbox": [bx, by, bw, bh],
                    "area": int(aug_mask.sum() / 255),
                    "iscrowd": 0,
                }
                new_annotations.append(new_ann)
                next_ann_id += 1

            next_img_id += 1
            generated += 1

    for img_entry in new_images:
        fn = img_entry["file_name"]
        if not fn.startswith("images/"):
            img_entry["file_name"] = f"images/{Path(fn).name}"

    out_coco = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco["categories"],
    }

    out_json = output_dir / f"train_augmented.json"
    if not dry_run:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(out_coco, f, indent=2, ensure_ascii=False)

    print(f"  Originals copied: {copied_originals}")
    print(f"  Augmented images generated: {generated}")
    print(f"  Total images in JSON: {len(new_images)}")
    print(f"  Total annotations in JSON: {len(new_annotations)}")
    if not dry_run:
        print(f"  Output JSON: {out_json}")
        print(f"  Output images: {output_images_dir}")

    return out_json


def main():
    parser = argparse.ArgumentParser(description="Offline augmentation for wound dataset")
    parser.add_argument("--num-augments", type=int, default=3,
                        help="Number of augmented variants per image (default: 3)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview counts without writing files")
    args = parser.parse_args()

    data_root = PROJECT_ROOT / "data" / "wound_focus_clean"
    ann_train = data_root / "train_wound_only.json"
    output_dir = data_root / "augmented"

    if not ann_train.exists():
        print(f"[ERROR] Training annotation not found: {ann_train}")
        return

    print("=" * 60)
    print("Offline Augmentation for YOLO11m + U-Net++")
    print("=" * 60)
    print(f"  Source: {ann_train}")
    print(f"  Output: {output_dir}")
    print(f"  Augments per image: {args.num_augments}")
    print(f"  Seed: {args.seed}")
    print()

    augment_dataset(
        ann_json=ann_train,
        data_root=data_root,
        output_dir=output_dir,
        num_augments=args.num_augments,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    print("\nDone. To use augmented data, update config.yaml:")
    print('  ann_train: "data/wound_focus_clean/augmented/train_augmented.json"')
    print('  data_root_train: "data/wound_focus_clean/augmented"')
    print('  (keep data_root as data/wound_focus_clean for val/test images)')


if __name__ == "__main__":
    main()
