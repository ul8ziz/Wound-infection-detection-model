"""
Pipeline Utilities for YOLO11m + U-Net++ Experiment
=====================================================

Self-contained data handling:
- COCO-to-YOLO format conversion (polygon segmentation labels)
- YOLO dataset.yaml generation
- U-Net++ ROI crop dataset (WoundROIDataset)
- Albumentations transforms for ROI crops
- COCO dataset class for combined evaluation (WoundDataset)
- Utility helpers (seed, device, etc.)

Zero shared imports with experiments/maskrcnn/.
"""

import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

try:
    from pycocotools.coco import COCO
    HAS_COCO = True
except ImportError:
    HAS_COCO = False
    COCO = None

logger = logging.getLogger(__name__)

WOUND_ONLY_CLASSES = ["wound"]
WOUND_MARKER_CLASSES = ["wound", "marker"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def imread_bgr_ultralytics_safe(path: Path) -> Optional[np.ndarray]:
    """Load BGR image without ``cv2.imread`` (Ultralytics monkey-patches ``cv2.imread`` globally)."""
    try:
        buf = path.read_bytes()
    except OSError:
        return None
    arr = np.frombuffer(buf, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ============================================================================
# Utilities
# ============================================================================

def set_seed(seed: int = 42) -> None:
    """Set seed for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return appropriate torch device."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_config(
    config_path: Union[str, Path],
    *,
    validate_combined: bool = False,
) -> dict:
    """Load YAML config file.

    If ``validate_combined`` is True, parses ``combined:`` via
    ``CombinedInferenceConfig`` (same keys as ``config.yaml``) to catch
    typos early. Requires importing from ``combined`` (experiment cwd on path).
    """
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if validate_combined and data is not None:
        try:
            from combined.config import combined_config_from_dict
            combined_config_from_dict(data)
        except Exception as e:
            raise ValueError(f"Invalid combined inference config: {e}") from e
    return data


# ============================================================================
# COCO-to-YOLO Segmentation Converter
# ============================================================================

def _load_coco_json(path: Union[str, Path]) -> dict:
    """Load COCO JSON with UTF-8 encoding."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def coco_to_yolo_seg(
    coco_json_path: Union[str, Path],
    output_labels_dir: Union[str, Path],
    images_root: Union[str, Path],
    class_names: Optional[List[str]] = None,
) -> int:
    """
    Convert COCO polygon annotations to YOLO segmentation label files.

    Each output .txt contains lines: ``class_id x1 y1 x2 y2 ... xN yN``
    where coordinates are normalised to [0, 1].

    Returns the number of label files written.
    """
    coco = _load_coco_json(coco_json_path)
    output_labels_dir = Path(output_labels_dir)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    images_root = Path(images_root)

    cat_id_to_idx: Dict[int, int] = {}
    for cat in coco.get("categories", []):
        if class_names is None or cat["name"] in class_names:
            cat_id_to_idx[cat["id"]] = len(cat_id_to_idx)

    img_lookup = {img["id"]: img for img in coco["images"]}
    img_anns: Dict[int, list] = {}
    for ann in coco["annotations"]:
        if ann["category_id"] in cat_id_to_idx:
            img_anns.setdefault(ann["image_id"], []).append(ann)

    written = 0
    for img_id, img_info in img_lookup.items():
        w, h = img_info["width"], img_info["height"]
        fname = Path(img_info["file_name"]).stem
        lines: List[str] = []

        for ann in img_anns.get(img_id, []):
            cls_idx = cat_id_to_idx[ann["category_id"]]
            for seg in ann.get("segmentation", []):
                if len(seg) < 6:
                    continue
                coords = np.array(seg, dtype=np.float64).reshape(-1, 2)
                coords[:, 0] = np.clip(coords[:, 0] / w, 0.0, 1.0)
                coords[:, 1] = np.clip(coords[:, 1] / h, 0.0, 1.0)
                flat = " ".join(f"{c:.6f}" for c in coords.flatten())
                lines.append(f"{cls_idx} {flat}")

        label_path = output_labels_dir / f"{fname}.txt"
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        written += 1

    logger.info("Wrote %d YOLO label files to %s", written, output_labels_dir)
    return written


def _create_image_list(coco_json_path: Union[str, Path], images_root: Union[str, Path]) -> List[str]:
    """Return list of absolute image paths from a COCO JSON."""
    coco = _load_coco_json(coco_json_path)
    images_root = Path(images_root)
    paths = []
    for img in coco["images"]:
        p = images_root / img["file_name"]
        paths.append(str(p.resolve()))
    return paths


def image_path_to_yolo_label_path(image_path: Union[str, Path]) -> Path:
    """
    Match Ultralytics ``img2label_paths`` (see ultralytics.data.utils):
    replace the ``.../images/...`` segment with ``.../labels/...`` and use ``.txt``.
    """
    x = str(Path(image_path).resolve())
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    return Path(sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt")


def create_dataset_yaml(
    output_path: Union[str, Path],
    images_root: Union[str, Path],
    labels_root: Union[str, Path],
    train_json: Union[str, Path],
    val_json: Union[str, Path],
    test_json: Union[str, Path],
    class_names: Optional[List[str]] = None,
    train_images_root: Optional[Union[str, Path]] = None,
    val_images_root: Optional[Union[str, Path]] = None,
    test_images_root: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Generate a YOLO dataset.yaml and per-split image list files.

    Ultralytics expects paths in dataset.yaml to point to directories or
    text files listing image paths.  We write ``train.txt``, ``val.txt``,
    ``test.txt`` alongside the yaml.

    When training uses offline-augmented images in a different folder than
    val/test (e.g. ``data_root_train``), pass ``train_images_root`` while
    ``val_images_root`` / ``test_images_root`` default to ``images_root``.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_root = Path(images_root).resolve()
    roots = {
        "train": Path(train_images_root).resolve() if train_images_root is not None else base_root,
        "val": Path(val_images_root).resolve() if val_images_root is not None else base_root,
        "test": Path(test_images_root).resolve() if test_images_root is not None else base_root,
    }

    list_dir = output_path.parent
    for split, json_path in [("train", train_json), ("val", val_json), ("test", test_json)]:
        paths = _create_image_list(json_path, roots[split])
        list_file = list_dir / f"{split}.txt"
        with open(list_file, "w", encoding="utf-8") as f:
            f.write("\n".join(paths))

    names = class_names or WOUND_ONLY_CLASSES
    ds_config = {
        "path": str(output_path.parent.resolve()),
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "names": {i: n for i, n in enumerate(names)},
    }
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(ds_config, f, default_flow_style=False, allow_unicode=True)

    logger.info("Dataset YAML written to %s", output_path)
    return output_path


def prepare_yolo_dataset(
    config: dict,
    script_dir: Path,
) -> Path:
    """
    End-to-end: convert COCO annotations to YOLO format and generate
    dataset.yaml.  Returns path to the generated dataset.yaml.

    Label ``.txt`` files are written under ``<data_root>/labels/``, with the same
    stem as each image under ``<data_root>/images/``.  This matches Ultralytics,
    which resolves labels by replacing the ``images`` path segment with
    ``labels`` (see ``ultralytics.data.utils.img2label_paths``).  Writing labels
    under ``yolo_data/labels/train`` breaks training (no labels found, zero mAP).

    The class list is read from ``config["classes"]``.  Defaults to
    ``WOUND_ONLY_CLASSES`` (single-class).  Set to ``["wound", "marker"]``
    in config.yaml to enable marker detection and dynamic calibration.
    """
    project_root = script_dir.parent.parent
    data_root = (project_root / config["data_root"]).resolve()
    data_root_train = (project_root / config.get("data_root_train", config["data_root"])).resolve()

    class_names = config.get("classes", WOUND_ONLY_CLASSES)

    yolo_data_dir = script_dir / "yolo_data"

    # Train labels live next to augmented images; val/test next to main data_root
    split_roots = {
        "train": data_root_train,
        "val": data_root,
        "test": data_root,
    }
    for split, ann_key in [("train", "ann_train"), ("val", "ann_val"), ("test", "ann_test")]:
        ann_path = (project_root / config[ann_key]).resolve()
        root_for_split = split_roots[split]
        labels_dir = root_for_split / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Converting {split} ({ann_path.name}) ...")
        n = coco_to_yolo_seg(ann_path, labels_dir, root_for_split, class_names=class_names)
        print(f"    -> {n} label files written to {labels_dir}")

    dataset_yaml = yolo_data_dir / "dataset.yaml"
    create_dataset_yaml(
        output_path=dataset_yaml,
        images_root=data_root,
        labels_root=data_root / "labels",
        train_json=(project_root / config["ann_train"]).resolve(),
        val_json=(project_root / config["ann_val"]).resolve(),
        test_json=(project_root / config["ann_test"]).resolve(),
        class_names=class_names,
        train_images_root=data_root_train,
        val_images_root=data_root,
        test_images_root=data_root,
    )
    return dataset_yaml


def validate_yolo_dataset(dataset_yaml: Union[str, Path]) -> bool:
    """Quick sanity check on a YOLO dataset.yaml."""
    dataset_yaml = Path(dataset_yaml)
    if not dataset_yaml.exists():
        print(f"[FAIL] dataset.yaml not found: {dataset_yaml}")
        return False

    with open(dataset_yaml, "r", encoding="utf-8") as f:
        ds = yaml.safe_load(f)

    ok = True
    base = Path(ds.get("path", "."))
    for split in ["train", "val", "test"]:
        list_file = base / ds.get(split, f"{split}.txt")
        if not list_file.exists():
            print(f"[FAIL] {split} list not found: {list_file}")
            ok = False
            continue
        with open(list_file, "r", encoding="utf-8") as f:
            paths = [l.strip() for l in f if l.strip()]
        missing = [p for p in paths[:10] if not Path(p).exists()]
        if missing:
            print(f"[FAIL] {split}: {len(missing)} of first 10 images missing")
            ok = False
        else:
            print(f"[OK] {split}: {len(paths)} images listed, spot-check passed")

        # Ultralytics maps each image path to a label path (images -> labels)
        missing_lbl = []
        nonempty = 0
        for p in paths[:50]:
            lp = image_path_to_yolo_label_path(p)
            if not lp.exists():
                missing_lbl.append(str(lp))
            elif lp.stat().st_size > 0:
                nonempty += 1
        if missing_lbl:
            print(f"[FAIL] {split}: {len(missing_lbl)} of first 50 YOLO label files missing (expected next to images)")
            for m in missing_lbl[:3]:
                print(f"       e.g. {m}")
            ok = False
        else:
            print(f"[OK] {split}: YOLO label paths (images->labels) exist; {nonempty}/50 sample .txt non-empty")

    names = ds.get("names", {})
    print(f"[OK] Classes: {names}")
    return ok


# ============================================================================
# U-Net++ ROI Dataset
# ============================================================================

class WoundROIDataset(Dataset):
    """
    Produces (image_crop, mask_crop) pairs for U-Net++ training.

    For each annotation in the COCO JSON, crops the wound ROI using the
    ground-truth bounding box (expanded by ``roi_padding`` fraction) and
    creates the corresponding binary mask.
    """

    def __init__(
        self,
        root: Union[str, Path],
        annotation_file: Union[str, Path],
        transforms: Optional[A.Compose] = None,
        roi_padding: float = 0.1,
        target_classes: Optional[List[str]] = None,
    ):
        self.root = Path(root)
        self.transforms = transforms
        self.roi_padding = roi_padding

        with open(annotation_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        target_classes = target_classes or WOUND_ONLY_CLASSES
        cat_ids = {c["id"] for c in coco["categories"] if c["name"] in target_classes}

        self.images = {img["id"]: img for img in coco["images"]}
        self.samples: List[Tuple[int, dict]] = []
        for ann in coco["annotations"]:
            if ann["category_id"] in cat_ids:
                bbox = ann.get("bbox", [])
                if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
                    self.samples.append((ann["image_id"], ann))

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_image_path(self, file_name: str) -> Path:
        """Resolve COCO ``file_name`` under ``self.root``.

        Offline-augmented JSON stores paths like ``images/foo_aug1.jpg`` relative
        to ``data/wound_focus_clean/augmented``. If ``data_root_train`` was
        omitted from config, ``root`` may be the parent ``wound_focus_clean``;
        then the file actually lives under ``root/augmented/...``.
        """
        p = self.root / file_name
        if p.is_file():
            return p
        alt = self.root / "augmented" / file_name
        if alt.is_file():
            return alt
        return p

    def _expand_bbox(self, bbox: List[float], img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        """Expand COCO bbox [x, y, w, h] by roi_padding and clamp to image."""
        x, y, w, h = bbox
        pad_x = w * self.roi_padding
        pad_y = h * self.roi_padding
        x1 = max(0, int(x - pad_x))
        y1 = max(0, int(y - pad_y))
        x2 = min(img_w, int(x + w + pad_x))
        y2 = min(img_h, int(y + h + pad_y))
        return x1, y1, x2, y2

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_id, ann = self.samples[index]
        img_info = self.images[img_id]
        img_path = self._resolve_image_path(img_info["file_name"])
        if not img_path.is_file():
            raise FileNotFoundError(
                f"U-Net ROI image not found: {img_path}\n"
                f"  root={self.root}\n"
                f"  file_name={img_info['file_name']}\n"
                "  Fix: set `data_root_train: \"data/wound_focus_clean/augmented\"` in "
                "config.yaml when using `train_augmented.json`, and ensure "
                "`augment_offline.py` was run so images exist under augmented/images/."
            )

        image = imread_bgr_ultralytics_safe(img_path)
        if image is None:
            h = img_info.get("height", 256)
            w = img_info.get("width", 256)
            image = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_h, img_w = image.shape[:2]

        mask_full = np.zeros((img_h, img_w), dtype=np.uint8)
        for seg in ann.get("segmentation", []):
            if len(seg) < 6:
                continue
            poly = np.array(seg, dtype=np.float32).reshape(-1, 2)
            if poly.shape[0] < 3:
                continue
            cv2.fillPoly(mask_full, [poly.astype(np.int32)], 1)

        x1, y1, x2, y2 = self._expand_bbox(ann["bbox"], img_w, img_h)
        crop_img = image[y1:y2, x1:x2]
        crop_mask = mask_full[y1:y2, x1:x2]

        if crop_img.size == 0 or crop_mask.size == 0:
            crop_img = image
            crop_mask = mask_full

        if self.transforms:
            transformed = self.transforms(image=crop_img, mask=crop_mask)
            crop_img = transformed["image"]
            crop_mask = transformed["mask"]

        if not isinstance(crop_img, torch.Tensor):
            crop_img = torch.from_numpy(crop_img).permute(2, 0, 1).float() / 255.0

        if not isinstance(crop_mask, torch.Tensor):
            crop_mask = torch.from_numpy(crop_mask)

        # BCEWithLogitsLoss requires float32 targets in [0, 1]; ToTensorV2 often leaves masks as uint8
        crop_mask = torch.as_tensor(crop_mask, dtype=torch.float32)
        if crop_mask.numel() > 0 and crop_mask.max() > 1.0:
            crop_mask = crop_mask / 255.0
        crop_mask = crop_mask.clamp(0.0, 1.0).contiguous()

        if crop_mask.ndim == 2:
            crop_mask = crop_mask.unsqueeze(0)

        return crop_img, crop_mask


def get_unet_transforms(
    train: bool = True,
    image_size: Tuple[int, int] = (256, 256),
) -> A.Compose:
    """Albumentations pipeline for U-Net++ ROI crops.

    Training pipeline includes medically-safe geometric and photometric
    augmentations: shift-scale-rotate, CLAHE, color jitter, and light
    elastic transform (kept mild to preserve wound geometry).
    """
    if train:
        return A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.15, rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT_101, p=0.5,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.3),
            A.GaussNoise(p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
            A.ElasticTransform(alpha=50, sigma=10, p=0.1),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])


def create_unet_datasets(
    config: dict,
    script_dir: Path,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Build train / val / test WoundROIDataset instances."""
    project_root = script_dir.parent.parent
    data_root = (project_root / config["data_root"]).resolve()
    data_root_train = (project_root / config.get("data_root_train", config["data_root"])).resolve()
    unet_cfg = config["unet"]
    image_size = tuple(unet_cfg["input_size"])
    roi_padding = unet_cfg.get("roi_padding", 0.1)

    print(f"  U-Net train root: {data_root_train}")
    print(f"  U-Net train ann:  {config['ann_train']}")
    if data_root_train.resolve() == data_root.resolve() and "augmented" in str(
        config.get("ann_train", "")
    ):
        print(
            "  [WARNING] data_root_train equals data_root but ann_train references "
            "augmented data — set data_root_train to data/wound_focus_clean/augmented "
            "or images will be looked up under the wrong folder.",
        )

    train_ds = WoundROIDataset(
        root=data_root_train,
        annotation_file=str((project_root / config["ann_train"]).resolve()),
        transforms=get_unet_transforms(train=True, image_size=image_size),
        roi_padding=roi_padding,
    )
    val_ds = WoundROIDataset(
        root=data_root,
        annotation_file=str((project_root / config["ann_val"]).resolve()),
        transforms=get_unet_transforms(train=False, image_size=image_size),
        roi_padding=roi_padding,
    )
    test_ds = WoundROIDataset(
        root=data_root,
        annotation_file=str((project_root / config["ann_test"]).resolve()),
        transforms=get_unet_transforms(train=False, image_size=image_size),
        roi_padding=roi_padding,
    )
    return train_ds, val_ds, test_ds


def unet_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack batches; force mask batch to float32 so default_collate never keeps uint8."""
    imgs = torch.stack([b[0] for b in batch], dim=0)
    masks = torch.stack([torch.as_tensor(b[1], dtype=torch.float32) for b in batch], dim=0)
    return imgs, masks


def make_unet_dataloaders(
    train_ds: Dataset,
    val_ds: Dataset,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and val DataLoaders for U-Net++."""
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=unet_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=unet_collate_fn,
    )
    return train_loader, val_loader


# ============================================================================
# COCO Dataset for Combined Evaluation
# ============================================================================

class WoundDataset(Dataset):
    """
    Full-image COCO dataset for combined pipeline evaluation.
    Returns (image_tensor, target_dict) matching Mask R-CNN conventions
    so we can compute COCO-style metrics.
    """

    def __init__(
        self,
        root: Union[str, Path],
        annotation_file: Union[str, Path],
        image_size: Tuple[int, int] = (640, 640),
        target_classes: Optional[List[str]] = None,
    ):
        self.root = Path(root)
        self.image_size = image_size

        with open(annotation_file, "r", encoding="utf-8") as f:
            self.coco_json = json.load(f)

        target_classes = target_classes or WOUND_ONLY_CLASSES
        self.class_mapping: Dict[int, int] = {}
        for cat in self.coco_json.get("categories", []):
            if cat["name"] in target_classes:
                new_id = target_classes.index(cat["name"]) + 1
                self.class_mapping[cat["id"]] = new_id

        self.num_classes = len(target_classes) + 1
        self.images = {img["id"]: img for img in self.coco_json["images"]}
        self.img_to_anns: Dict[int, list] = {}
        for ann in self.coco_json["annotations"]:
            if ann["category_id"] in self.class_mapping:
                self.img_to_anns.setdefault(ann["image_id"], []).append(ann)

        self.ids = list(self.images.keys())

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict]:
        img_id = self.ids[index]
        img_info = self.images[img_id]
        img_path = self.root / img_info["file_name"]

        image = imread_bgr_ultralytics_safe(img_path)
        if image is None:
            image = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = image.shape[:2]
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]))

        anns = self.img_to_anns.get(img_id, [])
        boxes, labels, masks, areas = [], [], [], []

        for ann in anns:
            if ann["category_id"] not in self.class_mapping:
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue

            scale_x = self.image_size[1] / orig_w
            scale_y = self.image_size[0] / orig_h
            x1 = x * scale_x
            y1 = y * scale_y
            x2 = (x + w) * scale_x
            y2 = (y + h) * scale_y
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_mapping[ann["category_id"]])
            areas.append(ann.get("area", w * h))

            mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            for seg in ann.get("segmentation", []):
                if len(seg) < 6:
                    continue
                poly = np.array(seg, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [poly], 1)
            mask = cv2.resize(mask, (self.image_size[1], self.image_size[0]),
                              interpolation=cv2.INTER_NEAREST)
            masks.append(mask)

        new_h, new_w = self.image_size
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        img_tensor = (img_tensor - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)

        target: Dict[str, torch.Tensor] = {"image_id": torch.tensor([img_id])}
        if boxes:
            target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.tensor(labels, dtype=torch.int64)
            target["masks"] = torch.tensor(np.array(masks), dtype=torch.uint8)
            target["area"] = torch.tensor(areas, dtype=torch.float32)
            target["iscrowd"] = torch.zeros(len(boxes), dtype=torch.int64)
        else:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["masks"] = torch.zeros((0, new_h, new_w), dtype=torch.uint8)
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        target["orig_size"] = torch.tensor([orig_h, orig_w])
        target["file_name"] = img_info["file_name"]
        return img_tensor, target
