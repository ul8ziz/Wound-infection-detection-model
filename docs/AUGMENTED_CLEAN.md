# augmented_clean — Offline Augmentation Documentation

This document describes the `augmented_clean` dataset: how it is generated, its inputs, outputs, and usage.

---

## 1. Overview

**`augmented_clean`** is an offline-augmented dataset produced by `scripts/apply_augmentation_only.py`. It expands the cleaned dataset by creating multiple augmented copies of each image with aligned annotations. The output is stored in `data/augmented_clean/` and used for training when `data_mode: "clean_offline_aug"`.

**Important:** `augmented_clean` is built **only** from `annotations_cleaned.json`. The old `data/augmented/` (from pre-cleaning annotations) is contaminated and must not be used.

---

## 2. How It Works

### 2.1 Workflow

```
annotations_cleaned.json  +  original images (from data_root)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  For each image:                                             │
│  1. Load image and its annotations (bboxes, masks, labels)   │
│  2. Optionally copy original image to output (copy_original)  │
│  3. Apply augmentation N times (augmentations_per_image)     │
│     - Resize to image_size (e.g. 512×512)                     │
│     - Geometric: HorizontalFlip, VerticalFlip, Rotate, Affine│
│     - Photometric: Brightness, Contrast, Hue, Noise, CLAHE  │
│  4. Save augmented images to output_images_dir               │
│  5. Convert masks to RLE and write new annotations          │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
annotations_augmented_clean.json  +  images/
```

### 2.2 Processing Steps (per image)

1. **Load** image from `data_root / image_info['file_name']`.
2. **Extract** annotations: bboxes (COCO format `[x,y,w,h]`), masks (from polygon/RLE), category_ids.
3. **Copy original** (if `copy_original=True`): save original image as-is; reuse original annotations.
4. **Augment N times**: apply the augmentation pipeline; save each augmented image as `{stem}_aug{i}{ext}` (e.g. `IMG_001_aug1.jpg`).
5. **Write annotations**: for each new image, create COCO annotations with updated bboxes and RLE segmentations.

### 2.3 Augmentation Pipeline (offline)

The script builds its own pipeline (no `ToTensorV2`/`Normalize` since images are saved to disk):

| Stage | Transforms |
|-------|------------|
| Resize | `LongestMaxSize` → `PadIfNeeded` to `image_size` |
| Geometric | `HorizontalFlip`, `VerticalFlip`, `Rotate` (±10°), `Affine` (translate, scale, rotate, shear) |
| Photometric | `RandomBrightnessContrast`, `HueSaturationValue`, `GaussNoise`, `CLAHE` |

When `preserve_marker=True`, geometric transforms are conservative (smaller limits) to keep the 3×3 cm marker geometry valid.

---

## 3. Inputs

| Input | Path / Format | Description |
|-------|----------------|-------------|
| **Annotations** | `data/annotations_cleaned.json` | COCO format. **Required.** No fallback to splits. |
| **Images** | `data_root` + `image_info['file_name']` | Original images referenced in annotations (e.g. `task_0/data/IMG_001.jpg`). |

### Annotations structure (COCO)

```json
{
  "images": [{"id": 1, "file_name": "task_0/data/IMG_001.jpg", "width": 1920, "height": 1080}, ...],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 2, "bbox": [x,y,w,h], "segmentation": {...}, ...}],
  "categories": [{"id": 1, "name": "AllWound"}, ...]
}
```

### Prerequisites

- Run `scripts/clean_dataset.py` first to produce `data/annotations_cleaned.json`.
- All image paths in annotations must exist under `data_root`.

---

## 4. Outputs

| Output | Path | Description |
|--------|------|-------------|
| **Images** | `data/augmented_clean/images/` | Augmented images (and originals if `copy_original=True`). |
| **Annotations** | `data/augmented_clean/annotations_augmented_clean.json` | COCO format with all new images and annotations. |

### Output structure

```
data/augmented_clean/
├── images/
│   ├── IMG_001.jpg          # Original (if copy_original=True)
│   ├── IMG_001_aug1.jpg     # Augmented copy 1
│   ├── IMG_001_aug2.jpg     # Augmented copy 2
│   ├── IMG_001_aug3.jpg     # Augmented copy 3
│   └── ...
└── annotations_augmented_clean.json
```

### Annotations format (output)

- Same COCO structure as input.
- `images[].file_name` uses `images/{filename}` (e.g. `images/IMG_001_aug1.jpg`).
- `annotations[].bbox` in COCO format `[x, y, w, h]`.
- `annotations[].segmentation` in RLE format (from masks).

### Typical counts

- **Original images:** ~530 (from `annotations_cleaned.json`).
- **Per image:** 1 original + 3 augmented = 4 images (with default `augmentations_per_image=3`, `copy_original=True`).
- **Total:** ~2120 images.

---

## 5. Configuration

Configurable in `scripts/apply_augmentation_only.py`:

| Parameter | Default | Description |
|-----------|--------|-------------|
| `data_root` | `../data` | Root for original images. |
| `annotation_file` | `../data/annotations_cleaned.json` | Input annotations. |
| `output_root` | `../data/augmented_clean` | Output directory. |
| `output_images_dir` | `images` | Subfolder for images under `output_root`. |
| `output_annotations` | `annotations_augmented_clean.json` | Output annotations filename. |
| `augmentations_per_image` | `3` | Number of augmented copies per original. |
| `image_size` | `(512, 512)` | Target size for augmented images. |
| `preserve_marker` | `True` | Use conservative geometric transforms. |
| `intensity` | `"moderate"` | `"light"`, `"moderate"`, or `"aggressive"`. |
| `copy_original` | `True` | Include original images in output. |
| `seed` | `42` | Random seed for reproducibility. |

---

## 6. Usage

### Generate augmented_clean

```bash
cd scripts
python apply_augmentation_only.py
```

**Requirements:**

- `data/annotations_cleaned.json` exists (run `clean_dataset.py` first).
- Script is run from `scripts/` (paths are relative to project root).

### Use in training

Set `data_mode: "clean_offline_aug"` in CONFIG (notebook or `train_model.py`):

```python
CONFIG = {
    'data_mode': 'clean_offline_aug',  # Use augmented_clean
    ...
}
```

Or via CLI:

```bash
cd experiments/maskrcnn
python train_model.py --data-mode clean_offline_aug
```

---

## 7. Summary

| Aspect | Details |
|--------|---------|
| **Script** | `scripts/apply_augmentation_only.py` |
| **Input** | `data/annotations_cleaned.json` + images under `data_root` |
| **Output** | `data/augmented_clean/images/` + `annotations_augmented_clean.json` |
| **Format** | COCO (images, annotations, categories) |
| **Expansion** | ~4× (1 original + 3 augmented per image) |
| **Training mode** | `data_mode: "clean_offline_aug"` |
