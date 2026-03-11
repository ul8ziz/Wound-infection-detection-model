# Augmentation Pipeline Documentation

This document describes the augmentation workflow for wound detection training, including online and offline augmentation, medical safety constraints, and training modes.

---

## 1. What Is Online Augmentation?

**Online augmentation** is applied at training time, per sample, inside the dataset pipeline. Each time an image is loaded for a batch, random transforms are applied before the image is passed to the model.

**Where it is applied in code:**
- `experiments/maskrcnn/pipeline_utils.py`: `WoundDataset.__getitem__` calls `self.transforms(image=..., bboxes=..., labels=..., masks=masks)`
- `get_transforms()` imports `get_medical_augmentation_pipeline` from `scripts/augmentation_strategy.py` when `use_medical_augmentation=True`
- The returned `A.Compose` is stored in `WoundDataset.transforms` and applied per sample

---

## 2. What Is Offline Augmentation?

**Offline augmentation** is a one-time preprocessing step that generates new images and annotations on disk. The script `scripts/apply_augmentation_only.py` reads `data/annotations_cleaned.json`, applies transforms, and writes:
- `data/augmented_clean/images/` — augmented images
- `data/augmented_clean/annotations_augmented_clean.json` — COCO annotations

Offline augmentation expands the dataset (e.g. 3 augmented copies + original per image) before training. It is optional.

---

## 3. Why the Old Offline Dataset Was Unsafe

The folder `data/augmented/` with `annotations_augmented.json` was generated **before** the dataset was cleaned. It was built from pre-cleaning annotations (e.g. `data/splits/train.json` or raw COCO), which contained:
- Invalid polygons
- Zero-area masks
- Incorrect category mappings
- Annotations outside image bounds

Using this contaminated data would train the model on incorrect labels and hurt performance. **Do not use `data/augmented/` or `annotations_augmented.json`.**

---

## 4. Chosen Strategy and Rationale

**Strategy: Option A — Clean dataset + online augmentation only (recommended).**

| Criterion | Option A | Option B (offline + online) |
|-----------|----------|-----------------------------|
| Dataset size | ~530 images | ~2120 (4× expansion) |
| Medical safety | High | Depends on validation |
| Duplication risk | None | High (same transforms twice) |
| Implementation | Simpler | More complex |

**Rationale:** With ~530 cleaned images and strong online augmentation (moderate intensity), offline expansion adds redundancy. Offline and online would both apply flips, rotations, and photometric transforms. Online-only reduces complexity and avoids validation/regeneration overhead.

**Option B** (clean + offline_clean + online) is supported for experiments but not the default. If used, offline should be complementary (e.g. geometric only) to avoid stacking identical transforms.

---

## 5. Transforms Used and Medical Acceptability

### Geometric (preserve_marker=True)
- **HorizontalFlip** (p=0.5), **VerticalFlip** (p=0.2): Safe; preserve marker shape
- **Rotate** (±10°): Conservative; keeps 3×3 cm marker usable
- **Affine** (translate ±5%, scale 0.95–1.05, rotate ±8°, shear ±3°): Limited; preserves marker geometry

### Photometric (moderate intensity)
- **RandomBrightnessContrast**, **RandomGamma**: Simulate lighting variations
- **HueSaturationValue** (hue ±5, sat ±10, val ±10): Limited; preserves tissue colors
- **GaussNoise**: Sensor noise
- **GaussianBlur**, **MotionBlur**: Slight focus variation
- **CLAHE**: Medical contrast enhancement

---

## 6. Forbidden Transforms and Reasons

| Transform | Reason |
|------------|--------|
| ElasticTransform, GridDistortion, OpticalDistortion | Non-linear deformations distort the 3×3 cm marker; invalidate pixel-to-cm² conversion |
| Perspective, PiecewiseAffine | Strong perspective distorts marker geometry |
| RandomCrop (uncontrolled) | Could crop out marker or wound center |
| Cutout / CoarseDropout on masks | Incomplete annotations confuse the model |
| ChannelShuffle | Destroys tissue color semantics (e.g. red necrosis → green); medically inappropriate |
| Strong ColorJitter, Posterize | Unrealistic colors hurt generalization |

---

## 7. How to Regenerate Offline Augmentation (If Used)

From the project root:

```bash
cd scripts
python apply_augmentation_only.py
```

**Requirements:**
- `data/annotations_cleaned.json` must exist (run `clean_dataset.py` first)
- No fallback to pre-cleaning splits; cleaned annotations only

**Output:**
- `data/augmented_clean/images/`
- `data/augmented_clean/annotations_augmented_clean.json`

**Full documentation:** See [docs/AUGMENTED_CLEAN.md](AUGMENTED_CLEAN.md) for detailed inputs, outputs, workflow, and configuration.

---

## 8. How to Train in Each Mode

### Mode 1: Clean only
- No online augmentation
- Set `data_mode: "clean_only"` in CONFIG
- `use_medical_augmentation` is forced to `False`

### Mode 2: Clean + online augmentation (recommended)
- Online augmentation applied per sample
- Set `data_mode: "clean_online_aug"` in CONFIG
- `use_medical_augmentation: True`, `intensity: "moderate"`

### Mode 3: Clean + offline_clean + online augmentation
- Train on `data/augmented_clean/`; val on cleaned split
- Set `data_mode: "clean_offline_aug"` in CONFIG
- Requires `apply_augmentation_only.py` to be run first

**CLI (train_model.py):**
```bash
cd experiments/maskrcnn
python train_model.py
# Or: python train_model.py --data-mode clean_online_aug
```
Edit CONFIG in `train_model.py` or use `--data-mode` to set the mode.

**Notebook (training_pipeline.ipynb):**
Edit CONFIG cell: `'data_mode': 'clean_online_aug'`.

---

## 9. Practical Recommendation

**Use Mode 2 (clean + online augmentation)** for training. It provides:
- Medically safe transforms
- No contamination risk
- Sufficient variability for ~530 images
- Simpler pipeline

Delete `data/augmented/` if it exists; it is contaminated and unused.
