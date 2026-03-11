# Augmentation Strategy Redesign — Final Report

**Date:** 2026-03-11

---

## 1. Current Online Augmentation Audit

| Item | Status |
|------|--------|
| **Location** | `WoundDataset.__getitem__` in `experiments/maskrcnn/pipeline_utils.py` |
| **Source** | `get_medical_augmentation_pipeline()` from `scripts/augmentation_strategy.py` |
| **Geometric** | HorizontalFlip, VerticalFlip, Rotate(±10°), Affine (translate ±5%, scale 0.95–1.05, shear ±3°) |
| **Photometric** | RandomBrightnessContrast, RandomGamma, HueSaturationValue, GaussNoise, Blur, CLAHE |
| **Medical safety** | Appropriate for wound images; marker geometry preserved |
| **Change made** | Removed ChannelShuffle from aggressive intensity (medically unsafe) |

---

## 2. Was the Old Offline Augmentation Contaminated?

**Yes.** `data/augmented/` with `annotations_augmented.json` was generated from pre-cleaning annotations. Training no longer uses it.

---

## 3. Which Strategy Is Best and Why

**Option A — Clean dataset + online augmentation only.**

- ~530 cleaned images with strong online augmentation is sufficient
- No duplication between offline and online transforms
- Simpler pipeline, no validation/regeneration overhead
- Offline augmentation (Mode 3) supported but not recommended by default

---

## 4. Files Modified

| File | Changes |
|------|---------|
| `scripts/augmentation_strategy.py` | Removed ChannelShuffle; clarified mask handling comment |
| `scripts/apply_augmentation_only.py` | Enforced cleaned-only input; removed fallback to splits |
| `experiments/maskrcnn/train_model.py` | Implemented data_mode logic (clean_only, clean_online_aug, clean_offline_aug) |
| `experiments/maskrcnn/training_pipeline.ipynb` | Implemented DATA_MODE logic; train/val roots per mode |
| `README.md` | Added augmentation summary, mode table, and exact commands |

---

## 5. Files Added

| File | Purpose |
|------|---------|
| `docs/augmentation_pipeline.md` | Full augmentation workflow documentation |
| `docs/AUGMENTATION_STRATEGY_REPORT.md` | This final report |

---

## 6. Whether Offline Augmentation Is Still Recommended

**No.** For this project, clean + online augmentation (Mode 2) is recommended. Offline augmentation (Mode 3) is available for experiments but adds redundancy.

---

## 7. Exact Training Recommendation

Use **Mode 2 (clean_online_aug)** with:
- `data_mode: "clean_online_aug"`
- `use_medical_augmentation: True`
- `intensity: "moderate"`
- `preserve_marker: True`

---

## 8. Exact Commands to Run

```bash
# 1. Clean dataset
cd scripts
python clean_dataset.py --input-mode cvat --data-root ../data

# 2. Regenerate splits (optional)
python clean_dataset.py --input-mode coco --input-file ../data/annotations_cleaned.json --split

# 3. Regenerate offline augmentation (optional, Mode 3 only)
python apply_augmentation_only.py

# 4. Training
cd ../experiments/maskrcnn
python train_model.py
```

Edit CONFIG in `train_model.py` or run: `python train_model.py --data-mode clean_online_aug`

---

## 9. Documentation Files Created

- `docs/augmentation_pipeline.md` — Full augmentation workflow
- `docs/AUGMENTATION_STRATEGY_REPORT.md` — This report
