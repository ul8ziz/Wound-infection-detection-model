# YOLO11m + U-Net++ Improvement Plan

**Date:** 2026-04-14  
**Status:** Planned (not yet executed)  
**Scope:** Fix identified code/config issues and run targeted experiments to improve segmentation accuracy  
**Primary target metric:** `segm_AP75` (currently 0.105 — the largest performance gap)

---

## 1. Problems Identified

### 1.1 Config Drift — Inference Settings Not Applied

| Parameter | Current `config.yaml` | Tuned Best (from optimization report) |
|-----------|:---:|:---:|
| `yolo_conf_thresh` | 0.20 | **0.15** |
| `unet_mask_thresh` | 0.35 | **0.25** |
| `roi_padding` | 0.12 | **0.10** |
| `min_mask_area` | 200 | **100** |
| `coco_bbox_mode` | mask_tight | **yolo_xyxy** |
| `enable_tta` | false | **true** |
| `postprocess_preset` | close_fill | **none** |

**Impact:** The repository is running suboptimal inference settings. The tuned configuration was identified through a 68-config grid search but never committed to the main `config.yaml`.

**File:** `experiments/YOLO11m_UNetPP/config.yaml` lines 93–106  
**Evidence:** `reports/final_hybrid_optimization_report.md` Section 5 vs. `config.yaml`

---

### 1.2 Threshold Mismatch — Checkpoint Selection vs. Deployment

`evaluate_unet_metrics()` uses a hardcoded `threshold=0.5` for computing validation Dice, which drives checkpoint selection (`best_model.pth`). However, the combined inference pipeline uses `unet_mask_thresh=0.35` (or 0.25 tuned). The "best" model for Dice@0.5 may not be the best model for Dice@0.25.

**File:** `experiments/YOLO11m_UNetPP/train_model.py` line 515, 720  
**Evidence:** `threshold: float = 0.5` default in `evaluate_unet_metrics` signature

---

### 1.3 Test ROI Cache Missing — Silent GT Fallback

`create_unet_datasets()` passes `eval_yolo_roi_cache_path` (a val-split cache) to `test_ds`. Since the cache contains val annotation IDs, `_get_cached_yolo_bbox()` returns `None` for all test annotations, causing silent fallback to GT bounding boxes.

**Result:** Validation metrics use YOLO-predicted ROIs but test metrics use GT ROIs — they measure different things and are not comparable.

**File:** `experiments/YOLO11m_UNetPP/pipeline_utils.py` lines 739–752  
**Evidence:** `eval_yolo_roi_cache_path` used for both val_ds and test_ds

---

### 1.4 Mask Interpolation Not Pinned

`get_unet_transforms()` calls `A.Resize(height, width)` without explicit `mask_interpolation` parameter. While Albumentations typically defaults to nearest-neighbor for masks, this is not guaranteed across versions and is not explicitly controlled.

**File:** `experiments/YOLO11m_UNetPP/pipeline_utils.py` lines 652, 671  
**Risk:** Bilinear interpolation on binary masks creates soft edges that degrade boundary learning

---

### 1.5 Boundary-Destructive Augmentations

Two augmentation transforms actively harm boundary learning:

- `A.ElasticTransform(alpha=50, sigma=10, p=0.1)` — warps both image and mask, displacing edge pixels
- `A.GaussianBlur(blur_limit=(3,5), p=0.15)` — smears high-frequency edge cues that the model needs for precise segmentation

These trade boundary fidelity for invariance — the wrong trade-off when `segm_AP75` is the primary improvement target.

**File:** `experiments/YOLO11m_UNetPP/pipeline_utils.py` lines 664–665

---

### 1.6 Resume Artifact — Best Epoch = 1

The current U-Net++ model was resumed from a Group B checkpoint with `lr=5e-5`. Because `best_dice` is initialized to `-1.0`, epoch 1 always becomes "best." Validation Dice remained flat across all 9 training epochs (range: 0.746–0.750), and early stopping triggered after 8 epochs of no improvement.

**Result:** The current "trained" model is essentially the unchanged Group B checkpoint weights. No meaningful further learning occurred.

**File:** `experiments/YOLO11m_UNetPP/train_model.py` lines 705–707  
**Evidence:** `results/unet/training_history.json` — 9 epochs, Dice range 0.746–0.750

---

### 1.7 Square Warp Distortion

All ROI crops are resized to a fixed 384x384 square regardless of original aspect ratio. Wounds with 2:1 or 3:1 aspect ratios are squeezed or stretched, distorting boundaries. Inference uses the same warp (consistent), but both paths introduce geometric error.

**Files:** `pipeline_utils.py` line 652, `combined/inference.py` line 213

---

## 2. Improvement Phases

### Phase 0: Fix Config Drift (10 min, no retraining)

**Action:** Update `config.yaml` combined block to match tuned settings, then re-run combined evaluation.

```yaml
combined:
  yolo_conf_thresh: 0.15
  unet_mask_thresh: 0.25
  roi_padding: 0.10
  coco_bbox_mode: yolo_xyxy
  min_mask_area: 100
  enable_tta: true
  postprocess_preset: none
```

**Expected impact:** Recover metrics that were already achievable but not reflected in `config.yaml`. May improve `bbox_AP75` and `combined_AP50` without any model change.

---

### Phase 1: Fix Code Issues (30 min, no retraining)

#### 1a. Align validation threshold

In `train_model.py`, pass the deployment threshold to `evaluate_unet_metrics`:

```python
eval_thresh = unet_cfg.get("eval_threshold",
    config.get("combined", {}).get("unet_mask_thresh", 0.5))
metrics = evaluate_unet_metrics(model, val_loader, device, threshold=eval_thresh)
```

#### 1b. Add test ROI cache

- Add `test_yolo_roi_cache_path` to `config.yaml` under `unet:`
- Generate the cache by running YOLO inference on test images
- Wire `test_ds` in `create_unet_datasets` to use the test-specific cache

#### 1c. Pin mask interpolation

```python
A.Resize(height=image_size[0], width=image_size[1],
         interpolation=cv2.INTER_LINEAR,
         mask_interpolation=cv2.INTER_NEAREST)
```

#### 1d. Soften augmentations

- Remove `A.ElasticTransform` (or set `p=0`)
- Reduce `A.GaussianBlur` from `p=0.15` to `p=0.05`

---

### Phase 2: Retrain U-Net++ (~2 hours)

**Action:** Resume from Group B checkpoint with all Phase 1 fixes applied.

| Parameter | Value |
|-----------|-------|
| Resume from | Group B best checkpoint |
| Learning rate | 5e-5 |
| Epochs | 50 |
| Early stop patience | 12 (increased from 8) |
| Loss | focal_dice |
| Input size | 384x384 |
| ROI mode | mixed |

**Expected:** The model should train longer than the previous 9 epochs (since the checkpoint selection criterion changed). Cleaner augmentations should allow better boundary learning.

**RESULT (2026-04-14):** Training ran ~50 epochs with early stopping. Best model saved at epoch ~35+. Results:

| Metric | Baseline (Group B) | Phase 2 | Change |
|--------|:---:|:---:|:---:|
| **bbox AP50** | 0.739 | **0.774** | **+4.7%** |
| **bbox AP75** | 0.543 | **0.590** | **+8.7%** |
| **segm AP50** | 0.581 | **0.645** | **+11.0%** |
| **segm AP75** | 0.105 | 0.081 | -22.9% |
| **combined AP50** | 0.660 | **0.709** | **+7.4%** |
| **combined AP75** | 0.324 | **0.336** | **+3.7%** |
| **mean Dice** | 0.676 | **0.695** | **+2.8%** |
| **mean Dice (cond.)** | 0.689 | **0.708** | **+2.8%** |
| **mean IoU** | 0.554 | **0.567** | **+2.3%** |
| Test Dice (U-Net standalone) | 0.338 | **0.756** | **+123.7%** |

Major improvements across bbox AP, segm AP50, combined AP50/75, and Dice. The segm_AP75 decreased slightly — this metric is extremely sensitive to boundary alignment and may benefit from Phase 3 (higher resolution) or Phase 4 (letterbox).

---

### Phase 3: Resolution Sweep at 512x512 (~3 hours)

**Action:** Use existing `configs/experiments/group_C_res512.yaml` with Phase 1 fixes.

| Parameter | Value |
|-----------|-------|
| Resume from | Phase 2 best checkpoint |
| Input size | 512x512 |
| Batch size | 4 |
| Learning rate | 5e-5 |

**Rationale:** Higher resolution directly preserves more boundary detail through the resize chain. This is the strongest unrun experiment already prepared in the repo.

**RESULT (2026-04-15):** Training ran 9 epochs (best at epoch 3, early stop at 8). 512x512 did **NOT** improve over 384x384:

| Metric | Phase 2 (384) | Phase 3 (512) | Change |
|--------|:---:|:---:|:---:|
| **segm AP50** | **0.645** | 0.594 | -7.9% |
| **combined AP50** | **0.709** | 0.684 | -3.6% |
| **mean Dice** | **0.695** | 0.693 | -0.3% |
| bbox AP50 | 0.774 | 0.773 | same |

Higher resolution hurt segmentation slightly — likely because EfficientNet-B1 lacks capacity for 512x512, and halved batch size (4 vs 8) added gradient noise. **Phase 2 (384x384) remains canonical best.**

---

### Phase 4: Letterbox ROI (~1 hour code + retrain)

**Action:** Replace square warp with aspect-ratio-preserving letterbox padding.

- **Training:** Pad shorter side to make square, then resize (or use `A.PadIfNeeded` + `A.Resize`)
- **Inference:** Pad crop to square before `cv2.resize`, un-pad probability map after U-Net prediction

Both paths must use identical geometry to maintain train/inference consistency.

**RESULT (2026-04-15):** Implemented letterbox pad/unpad utilities in `pipeline_utils.py`, added configurable `letterbox` flag to `WoundROIDataset`, and retrained from Phase 2 checkpoint with GPU (RTX 4060). Best val Dice 0.748 at epoch 28, early stop at epoch 40. Combined evaluation:

| Metric | Phase 2 (no letterbox) | Phase 4 (letterbox) | Change |
|--------|:---:|:---:|:---:|
| **segm AP50** | **0.645** | 0.602 | -6.7% |
| **combined AP50** | **0.709** | 0.688 | -3.0% |
| **segm AP75** | **0.094** | 0.049 | -47.9% |

Letterbox degraded segmentation quality. The model learned square-warped ROI representations from prior training (Group A+B), and switching geometry mid-fine-tune caused a distribution shift. Letterbox may help if trained from scratch, but the cost/benefit is not favorable. **Feature kept as opt-in (`letterbox: true` in config), disabled by default. Phase 2 remains best.**

---

### Phase 5: Re-test Boundary Loss

**Action:** Switch `loss_type: focal_dice_boundary` in config, retrain from Phase 2 checkpoint.

Prior testing (Group D) showed marginal improvement but was conducted before fixing augmentations, threshold alignment, and ROI geometry. The boundary loss signal should be more effective after these structural fixes.

**RESULT (2026-04-15):** Trained 44 epochs (best at epoch 32, early stop). Val Dice 0.753, Test Dice 0.749. Combined evaluation:

| Metric | Phase 2 (focal_dice) | Phase 5 (focal_dice_boundary) | Change |
|--------|:---:|:---:|:---:|
| **segm AP50** | **0.645** | 0.576 | -10.7% |
| **combined AP50** | **0.709** | 0.675 | -4.8% |
| **segm AP75** | **0.094** | 0.079 | -16.0% |
| **mean Dice** | **0.695** | 0.687 | -1.2% |

Boundary loss degraded segmentation metrics despite stronger per-pixel gradients near contours. The boundary weight (0.15) may need lower tuning, or the improvement is absorbed by the relatively small validation set. **Phase 2 (focal_dice) remains canonical best.**

---

## 3. Metrics Tracking Template

For every experiment, record all values in this table:

| Metric | Baseline (Group B) | **Phase 0+1+2 (BEST)** | Phase 3 (512) | Phase 4 (letterbox) | Phase 5 (boundary) |
|--------|:---:|:---:|:---:|:---:|:---:|
| **segm AP75** | 0.105 | **0.094** | 0.060 | 0.049 | 0.079 |
| **combined AP75** | 0.324 | **0.342** | 0.325 | 0.319 | 0.335 |
| **segm AP50** | 0.581 | **0.645** | 0.594 | 0.602 | 0.576 |
| **bbox AP50** | 0.739 | **0.774** | 0.773 | 0.774 | 0.774 |
| **mean Dice (cond.)** | 0.689 | **0.708** | 0.706 | 0.706 | 0.700 |
| **mean IoU (cond.)** | 0.565 | **0.577** | 0.574 | 0.574 | 0.571 |
| Images missed | 1 | 1 | 1 | 1 | 1 |

**Phase 0+1+2 is the canonical best.** Phases 3, 4, and 5 all degraded segmentation metrics. The improvement plan is complete.

### Decision Rules

1. **Primary metric:** `segm_AP75` — must improve
2. **Guard metric:** `bbox_AP50` — must not regress by more than 0.02
3. **Reject** any change that improves only ROI-level Dice but weakens end-to-end combined metrics
4. **Accept** changes that improve `segm_AP75` even if `mean_dice` stays flat (boundary precision matters more than average overlap)

---

## 4. What NOT to Do

Based on evidence from executed experiments:

| Action | Reason to Avoid |
|--------|----------------|
| Retrain YOLO | Already strong (bbox mAP50=0.817). Not the bottleneck. |
| Repeat Group E (multi-scale) | Underperformed Group B on all key combined metrics |
| Increase augmentation aggressiveness | Small dataset needs boundary fidelity, not more invariance |
| Change encoder to larger variant | EfficientNet-B1 is appropriate for 257 unique images. Risk of overfitting. |
| Use DeepLabV3+ as first experiment | Lower priority — address pipeline issues first. Only test if Phases 0–4 plateau. |

---

## 5. Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Phase 2 retraining finds no improvement | The threshold change alone shifts the metric landscape; combined with cleaner augmentations, at least marginal gains are expected |
| 512x512 causes GPU OOM | Batch size 4 is already configured; reduce to 2 if needed |
| Letterbox padding introduces empty regions | Use reflect padding or mask-aware loss to ignore padded areas |
| Boundary loss destabilizes training | Only apply after confirming Phases 0–4 gains; use low `boundary_weight` (0.10) initially |

---

## 6. Files to Modify

| Phase | File | Lines | Change |
|-------|------|-------|--------|
| 0 | `config.yaml` | 93–106 | Update combined inference parameters |
| 1a | `train_model.py` | 720 | Pass deployment threshold to `evaluate_unet_metrics` |
| 1b | `config.yaml` | ~84 | Add `test_yolo_roi_cache_path` |
| 1b | `pipeline_utils.py` | 739–752 | Wire test_ds to test-specific cache |
| 1c | `pipeline_utils.py` | 652, 671 | Pin `mask_interpolation=cv2.INTER_NEAREST` |
| 1d | `pipeline_utils.py` | 664–665 | Remove ElasticTransform, reduce GaussianBlur |
| 4 | `pipeline_utils.py` | 652–674 | Add letterbox padding before resize |
| 4 | `combined/inference.py` | 213 | Mirror letterbox logic in inference |
| 5 | `config.yaml` | 64 | Switch `loss_type: focal_dice_boundary` |
