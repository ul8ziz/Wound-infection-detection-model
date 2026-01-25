# Inference and Wound Analysis Guide

## Overview

This guide explains the improved inference pipeline, confidence threshold handling, and wound area/infection detection capabilities.

## Key Changes

### 1. New Inference Functions (`train_model.py`)

#### `run_inference(model, image, device, conf_thresh=0.5)`
- Runs inference on a single image
- Returns filtered predictions (above threshold) and raw predictions
- Handles image loading from path, numpy array, or tensor

#### `run_wound_inference(model, image, device, conf_thresh, wound_class_ids, infection_class_ids, marker_class_id)`
- Complete wound analysis pipeline
- Computes wound area in cm² using reference marker
- Detects infection indicators
- Returns structured results dictionary

### 2. Improved Visualization (Notebook Cell 15)

- **Configurable threshold**: Set `CONF_THRESH` at the top of the cell (default: 0.3)
- **Color-coded predictions**:
  - Red: High confidence (≥ 0.7)
  - Orange: Medium confidence (0.5 - 0.7)
  - Yellow: Low confidence (threshold - 0.5)
- **Statistics display**: Shows raw vs filtered detections, max scores

### 3. Training Diagnostics

- `validate_one_epoch()` now tracks prediction scores when `track_predictions=True`
- Logs mean/median/max scores and counts at thresholds 0.3, 0.5, 0.7
- Helps diagnose low recall issues

## Step-by-Step Usage

### Step 1: Run Inference with Different Thresholds

```python
from train_model import run_inference

# Low threshold for high recall
result_low = run_inference(model, image, device, conf_thresh=0.25)
print(f"Detections at 0.25: {result_low['num_detections']}")

# Standard threshold
result_std = run_inference(model, image, device, conf_thresh=0.5)
print(f"Detections at 0.5: {result_std['num_detections']}")

# High threshold for precision
result_high = run_inference(model, image, device, conf_thresh=0.7)
print(f"Detections at 0.7: {result_high['num_detections']}")
```

### Step 2: Visualize Predictions

1. Open Cell 15 in the notebook
2. Adjust `CONF_THRESH` (try 0.25, 0.3, 0.5)
3. Run the cell to see:
   - Ground truth boxes (green)
   - Predictions color-coded by confidence
   - Statistics for each image

### Step 3: Compute Wound Area and Infection

1. Open Cell 17 in the notebook
2. Configure class IDs:
   ```python
   WOUND_CLASS_IDS = [1, 2]  # Your wound class IDs
   INFECTION_CLASS_IDS = [3, 4, 5]  # Your infection class IDs
   MARKER_CLASS_ID = 6  # Your marker class ID
   ```
3. Run the cell to get:
   - Wound area in pixels and cm²
   - Infection detection flags
   - All detections with scores

## Training Recommendations for Better Recall

### Current Settings Analysis

**Current configuration:**
- Optimizer: SGD (lr=0.005, momentum=0.9, weight_decay=0.0005)
- Scheduler: StepLR (step_size=5, gamma=0.1)
- Epochs: 20
- Batch size: 4

**Issues:**
1. **Learning rate may be too high** for fine-tuning Mask R-CNN
2. **Too few epochs** - Mask R-CNN typically needs 30-50+ epochs
3. **No warmup** - sudden high LR can destabilize training
4. **Scheduler too aggressive** - reduces LR every 5 epochs

### Recommended Improvements

#### Option 1: Conservative Training (Recommended)
```python
# In notebook Cell 9, update optimizer and scheduler:
optimizer = optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
# Use CosineAnnealingLR for smoother decay
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# In CONFIG:
"epochs": 50  # Train longer
```

#### Option 2: With Warmup
```python
from torch.optim.lr_scheduler import LambdaLR

def warmup_lr(epoch, warmup_epochs=5):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 1.0

optimizer = optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda e: warmup_lr(e, 5))
```

#### Option 3: Lower Initial LR
```python
optimizer = optim.SGD(params, lr=0.0005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
```

### Data Augmentation

Current augmentations are good, but consider:
- **Less aggressive augmentation** if dataset is small
- **More epochs** to compensate for augmentation
- **Class balancing** if some classes are rare

## Thesis Explanation

### Why 0.5 Threshold Leads to "No Predictions"

**Problem:**
Using a confidence threshold of 0.5 for Mask R-CNN inference often results in zero detections, even when ground-truth annotations exist. This occurs because:

1. **Training dynamics**: Mask R-CNN models, especially when fine-tuned on small medical datasets, may not produce high-confidence predictions initially. The model learns to distinguish classes, but prediction scores remain conservative.

2. **Class imbalance**: Medical datasets often have imbalanced classes (e.g., many background regions, few wound instances). The model becomes cautious, assigning lower scores to avoid false positives.

3. **Limited training**: With only 20 epochs and potentially suboptimal learning rates, the model may not fully converge, resulting in lower confidence scores across all predictions.

**Solution:**
Lowering the confidence threshold (e.g., to 0.3 or 0.25) increases recall by accepting predictions with moderate confidence. This is particularly important for medical applications where missing a wound (false negative) is more critical than a false positive. The trade-off is increased false positives, but these can be filtered post-processing or by training longer.

**Impact on Wound Area and Infection Detection:**
When no predictions pass the 0.5 threshold:
- **Wound area cannot be computed** - requires at least one wound detection
- **Marker detection fails** - prevents pixel-to-cm² conversion
- **Infection indicators are missed** - no infection class detections

By lowering the threshold to 0.3, we enable:
- Wound detection even with moderate confidence
- Marker detection for area conversion
- Infection class detection for clinical assessment

**Training improvements** (longer training, better LR schedule) help the model produce higher confidence scores naturally, reducing the need for very low thresholds while maintaining high recall.

## Code Locations

- **Inference functions**: `notebooks/train_model.py` (functions: `run_inference`, `run_wound_inference`)
- **Visualization**: `notebooks/1.8.2025/fixed_training_pipeline.ipynb` (Cell 15)
- **Wound analysis**: `notebooks/1.8.2025/fixed_training_pipeline.ipynb` (Cell 17)
- **Training loop**: `notebooks/1.8.2025/fixed_training_pipeline.ipynb` (Cell 11)

## Quick Reference

```python
# Basic inference
result = run_inference(model, image_path, device, conf_thresh=0.3)

# Wound analysis
result = run_wound_inference(
    model, image_path, device,
    conf_thresh=0.3,
    wound_class_ids=[1, 2],
    infection_class_ids=[3, 4, 5],
    marker_class_id=6
)

# Access results
wound_area_cm2 = result['wound_area_cm2']
infection_flags = result['infection_flags']
detections = result['detections']
```

