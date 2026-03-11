# Model Selection and Checkpoints

This document explains the checkpoint strategy for the wound detection training pipeline, including why we use validation metrics instead of loss, how best and last checkpoints work, and how to use them.

---

## 1. Why Saving Every Epoch Is Not Ideal

Saving a checkpoint for every epoch (`checkpoint_epoch_1.pth`, `checkpoint_epoch_2.pth`, ...) has several drawbacks:

- **Storage**: Each checkpoint can be hundreds of MB. With 50–80 epochs, this consumes tens of GB.
- **Maintenance**: Many files make it harder to manage and choose the right model.
- **Redundancy**: Most epochs are not useful for inference; only the best (and sometimes the last) matter.

**Best practice**: Save only what you need: the best model for inference and the last checkpoint for resuming training.

---

## 2. Why Loss Is Not Reliable for Detection/Segmentation

Training loss and validation loss are poor indicators of model quality for object detection and instance segmentation:

- **Loss components**: Mask R-CNN combines classification, box regression, and mask losses. A lower loss does not guarantee better detection or segmentation.
- **Metric mismatch**: The goal is high AP (Average Precision), not low loss. A model can have lower loss but worse AP due to calibration or overfitting.
- **Task-specific metrics**: COCO metrics (bbox_AP50, segm_AP50) directly measure detection and segmentation quality.

**Conclusion**: Use validation metrics (AP50) to select the best model, not loss.

---

## 3. Why combined_AP50 Is Used

This project performs both:

- **Object detection** (bounding boxes)
- **Instance segmentation** (masks)

We need a single metric that reflects both. The chosen metric is:

```
combined_AP50 = (bbox_AP50 + segm_AP50) / 2
```

- **bbox_AP50**: Average Precision at IoU 0.5 for bounding boxes
- **segm_AP50**: Average Precision at IoU 0.5 for segmentation masks

Using the average gives equal weight to detection and segmentation. The model with the **highest** `combined_AP50` on the validation set is saved as the best model.

---

## 4. The Formula

```
combined_AP50 = (bbox_AP50 + segm_AP50) / 2
```

- If segmentation metrics are unavailable (e.g. fallback evaluator), `combined_AP50 = bbox_AP50`.
- Higher is better (unlike loss).

---

## 5. Difference Between best_model.pth and last_checkpoint.pth

| File | Purpose | When Saved | Content |
|------|---------|------------|---------|
| **best_model.pth** | Inference | When `combined_AP50` improves | `model_state_dict`, `epoch`, `best_combined_AP50`, `bbox_AP50`, `segm_AP50`, `config`, `class_mapping` |
| **last_checkpoint.pth** | Resume training | Every epoch (overwritten) | `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict` (if AMP), `epoch`, `metrics` |

**best_model.pth** is for loading the best model for inference. It does not contain optimizer/scheduler state.

**last_checkpoint.pth** is for resuming training. It contains everything needed to continue from the last epoch.

---

## 6. How Training Resumes from last_checkpoint.pth

To resume training:

1. Load `last_checkpoint.pth` with `load_checkpoint()`.
2. Restore model, optimizer, and scheduler state.
3. Start the training loop from `epoch + 1`.

Example (conceptual):

```python
from train_model import load_checkpoint

checkpoint = load_checkpoint(
    model, 
    path="checkpoints/last_checkpoint.pth",
    optimizer=optimizer,
    scheduler=scheduler
)
start_epoch = checkpoint["epoch"]  # Resume from next epoch
```

The training script does not yet implement automatic resume; you would need to add logic to detect `last_checkpoint.pth` and load it before the loop.

---

## 7. How to Use best_model.pth for Inference

```python
import torch
from pathlib import Path

# Load checkpoint
checkpoint = torch.load("checkpoints/best_model.pth", map_location="cpu", weights_only=False)

# Get model state (supports both keys for compatibility)
model_state = checkpoint.get("model") or checkpoint.get("model_state_dict")
model.load_state_dict(model_state)

model.eval()
# Run inference...
```

The checkpoint also includes `config`, `class_mapping`, `epoch`, and `best_combined_AP50` for reference.

---

## 8. How to Change the Metric in the Future

To use a different metric for best model selection:

1. **In `train_model.py`**:
   - Change the metric used in the training loop (e.g. `segm_AP75` instead of `combined_AP50`).
   - Update the comparison: higher-is-better vs lower-is-better.
   - Update `save_best_checkpoint()` to store the new metric.

2. **In `training_pipeline.ipynb`**:
   - Mirror the same logic in the training cell.

3. **In CONFIG**:
   - Add a `best_metric_name` option if you want it configurable.

Example for a different metric:

```python
# Higher is better (e.g. AP)
is_best = current_metric > (best_metric + early_stop_min_delta)

# Lower is better (e.g. loss)
is_best = current_metric < (best_metric - early_stop_min_delta)
```

---

## 9. Best Practices for Model Checkpointing

1. **Save only what you need**: Best model + last checkpoint.
2. **Use validation metrics, not loss**: For detection/segmentation, use AP-based metrics.
3. **Use a single clear metric**: e.g. `combined_AP50` for joint detection and segmentation.
4. **Separate inference vs resume**: `best_model.pth` for inference, `last_checkpoint.pth` for resume.
5. **Include metadata**: Epoch, metrics, config, and class mapping in checkpoints.
6. **Optional debug checkpoints**: Use `save_checkpoint_every_n_epochs` (e.g. 10) only when debugging; keep it disabled by default.

---

## 10. Optional: Debug Checkpoints Every N Epochs

For debugging, you can save a checkpoint every N epochs:

```python
CONFIG["save_checkpoint_every_n_epochs"] = 10  # Save checkpoint_epoch_10.pth, checkpoint_epoch_20.pth, ...
```

This creates `checkpoint_epoch_N.pth` files. Disable by setting to `0` (default).
