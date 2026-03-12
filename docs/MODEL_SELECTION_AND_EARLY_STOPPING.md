# Model Selection and Early Stopping

This document explains how the wound detection training pipeline selects the best model and when it stops training. Early stopping and best-model selection are **AP-based only**—loss is never used as the main criterion.

---

## 1. Why AP Is Better Than Loss for This Project

Training loss and validation loss are **poor indicators** of model quality for object detection and instance segmentation:

| Aspect | Loss | AP (Average Precision) |
|--------|------|-------------------------|
| **What it measures** | Sum of classification, box regression, and mask losses | Detection/segmentation quality at IoU 0.5 |
| **Task alignment** | Indirect; lower loss ≠ better detections | Direct; higher AP = better detections |
| **Calibration** | Can decrease while AP stagnates or worsens | Directly reflects what we care about |
| **Overfitting** | Loss can keep decreasing while AP drops | AP on validation set reveals overfitting |

**Conclusion**: For wound detection and segmentation, we monitor **combined_AP50** and ignore loss for model selection and early stopping.

---

## 2. Why combined_AP50 Is Used

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

## 3. Why 50 Epochs Is the Preferred Training Limit

- **Compute budget**: Training Mask R-CNN on 1024×1024 images with batch_size=2 is expensive; 50 epochs is a reasonable default for a single-GPU setup.
- **Diminishing returns**: AP typically plateaus after 30–50 epochs; extra epochs often add little benefit.
- **Early stopping**: With patience=12, training can stop earlier if no improvement; 50 epochs is the **maximum**, not a fixed target.
- **Reproducibility**: A fixed cap makes experiments comparable and avoids runaway training.

---

## 4. How Early Stopping Works

Early stopping is **AP-based only**:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| **monitor** | `combined_AP50` | Metric watched for improvement |
| **mode** | `max` | Higher is better |
| **patience** | 12 | Stop if no improvement for 12 consecutive epochs |

**Logic**:

1. After each epoch, compute `combined_AP50` on the validation set.
2. If `combined_AP50 > best_combined_AP50` (any improvement), reset the patience counter and save `best_model.pth`.
3. Otherwise, increment `epochs_without_improve`.
4. If `epochs_without_improve >= patience`, stop training.

**Important**: Best-model saving uses `combined_AP50 > best_combined_AP50` with **no min_delta**. Requiring `min_delta` caused a bug when metrics were small (e.g. 0.001–0.004): the condition `combined_AP50 > best + 0.005` was never satisfied, so `best_model.pth` stayed at epoch 1. The `early_stop_min_delta` config is reserved for future use (e.g. "meaningful improvement" mode) and is not used for best-model selection.

**Loss is never used** for early stopping or best-model selection.

---

## 5. How best_model.pth Is Selected

- **Selection criterion**: Highest `combined_AP50` on the validation set.
- **When saved**: Whenever `combined_AP50 > best_combined_AP50` (any improvement; no min_delta).
- **First epoch**: Always saved on epoch 1 so `best_model.pth` exists even if `combined_AP50` is 0 (e.g. eval failed or no predictions).
- **Loss**: Never used for selection.

`last_checkpoint.pth` is updated **every epoch** for resume; it is independent of best-model selection.

---

## 6. Configuration Summary

| Parameter | Value | Notes |
|-----------|-------|-------|
| `epochs` | 50 | Maximum training epochs |
| `early_stop_patience` | 12 | Epochs without AP improvement before stop |
| `early_stop_min_delta` | 0.003 | Reserved for future use; not used for best-model selection |
| `monitor` | `combined_AP50` | Metric for early stopping and best model |
| `mode` | `max` | Higher is better |

---

## 7. Related Documentation

- [MODEL_SELECTION_AND_CHECKPOINTS.md](MODEL_SELECTION_AND_CHECKPOINTS.md) — Checkpoint formats, resume, and inference usage.
