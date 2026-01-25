# Comprehensive Data Augmentation Strategy for Medical Wound Segmentation

## Executive Summary

This guide provides a complete data expansion strategy for training Mask R-CNN on medical wound segmentation data. The strategy is specifically designed for:

- **Multi-task CVAT dataset** (~240 tasks with separate annotations)
- **Multi-class, multi-instance segmentation** (polygons per image)
- **Strong class imbalance** (common vs rare classes)
- **Critical marker preservation** (3×3 cm reference marker for area measurements)
- **High-resolution clinical photos** with varying conditions

---

## Table of Contents

1. [Dataset Analysis](#dataset-analysis)
2. [Augmentation Pipeline](#augmentation-pipeline)
3. [Transforms to Avoid](#transforms-to-avoid)
4. [Class Balancing Strategies](#class-balancing-strategies)
5. [Patch Extraction Strategy](#patch-extraction-strategy)
6. [Sampling Strategies](#sampling-strategies)
7. [Implementation Guide](#implementation-guide)
8. [Recommended Workflow](#recommended-workflow)

---

## Dataset Analysis

### Structure

- **~240 CVAT tasks** (task_0 to task_239)
- Each task has:
  - `data/` folder with wound images (PNG/JPEG)
  - `annotations.json` with polygon labels
  - `task.json` and `manifest.jsonl`

### Class Distribution

**Frequent Classes:**
- "ВсяРана" (entire wound area) - **most common**
- "Метка для размерности" (3×3 cm marker) - **critical for area conversion**

**Moderate Classes:**
- "Зона шва" (suture area)
- "Зона отека вокруг раны" (edema around wound)
- "Зона гиперемии вокруг" (hyperemia area)
- "Зона грануляций" (granulation)
- "Фибрин" (fibrin)

**Rare Classes:**
- "Зона некроза" (necrosis) - **clinically important but rare**
- "Гнойное отделяемое" (purulent discharge)
- "Сухожилие" (tendon)
- "Губка ВАК" (VAC sponge)
- "Глубины раны" (wound depths)

### Critical Constraints

1. **Marker Geometry Must Be Preserved**
   - The 3×3 cm marker is used for pixel-to-cm² conversion
   - Extreme warping would invalidate area measurements
   - Marker must remain approximately square

2. **Medical Realism**
   - Augmentations should simulate realistic clinical variations
   - Avoid unrealistic distortions or color shifts
   - Preserve tissue appearance characteristics

3. **Small Structure Preservation**
   - Rare classes (necrosis, granulation) are often small
   - Augmentations must not eliminate these structures
   - Need careful handling of small masks

---

## Augmentation Pipeline

### Overview

The augmentation pipeline is implemented in `scripts/augmentation_strategy.py` with three intensity levels:

- **Light**: Minimal changes, maximum marker preservation
- **Moderate**: Balanced augmentation (recommended default)
- **Aggressive**: Strong augmentations (use with caution)

### Pipeline Stages

#### 1. Resize (Always First)

```python
A.LongestMaxSize(max_size=max(image_size), interpolation=cv2.INTER_LINEAR)
A.PadIfNeeded(min_height=image_size[0], min_width=image_size[1], border_mode=cv2.BORDER_CONSTANT)
```

**Why:**
- Preserves aspect ratio (prevents marker distortion)
- Consistent input size for training
- Black padding maintains image structure

#### 2. Geometric Transforms (Safe for Medical Segmentation)

**With Marker Preservation (`preserve_marker=True`):**

```python
A.HorizontalFlip(p=0.5)                    # Safe, preserves marker shape
A.VerticalFlip(p=0.2)                     # Less common, lower probability
A.Rotate(limit=10, p=0.5)                 # Small rotations (±10°)
A.Affine(
    translate_percent=(-0.05, 0.05),      # Max 5% translation
    scale=(0.95, 1.05),                    # Max ±5% scale
    rotate=(-8, 8),                       # Small rotation
    shear=(-3, 3),                        # Minimal shear
    p=0.4
)
```

**Why These Limits:**
- Small rotations (±10°) preserve marker square shape
- Limited scale (±5%) maintains marker size ratio
- Minimal translation prevents marker from being cropped out
- Small shear preserves geometric relationships

#### 3. Photometric Transforms (Moderate Intensity)

```python
A.RandomBrightnessContrast(
    brightness_limit=0.2,
    contrast_limit=0.2,
    p=0.6
)  # Simulates lighting variations

A.RandomGamma(gamma_limit=(80, 120), p=0.4)  # Exposure variations

A.HueSaturationValue(
    hue_shift_limit=5,    # Small hue shifts (preserves tissue colors)
    sat_shift_limit=10,   # Moderate saturation
    val_shift_limit=10,   # Value shifts
    p=0.3
)

A.GaussNoise(var_limit=(5.0, 20.0), p=0.3)  # Sensor noise

A.OneOf([
    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
    A.MotionBlur(blur_limit=3, p=1.0),
], p=0.2)  # Focus issues

A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3)  # Contrast enhancement
```

**Why These Transforms:**
- Simulate realistic clinical variations (lighting, exposure, focus)
- Preserve tissue color characteristics (limited hue shifts)
- Add robustness to noise and blur
- CLAHE helps with varying contrast conditions

#### 4. Normalization (Always Last)

```python
A.Normalize(
    mean=[0.485, 0.456, 0.406],  # ImageNet stats (ResNet standard)
    std=[0.229, 0.224, 0.225]
)
ToTensorV2()
```

---

## Transforms to Avoid

### ❌ ElasticTransform / GridDistortion / OpticalDistortion

**Why Not:**
- Creates non-linear deformations
- Would distort the 3×3 cm marker
- Invalidates pixel-to-cm² area conversion

**Alternative:** Use small affine transforms instead

### ❌ Perspective / PiecewiseAffine

**Why Not:**
- Strong perspective changes distort marker geometry
- Marker would no longer be square
- Breaks area calculations

**Alternative:** Use small rotations and translations

### ❌ RandomCrop (without careful handling)

**Why Not:**
- Could crop out critical structures (marker, wound center)
- Loss of important annotations
- Especially problematic for rare classes

**Alternative:** 
- Use CenterCrop
- Implement wound-aware cropping (see below)
- Use padding instead of cropping

### ❌ Cutout / CoarseDropout (on masks)

**Why Not:**
- Medical segmentation requires complete masks
- Incomplete annotations confuse the model
- Breaks mask continuity

**Alternative:** Can use on images only (not masks) for robustness

### ❌ Strong ColorJitter / Posterize

**Why Not:**
- Medical images should remain realistic
- Unrealistic colors hurt generalization
- Tissue appearance is clinically important

**Alternative:** Use limited hue/saturation shifts

---

## Class Balancing Strategies

### Problem

Strong class imbalance:
- "ВсяРана" appears in most images
- Rare classes (necrosis, VAC sponge) appear in <5% of images
- Model may ignore rare classes

### Solutions

#### 1. Class-Weighted Loss

```python
from scripts.augmentation_strategy import compute_class_weights

# Compute weights from annotations
class_weights = compute_class_weights(annotations)

# Use in loss function (implement in training loop)
# Weight rare classes more heavily
```

#### 2. Oversampling Rare Classes

```python
def create_balanced_sampler(dataset, annotations):
    """
    Create a sampler that oversamples images containing rare classes.
    """
    from collections import Counter
    from torch.utils.data import WeightedRandomSampler
    
    # Identify rare class IDs
    class_counts = Counter()
    for ann in annotations['annotations']:
        class_counts[ann['category_id']] += 1
    
    rare_class_ids = {cat_id for cat_id, count in class_counts.items() 
                      if count < total_count * 0.05}  # <5% of total
    
    # Assign weights to samples
    sample_weights = []
    for idx in range(len(dataset)):
        img_id = dataset.ids[idx]
        anns = dataset.img_to_anns.get(img_id, [])
        
        # Check if image contains rare classes
        has_rare = any(ann['category_id'] in rare_class_ids for ann in anns)
        weight = 3.0 if has_rare else 1.0  # 3x more likely to sample
        sample_weights.append(weight)
    
    return WeightedRandomSampler(sample_weights, len(sample_weights))
```

#### 3. Focal Loss (Alternative)

Focal loss automatically down-weights easy examples and focuses on hard/rare cases:

```python
# Implement focal loss in training_engine.py
# Reduces contribution of easy examples (common classes)
# Increases focus on hard examples (rare classes)
```

#### 4. Multi-Task Learning

Train separate heads for:
- Common classes (wound, marker)
- Rare classes (necrosis, granulation, etc.)

This prevents rare classes from being overwhelmed.

---

## Patch Extraction Strategy

### Problem

- High-resolution images (e.g., 4000×3000 pixels)
- Full-image training is memory-intensive
- Limited number of images

### Solution: Extract Patches

```python
from scripts.augmentation_strategy import extract_patches_from_image

# Extract patches from high-res images
patches = extract_patches_from_image(
    image=image,
    masks=masks,
    patch_size=(1024, 1024),
    stride=(512, 512),  # 50% overlap
    min_mask_coverage=0.1  # At least 10% annotation coverage
)
```

### Benefits

1. **More Training Samples**: One image → multiple patches
2. **Memory Efficient**: Smaller patches fit in GPU memory
3. **Focus on Wound Regions**: Only patches with annotations
4. **Data Expansion**: Effectively multiplies dataset size

### Implementation Strategy

```python
class PatchDataset(Dataset):
    """
    Dataset that extracts patches from high-resolution images.
    """
    def __init__(self, base_dataset, patch_size=(1024, 1024), stride=(512, 512)):
        self.base_dataset = base_dataset
        self.patch_size = patch_size
        self.stride = stride
        self.patches = self._extract_all_patches()
    
    def _extract_all_patches(self):
        patches = []
        for idx in range(len(self.base_dataset)):
            image, target = self.base_dataset[idx]
            # Extract patches with annotations
            # Store (patch_image, patch_target) tuples
        return patches
```

### Considerations

- **Overlap**: 50% stride ensures no information loss at boundaries
- **Coverage Threshold**: `min_mask_coverage=0.1` ensures patches contain annotations
- **Marker Preservation**: Ensure marker is included in at least some patches
- **Validation**: Use full images for validation (no patching)

---

## Sampling Strategies

### 1. Stratified Sampling

Split dataset ensuring each split has:
- Similar class distribution
- Similar number of rare class examples
- Similar image quality distribution

```python
def stratified_split(annotations, train_ratio=0.8):
    """
    Split dataset ensuring rare classes are in both train and val.
    """
    # Group images by class presence
    # Ensure rare classes appear in both splits
    # Balance common/rare class ratio
```

### 2. Task-Aware Sampling

Since data comes from ~240 tasks:
- Ensure each task's images are split consistently
- Avoid data leakage (same patient/task in both train and val)
- Balance task distribution across splits

```python
def task_aware_split(tasks, train_ratio=0.8):
    """
    Split by task, not by individual images.
    Ensures no task appears in both train and val.
    """
    # Shuffle tasks
    # Split tasks (not images)
    # Collect images from train_tasks and val_tasks
```

### 3. Rare Class Oversampling

During training, oversample batches containing rare classes:

```python
# In DataLoader
sampler = WeightedRandomSampler(
    weights=sample_weights,  # Higher weight for rare class images
    num_samples=len(dataset),
    replacement=True
)
```

---

## Implementation Guide

### Step 1: Update Pipeline Utils

The `pipeline_utils.py` has been updated to support the new augmentation strategy:

```python
from pipeline_utils import get_transforms, create_dataset

# Use medical augmentation strategy
train_transform = get_transforms(
    train=True,
    image_size=(1024, 1024),
    use_medical_augmentation=True,  # Enable new strategy
    preserve_marker=True,            # Preserve marker geometry
    intensity="moderate"             # Moderate augmentation
)
```

### Step 2: Create Balanced Dataset

```python
from scripts.augmentation_strategy import compute_class_weights
from torch.utils.data import WeightedRandomSampler

# Compute class weights
class_weights = compute_class_weights(annotations)

# Create balanced sampler
sampler = create_balanced_sampler(train_dataset, annotations)

# Use in DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    sampler=sampler,  # Use weighted sampler
    num_workers=2
)
```

### Step 3: Apply Patch Extraction (Optional)

```python
from scripts.augmentation_strategy import extract_patches_from_image

# For high-resolution images, extract patches
# This can be done as a preprocessing step or in Dataset.__getitem__
```

### Step 4: Configure Training

```python
# In your training notebook or script
CONFIG = {
    "image_size": (1024, 1024),
    "use_medical_augmentation": True,
    "preserve_marker": True,
    "intensity": "moderate",
    "batch_size": 4,
    "epochs": 50
}

train_dataset = create_dataset(
    root=CONFIG["data_root"],
    annotation_file=CONFIG["train_ann"],
    train=True,
    image_size=CONFIG["image_size"]
)
```

---

## Recommended Workflow

### Phase 1: Baseline (No Augmentation)

1. Train with minimal augmentation (resize + normalize only)
2. Establish baseline metrics
3. Identify problematic classes

### Phase 2: Moderate Augmentation

1. Enable medical augmentation with `intensity="moderate"`
2. Use `preserve_marker=True`
3. Monitor:
   - Overall metrics (mAP, IoU)
   - Rare class performance (necrosis, granulation)
   - Marker detection accuracy

### Phase 3: Class Balancing

1. Implement class-weighted loss or oversampling
2. Focus on rare class improvement
3. Ensure common classes don't degrade

### Phase 4: Advanced Strategies (If Needed)

1. Patch extraction for high-res images
2. Wound-aware cropping
3. Task-aware splitting

### Phase 5: Fine-Tuning

1. Adjust augmentation intensity based on results
2. Experiment with `intensity="light"` or `"aggressive"` if needed
3. Fine-tune probabilities of individual transforms

---

## Monitoring and Validation

### Key Metrics to Track

1. **Overall Performance:**
   - mAP (mean Average Precision)
   - IoU (Intersection over Union)
   - Combined AP50 (bbox + segm)

2. **Class-Specific Performance:**
   - AP per class (especially rare classes)
   - Recall for rare classes (necrosis, granulation)
   - Precision for common classes (wound, marker)

3. **Marker-Specific:**
   - Marker detection rate
   - Marker shape preservation (should remain square)
   - Area calculation accuracy (pixel-to-cm² conversion)

### Validation Checks

1. **Visual Inspection:**
   - Check augmented images preserve marker shape
   - Verify rare classes are not eliminated
   - Ensure realistic appearance

2. **Metric Monitoring:**
   - Track rare class metrics separately
   - Monitor for overfitting (train/val gap)
   - Check marker detection consistency

---

## Troubleshooting

### Problem: Rare Classes Not Detected

**Solutions:**
- Increase oversampling weight for rare class images
- Use focal loss instead of standard cross-entropy
- Reduce augmentation intensity for rare class images
- Add class-weighted loss

### Problem: Marker Detection Degrades

**Solutions:**
- Set `preserve_marker=True`
- Reduce geometric augmentation intensity
- Use `intensity="light"` instead of "moderate"
- Check marker annotations are correct

### Problem: Overfitting

**Solutions:**
- Increase augmentation intensity
- Add more augmentation transforms
- Use patch extraction to increase dataset size
- Reduce model capacity or add regularization

### Problem: Memory Issues

**Solutions:**
- Use patch extraction (smaller images)
- Reduce batch size
- Use gradient accumulation
- Reduce image size

---

## References

- Albumentations Documentation: https://albumentations.ai/
- COCO Evaluation: https://cocodataset.org/#detection-eval
- Medical Image Augmentation Best Practices: [Research papers on medical image augmentation]

---

## Quick Start Example

```python
from pipeline_utils import create_dataset, make_dataloaders
from scripts.augmentation_strategy import get_medical_augmentation_pipeline

# Create datasets with medical augmentation
train_dataset = create_dataset(
    root="data",
    annotation_file="../data/splits/train.json",
    train=True,
    image_size=(1024, 1024)
)

# Or use directly:
from scripts.augmentation_strategy import get_medical_augmentation_pipeline

train_transform = get_medical_augmentation_pipeline(
    train=True,
    image_size=(1024, 1024),
    preserve_marker=True,
    intensity="moderate"
)

val_transform = get_medical_augmentation_pipeline(
    train=False,
    image_size=(1024, 1024)
)

# Use in dataset
dataset = WoundDataset(root="data", annotation_file="train.json", transforms=train_transform)
```

---

**Last Updated:** 2025-01-13  
**Author:** AI Assistant (Expert in Medical Image Analysis)

