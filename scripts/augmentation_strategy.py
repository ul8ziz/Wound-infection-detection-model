"""
Comprehensive Data Augmentation Strategy for Medical Wound Segmentation

This module provides a medically-appropriate augmentation pipeline for wound
segmentation tasks, with special consideration for:
- Preserving marker geometry (critical for area measurements)
- Handling class imbalance
- Safe geometric and photometric transforms
- Multi-task CVAT dataset structure
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Optional, Dict, List
import numpy as np
import cv2


# ============================================================================
# DATASET ANALYSIS
# ============================================================================

"""
DATASET CHARACTERISTICS ANALYSIS:

1. STRUCTURE:
   - ~240 separate CVAT tasks (task_0 to task_239)
   - Each task has its own annotations.json with polygon labels
   - Multi-class, multi-instance segmentation (polygons per image)
   - High-resolution clinical photos (variable sizes)

2. CLASS IMBALANCE:
   - Frequent classes: "ВсяРана" (entire wound), "Метка для размерности" (marker)
   - Moderate classes: "Зона шва", "Зона отека вокруг раны", "Зона гиперемии вокруг"
   - Rare classes: "Зона некроза", "Гнойное отделяемое", "Сухожилие", "Губка ВАК"
   - Background dominates (most of image is not wound)

3. CRITICAL CONSTRAINTS:
   - Marker geometry MUST be preserved (3×3 cm square used for area conversion)
   - Extreme warping would invalidate pixel-to-cm² conversion
   - Medical images require realistic augmentations (no unrealistic distortions)
   - Small structures (necrosis, granulation) need careful handling

4. VARIABILITY IN DATA:
   - Lighting: varying clinical lighting conditions
   - Pose: different camera angles and distances
   - Background: complex backgrounds (skin, bandages, medical equipment)
   - Scale: wounds vary significantly in size
   - Quality: some images may have motion blur or focus issues
"""


# ============================================================================
# AUGMENTATION PIPELINE DESIGN
# ============================================================================

def get_medical_augmentation_pipeline(
    train: bool = True,
    image_size: Tuple[int, int] = (1024, 1024),
    preserve_marker: bool = True,
    intensity: str = "moderate"  # "light", "moderate", "aggressive"
) -> A.Compose:
    """
    Returns a medically-appropriate augmentation pipeline for wound segmentation.
    
    Args:
        train: If True, applies training augmentations; otherwise validation (minimal)
        image_size: Target (height, width) for resizing
        preserve_marker: If True, limits transforms that could distort marker geometry
        intensity: Augmentation intensity level
        
    Returns:
        Albumentations Compose object with appropriate transforms
    """
    
    if not train:
        # Validation: minimal transforms (resize + normalize only)
        return A.Compose([
            A.Resize(height=image_size[0], width=image_size[1], interpolation=cv2.INTER_LINEAR),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
    
    # Training augmentations
    transforms = []
    
    # ========================================================================
    # STEP 1: RESIZE (always first)
    # ========================================================================
    # Use LongestMaxSize to preserve aspect ratio, then pad to square
    # This prevents distortion of the marker
    transforms.append(
        A.LongestMaxSize(
            max_size=max(image_size),
            interpolation=cv2.INTER_LINEAR,
            p=1.0
        )
    )
    
    # Pad to exact size if needed (maintains aspect ratio)
    transforms.append(
        A.PadIfNeeded(
            min_height=image_size[0],
            min_width=image_size[1],
            border_mode=cv2.BORDER_CONSTANT,
            p=1.0
        )
    )
    
    # ========================================================================
    # STEP 2: GEOMETRIC TRANSFORMS (safe for medical segmentation)
    # ========================================================================
    
    if preserve_marker:
        # CONSERVATIVE geometric transforms to preserve marker geometry
        # Small rotations and translations are acceptable
        # Avoid strong perspective/elastic deformations
        
        # Horizontal flip (safe, preserves marker shape)
        transforms.append(
            A.HorizontalFlip(p=0.5)
        )
        
        # Vertical flip (less common in medical images, but acceptable)
        transforms.append(
            A.VerticalFlip(p=0.2)  # Lower probability
        )
        
        # Small rotations (max ±10 degrees to preserve marker square shape)
        transforms.append(
            A.Rotate(
                limit=10,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            )
        )
        
        # Small affine transforms (translation + scale + rotation)
        # Limited scale to preserve marker size ratio
        transforms.append(
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},  # Max 5% translation
                scale=(0.95, 1.05),  # Max ±5% scale (preserves marker size)
                rotate=(-8, 8),  # Small rotation
                shear=(-3, 3),  # Minimal shear
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.4
            )
        )
        
    else:
        # More aggressive geometric transforms (use with caution)
        # Only if marker preservation is not critical
        transforms.append(A.HorizontalFlip(p=0.5))
        transforms.append(A.VerticalFlip(p=0.3))
        transforms.append(
            A.Rotate(
                limit=15,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.5
            )
        )
        transforms.append(
            A.Affine(
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                shear=(-5, 5),
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.5
            )
        )
    
    # ========================================================================
    # STEP 3: PHOTOMETRIC TRANSFORMS (safe for medical images)
    # ========================================================================
    
    if intensity == "light":
        # Light augmentations (minimal changes)
        transforms.extend([
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                brightness_by_max=True,
                p=0.5
            ),
            A.GaussNoise(
                std_range=(0.02, 0.05),
                p=0.2
            )
        ])
        
    elif intensity == "moderate":
        # Moderate augmentations (recommended default)
        transforms.extend([
            # Brightness/Contrast (simulates lighting variations)
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                brightness_by_max=True,
                p=0.6
            ),
            
            # Gamma correction (simulates exposure variations)
            A.RandomGamma(
                gamma_limit=(80, 120),  # 0.8 to 1.2 in normalized scale
                p=0.4
            ),
            
            # Color jitter (limited - medical images should remain realistic)
            A.HueSaturationValue(
                hue_shift_limit=5,  # Small hue shifts (preserves tissue colors)
                sat_shift_limit=10,  # Moderate saturation
                val_shift_limit=10,  # Value shifts
                p=0.3
            ),
            
            # Gaussian noise (simulates sensor noise)
            A.GaussNoise(
                std_range=(0.03, 0.08),
                p=0.3
            ),
            
            # Small blur (simulates slight focus issues)
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
            ], p=0.2),
            
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # Useful for medical images with varying contrast
            A.CLAHE(
                clip_limit=2.0,
                tile_grid_size=(8, 8),
                p=0.3
            )
        ])
        
    elif intensity == "aggressive":
        # Aggressive augmentations (use with caution)
        transforms.extend([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                brightness_by_max=True,
                p=0.7
            ),
            A.RandomGamma(
                gamma_limit=(70, 130),
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.4
            ),
            A.GaussNoise(
                std_range=(0.05, 0.12),
                p=0.4
            ),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.3),
            A.CLAHE(
                clip_limit=3.0,
                tile_grid_size=(8, 8),
                p=0.4
            ),
            # Channel shuffle (very aggressive, use sparingly)
            A.ChannelShuffle(p=0.1)
        ])
    
    # ========================================================================
    # STEP 4: NORMALIZATION & TENSOR CONVERSION (always last)
    # ========================================================================
    transforms.extend([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats (standard for ResNet)
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])
    
    # Compose with bbox and mask parameters
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='coco',
            label_fields=['labels'],
            min_visibility=0.1  # Filter out boxes that become too small
        ),
        additional_targets={'mask': 'mask'}  # Support multiple masks
    )


# ============================================================================
# TRANSFORMS TO AVOID (and why)
# ============================================================================

"""
TRANSFORMS THAT SHOULD NOT BE USED:

1. ElasticTransform / GridDistortion / OpticalDistortion:
   - WHY: These create non-linear deformations that would distort the 3×3 cm marker
   - IMPACT: Would invalidate pixel-to-cm² area conversion
   - ALTERNATIVE: Use small affine transforms instead

2. Perspective / PiecewiseAffine:
   - WHY: Strong perspective changes would distort marker geometry
   - IMPACT: Marker would no longer be a square, breaking area calculations
   - ALTERNATIVE: Use small rotations and translations

3. RandomCrop (without careful handling):
   - WHY: Could crop out critical structures (marker, wound center)
   - IMPACT: Loss of important annotations
   - ALTERNATIVE: Use CenterCrop or wound-aware cropping (see below)

4. Cutout / CoarseDropout (on masks):
   - WHY: Medical segmentation requires complete masks
   - IMPACT: Incomplete annotations would confuse the model
   - ALTERNATIVE: Can use on images only (not masks) for robustness

5. Strong ColorJitter / Posterize:
   - WHY: Medical images should remain realistic
   - IMPACT: Unrealistic colors could hurt generalization
   - ALTERNATIVE: Use limited hue/saturation shifts
"""


# ============================================================================
# ADVANCED: WOUND-AWARE CROPPING
# ============================================================================

def get_wound_aware_crop_transform(
    image_size: Tuple[int, int] = (1024, 1024),
    crop_scale: Tuple[float, float] = (0.7, 1.0),
    min_wound_coverage: float = 0.3
) -> A.Compose:
    """
    Advanced cropping that attempts to keep wound regions in view.
    
    This is a placeholder for a more sophisticated implementation that would:
    1. Detect wound bounding boxes
    2. Prefer crops that contain wound regions
    3. Ensure marker is included if present
    
    For now, returns a conservative random crop.
    """
    return A.Compose([
        A.RandomResizedCrop(
            height=image_size[0],
            width=image_size[1],
            scale=crop_scale,
            ratio=(0.8, 1.2),  # Slightly wider or taller
            interpolation=cv2.INTER_LINEAR,
            p=0.5  # Apply only 50% of the time
        ),
        # ... rest of pipeline
    ])


# ============================================================================
# CLASS-BALANCED SAMPLING STRATEGY
# ============================================================================

def compute_class_weights(annotations: Dict) -> Dict[int, float]:
    """
    Computes class weights for imbalanced dataset.
    
    Args:
        annotations: COCO-style annotation dict
        
    Returns:
        Dictionary mapping category_id -> weight
    """
    from collections import Counter
    
    # Count instances per class
    class_counts = Counter()
    for ann in annotations['annotations']:
        class_counts[ann['category_id']] += 1
    
    # Compute inverse frequency weights
    total = sum(class_counts.values())
    weights = {}
    for cat_id, count in class_counts.items():
        weights[cat_id] = total / (len(class_counts) * count)
    
    return weights


def create_balanced_sampler(
    dataset,
    annotations: Dict,
    rare_class_threshold: float = 0.05,
    rare_class_weight: float = 3.0
):
    """
    Creates a WeightedRandomSampler that oversamples images containing rare classes.
    
    Args:
        dataset: Dataset object with .ids and .img_to_anns attributes
        annotations: COCO-style annotation dict
        rare_class_threshold: Fraction of total annotations below which a class is considered rare
        rare_class_weight: Weight multiplier for images containing rare classes
        
    Returns:
        WeightedRandomSampler instance
    """
    from collections import Counter
    from torch.utils.data import WeightedRandomSampler
    
    # Count class frequencies
    class_counts = Counter()
    for ann in annotations['annotations']:
        class_counts[ann['category_id']] += 1
    
    total_count = sum(class_counts.values())
    rare_class_ids = {
        cat_id for cat_id, count in class_counts.items()
        if count < total_count * rare_class_threshold
    }
    
    # Assign weights to samples
    sample_weights = []
    for idx in range(len(dataset)):
        img_id = dataset.ids[idx]
        anns = dataset.img_to_anns.get(img_id, [])
        
        # Check if image contains rare classes
        has_rare = any(ann['category_id'] in rare_class_ids for ann in anns)
        weight = rare_class_weight if has_rare else 1.0
        sample_weights.append(weight)
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


# ============================================================================
# PATCH EXTRACTION STRATEGY
# ============================================================================

def extract_patches_from_image(
    image: np.ndarray,
    masks: List[np.ndarray],
    patch_size: Tuple[int, int] = (512, 512),
    stride: Tuple[int, int] = (256, 256),
    min_mask_coverage: float = 0.1
) -> List[Tuple[np.ndarray, List[np.ndarray]]]:
    """
    Extracts patches from high-resolution images, ensuring each patch
    contains at least some annotation coverage.
    
    This is useful for:
    - Training on high-res images without full-image memory issues
    - Creating more training samples from limited data
    - Focusing on wound regions
    
    Args:
        image: Input image (H, W, C)
        masks: List of binary masks (H, W)
        patch_size: Size of extracted patches
        stride: Stride for sliding window
        min_mask_coverage: Minimum fraction of patch that must contain annotations
        
    Returns:
        List of (patch_image, patch_masks) tuples
    """
    patches = []
    h, w = image.shape[:2]
    ph, pw = patch_size
    
    # Create combined mask (any annotation)
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for mask in masks:
        combined_mask = np.maximum(combined_mask, mask)
    
    # Sliding window extraction
    for y in range(0, h - ph + 1, stride[0]):
        for x in range(0, w - pw + 1, stride[1]):
            patch = image[y:y+ph, x:x+pw]
            
            # Check mask coverage
            mask_patch = combined_mask[y:y+ph, x:x+pw]
            coverage = np.sum(mask_patch > 0) / (ph * pw)
            
            if coverage >= min_mask_coverage:
                # Extract corresponding masks
                patch_masks = [m[y:y+ph, x:x+pw] for m in masks]
                patches.append((patch, patch_masks))
    
    return patches


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example: Create augmentation pipeline
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
    
    print("Training pipeline created with", len(train_transform.transforms), "transforms")
    print("Validation pipeline created with", len(val_transform.transforms), "transforms")

