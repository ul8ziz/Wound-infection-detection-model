"""
Data Augmentation Script (Standalone)
=====================================

This script applies data augmentation to the dataset and saves the augmented images
and annotations. It runs ONCE to generate augmented data that can be used for training.

The training itself is done in: notebooks/1.8.2025/fixed_training_pipeline.ipynb

Usage:
    cd scripts
    python apply_augmentation_only.py

Output:
    - Augmented images saved to: ../data/augmented/images/
    - New annotations file: ../data/augmented/annotations_augmented.json
"""

import sys
import os
from pathlib import Path
import json
import cv2
import numpy as np
from tqdm import tqdm
import shutil
from typing import Dict, List, Tuple, Optional
import random

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
# Also add notebooks directory for pipeline_utils and training_engine
notebooks_dir = current_dir.parent / "notebooks"
if notebooks_dir.exists():
    sys.path.insert(0, str(notebooks_dir))

from augmentation_strategy import get_medical_augmentation_pipeline
import albumentations as A
import torch

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    # Input paths (relative to project root, not script location)
    "data_root": "../data",  # Original data root (data/ in project root)
    "annotation_file": "../data/splits/train.json",  # Training annotations (data/splits/train.json)
    
    # Output paths
    "output_root": "../data/augmented",  # Where to save augmented data (inside data/)
    "output_images_dir": "images",  # Augmented images directory (relative to output_root)
    "output_annotations": "annotations_augmented.json",  # New annotations file (relative to output_root)
    
    # Augmentation settings
    "augmentations_per_image": 3,  # How many augmented versions per original image
    "image_size": (512, 512),  # Target image size
    "preserve_marker": True,  # Preserve marker geometry
    "intensity": "moderate",  # "light", "moderate", "aggressive"
    
    # Other settings
    "seed": 42,
    "copy_original": True,  # Copy original images to output (without augmentation)
}

# ============================================================================
# Helper Functions
# ============================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_annotations(annotation_file: str) -> Dict:
    """Load COCO format annotations."""
    annotation_path = Path(annotation_file)
    if not annotation_path.is_absolute():
        # Resolve relative to script location
        script_dir = Path(__file__).parent
        annotation_path = (script_dir / annotation_path).resolve()
    
    with open(annotation_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_image(image_path: str) -> np.ndarray:
    """Load image from path."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_image(image: np.ndarray, output_path: str):
    """Save image to path."""
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), img_bgr)

def convert_bbox_xyxy_to_coco(bbox_xyxy: List[float]) -> List[float]:
    """Convert bbox from [x1, y1, x2, y2] to COCO format [x, y, w, h]."""
    x1, y1, x2, y2 = bbox_xyxy
    return [x1, y1, x2 - x1, y2 - y1]

def convert_bbox_coco_to_xyxy(bbox_coco: List[float]) -> List[float]:
    """Convert bbox from COCO format [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = bbox_coco
    return [x, y, x + w, y + h]

def apply_augmentation(
    image: np.ndarray,
    bboxes: List[List[float]],
    masks: List[np.ndarray],
    labels: List[int],
    transform: A.Compose
) -> Tuple[np.ndarray, List[List[float]], List[np.ndarray], List[int]]:
    """
    Apply augmentation to image, bboxes, masks, and labels.
    
    Args:
        image: RGB image (H, W, 3)
        bboxes: List of bboxes in COCO format [x, y, w, h]
        masks: List of binary masks (H, W)
        labels: List of category IDs
        transform: Albumentations transform
        
    Returns:
        Augmented image, bboxes, masks, labels
    """
    # Prepare data for Albumentations
    # Bboxes should be in COCO format [x, y, w, h] in pixel coordinates
    # Clamp bboxes to image bounds before augmentation
    h, w = image.shape[:2]
    clamped_bboxes = []
    for bbox in bboxes:
        x, y, bw, bh = bbox
        # Clamp to image bounds
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        bw = max(1, min(bw, w - x))
        bh = max(1, min(bh, h - y))
        clamped_bboxes.append([x, y, bw, bh])
    
    try:
        transformed = transform(
            image=image,
            bboxes=clamped_bboxes,
            masks=masks,
            category_ids=labels
        )
    except Exception as e:
        # If augmentation fails, return original data
        print(f"Warning: Augmentation failed: {e}")
        return image, bboxes, masks, labels
    
    # Get augmented image (should be numpy array, not tensor)
    aug_image = transformed['image']
    
    # Ensure uint8 format (image should already be uint8 from Albumentations)
    if aug_image.dtype != np.uint8:
        if aug_image.max() <= 1.0:
            aug_image = (aug_image * 255).astype(np.uint8)
        else:
            aug_image = aug_image.astype(np.uint8)
    
    # Clamp bboxes to new image bounds after augmentation
    aug_h, aug_w = aug_image.shape[:2]
    clamped_aug_bboxes = []
    for bbox in transformed['bboxes']:
        x, y, bw, bh = bbox
        # Clamp to image bounds
        x = max(0, min(x, aug_w - 1))
        y = max(0, min(y, aug_h - 1))
        bw = max(1, min(bw, aug_w - x))
        bh = max(1, min(bh, aug_h - y))
        clamped_aug_bboxes.append([x, y, bw, bh])
    
    # Convert masks from tensor to numpy if needed
    aug_masks = []
    for mask in transformed['masks']:
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        # Ensure binary mask and correct size
        if mask.shape != (aug_h, aug_w):
            # Resize mask if needed
            mask = cv2.resize(mask.astype(np.uint8), (aug_w, aug_h), interpolation=cv2.INTER_NEAREST)
        if mask.dtype != np.uint8:
            mask = (mask > 0.5).astype(np.uint8)
        aug_masks.append(mask)
    
    return (
        aug_image,
        clamped_aug_bboxes,
        aug_masks,
        transformed['category_ids']
    )

def process_image(
    image_info: Dict,
    annotations: Dict,
    data_root: Path,
    output_images_dir: Path,
    transform: A.Compose,
    augmentations_per_image: int,
    copy_original: bool
) -> List[Dict]:
    """
    Process a single image: load, augment, and save.
    
    Returns:
        List of new image_info dictionaries for augmented images
    """
    # Load original image
    image_path = data_root / image_info['file_name']
    if not image_path.exists():
        print(f"Warning: Image not found: {image_path}")
        return []
    
    image = load_image(str(image_path))
    original_h, original_w = image.shape[:2]
    
    # Get annotations for this image
    image_id = image_info['id']
    anns = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]
    
    # Prepare bboxes and masks
    bboxes = []
    masks = []
    labels = []
    
    for ann in anns:
        # Bbox in COCO format [x, y, w, h]
        bbox = ann['bbox']
        bboxes.append(bbox)
        labels.append(ann['category_id'])
        
        # Load mask (if segmentation exists)
        if 'segmentation' in ann and ann['segmentation']:
            # Convert RLE or polygon to mask
            from pycocotools import mask as mask_util
            if isinstance(ann['segmentation'], dict):
                # RLE format
                mask = mask_util.decode(ann['segmentation'])
            else:
                # Polygon format - convert to mask
                rle = mask_util.frPyObjects(ann['segmentation'], original_h, original_w)
                mask = mask_util.decode(rle)
                if len(mask.shape) == 3:
                    mask = mask.sum(axis=2) > 0
            masks.append(mask.astype(np.uint8))
        else:
            # Create mask from bbox if no segmentation
            mask = np.zeros((original_h, original_w), dtype=np.uint8)
            x, y, w, h = [int(v) for v in bbox]
            mask[y:y+h, x:x+w] = 1
            masks.append(mask)
    
    new_image_infos = []
    
    # Copy original image if requested
    if copy_original:
        # Extract just the filename from the original path
        original_filename = Path(image_info['file_name']).name
        original_output_path = output_images_dir / original_filename
        save_image(image, str(original_output_path))
        
        new_image_info = image_info.copy()
        new_image_info['id'] = len(new_image_infos) + 1
        # Update file_name to point to images/ directory
        new_image_info['file_name'] = f"images/{original_filename}"
        new_image_infos.append(new_image_info)
    
    # Apply augmentations
    for aug_idx in range(augmentations_per_image):
        try:
            # Apply augmentation
            aug_image, aug_bboxes, aug_masks, aug_labels = apply_augmentation(
                image, bboxes, masks, labels, transform
            )
            
            # Image should already be uint8 from apply_augmentation
            
            # Save augmented image
            # Extract base filename from original path
            original_filename = Path(image_info['file_name']).name
            aug_filename = f"{Path(original_filename).stem}_aug{aug_idx+1}{Path(original_filename).suffix}"
            aug_output_path = output_images_dir / aug_filename
            save_image(aug_image, str(aug_output_path))
            
            # Create new image info
            new_image_info = {
                'id': len(new_image_infos) + 1,
                'file_name': f"images/{aug_filename}",  # Point to images/ subdirectory
                'width': aug_image.shape[1],
                'height': aug_image.shape[0]
            }
            new_image_infos.append((new_image_info, aug_bboxes, aug_masks, aug_labels))
            
        except Exception as e:
            print(f"Error augmenting image {image_info['file_name']} (aug {aug_idx+1}): {e}")
            continue
    
    return new_image_infos

# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main function to apply augmentation and save results."""
    
    print("=" * 80)
    print("Data Augmentation Script (Standalone)")
    print("=" * 80)
    print()
    
    # Set seed
    set_seed(CONFIG["seed"])
    
    # Resolve paths relative to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent  # Go up from scripts/ to project root
    data_root = (project_root / CONFIG["data_root"].lstrip("../")).resolve()
    annotation_file = (project_root / CONFIG["annotation_file"].lstrip("../")).resolve()
    
    output_root = (project_root / CONFIG["output_root"].lstrip("../")).resolve()
    output_images_dir = output_root / CONFIG["output_images_dir"]
    output_annotations_path = output_root / CONFIG["output_annotations"]
    
    # Create output directories
    output_images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Data root: {data_root}")
    print(f"Annotation file: {annotation_file}")
    print(f"Output directory: {output_root}")
    print(f"Augmentations per image: {CONFIG['augmentations_per_image']}")
    print(f"Intensity: {CONFIG['intensity']}")
    print(f"Preserve marker: {CONFIG['preserve_marker']}")
    print()
    
    # Load annotations
    print("Loading annotations...")
    annotations = load_annotations(annotation_file)
    print(f"Loaded {len(annotations['images'])} images")
    print(f"Loaded {len(annotations['annotations'])} annotations")
    print()
    
    # Create augmentation pipeline (without ToTensorV2 and Normalize for saving images)
    print("Creating augmentation pipeline...")
    
    # Import augmentation components directly
    import cv2
    from albumentations.pytorch import ToTensorV2
    
    # Build augmentation pipeline manually (without ToTensorV2 and Normalize)
    transforms_list = []
    
    # Resize transforms
    transforms_list.append(
        A.LongestMaxSize(
            max_size=max(CONFIG["image_size"]),
            interpolation=cv2.INTER_LINEAR,
            p=1.0
        )
    )
    transforms_list.append(
        A.PadIfNeeded(
            min_height=CONFIG["image_size"][0],
            min_width=CONFIG["image_size"][1],
            border_mode=cv2.BORDER_CONSTANT,
            fill_value=0,
            mask_fill_value=0,
            p=1.0
        )
    )
    
    # Geometric transforms (if preserve_marker is True, use conservative settings)
    if CONFIG["preserve_marker"]:
        transforms_list.append(A.HorizontalFlip(p=0.5))
        transforms_list.append(A.VerticalFlip(p=0.2))
        transforms_list.append(
            A.Rotate(
                limit=10,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                fill_value=0,
                mask_fill_value=0,
                p=0.5
            )
        )
        transforms_list.append(
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.95, 1.05),
                rotate=(-8, 8),
                shear=(-3, 3),
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                fill_value=0,
                mask_fill_value=0,
                p=0.4
            )
        )
    else:
        transforms_list.append(A.HorizontalFlip(p=0.5))
        transforms_list.append(A.VerticalFlip(p=0.3))
        transforms_list.append(
            A.Rotate(
                limit=15,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                fill_value=0,
                mask_fill_value=0,
                p=0.5
            )
        )
        transforms_list.append(
            A.Affine(
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                shear=(-5, 5),
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                fill_value=0,
                mask_fill_value=0,
                p=0.5
            )
        )
    
    # Photometric transforms based on intensity
    intensity = CONFIG["intensity"]
    if intensity == "light":
        transforms_list.append(A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3))
        transforms_list.append(A.GaussNoise(std_range=(0.02, 0.05), p=0.2))
    elif intensity == "moderate":
        transforms_list.append(A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5))
        transforms_list.append(A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3))
        transforms_list.append(A.GaussNoise(std_range=(0.03, 0.08), p=0.3))
        transforms_list.append(A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3))
    else:  # aggressive
        transforms_list.append(A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6))
        transforms_list.append(A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4))
        transforms_list.append(A.GaussNoise(std_range=(0.05, 0.12), p=0.4))
        transforms_list.append(A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.4))
    
    # Create transform without ToTensorV2 and Normalize
    transform = A.Compose(
        transforms_list,
        bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
        additional_targets={}
    )
    print("Augmentation pipeline created (without ToTensorV2/Normalize for saving).")
    print()
    
    # Process images
    print("Processing images...")
    new_images = []
    new_annotations = []
    new_image_id = 1
    new_ann_id = 1
    
    for image_info in tqdm(annotations['images'], desc="Augmenting images"):
        new_image_infos = process_image(
            image_info,
            annotations,
            data_root,
            output_images_dir,
            transform,
            CONFIG["augmentations_per_image"],
            CONFIG["copy_original"]
        )
        
        # Add new images
        for item in new_image_infos:
            if isinstance(item, tuple):
                # Augmented image with updated bboxes/masks
                new_img_info, aug_bboxes, aug_masks, aug_labels = item
                new_img_info['id'] = new_image_id
                new_images.append(new_img_info)
                
                # Create annotations with updated bboxes and masks
                for i, (bbox, mask, label) in enumerate(zip(aug_bboxes, aug_masks, aug_labels)):
                    new_ann = {
                        'id': new_ann_id,
                        'image_id': new_image_id,
                        'category_id': int(label),
                        'bbox': bbox,  # Already in COCO format [x, y, w, h]
                        'area': bbox[2] * bbox[3],  # w * h
                        'iscrowd': 0
                    }
                    # Convert mask to RLE for COCO format
                    try:
                        from pycocotools import mask as mask_util
                        rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
                        rle['counts'] = rle['counts'].decode('utf-8')
                        new_ann['segmentation'] = rle
                    except:
                        # Fallback: use bbox as segmentation
                        new_ann['segmentation'] = []
                    new_annotations.append(new_ann)
                    new_ann_id += 1
            else:
                # Original image (no augmentation)
                new_img_info = item
                new_img_info['id'] = new_image_id
                new_images.append(new_img_info)
                
                # Copy original annotations
                image_id = image_info['id']
                for ann in annotations['annotations']:
                    if ann['image_id'] == image_id:
                        new_ann = ann.copy()
                        new_ann['id'] = new_ann_id
                        new_ann['image_id'] = new_image_id
                        new_annotations.append(new_ann)
                        new_ann_id += 1
            
            new_image_id += 1
    
    # Create new annotations structure
    new_annotations_dict = {
        'info': annotations.get('info', {}),
        'licenses': annotations.get('licenses', []),
        'categories': annotations['categories'],
        'images': new_images,
        'annotations': new_annotations
    }
    
    # Save new annotations
    print()
    print("Saving augmented annotations...")
    with open(output_annotations_path, 'w', encoding='utf-8') as f:
        json.dump(new_annotations_dict, f, indent=2, ensure_ascii=False)
    
    print()
    print("=" * 80)
    print("Augmentation Complete!")
    print("=" * 80)
    print(f"Original images: {len(annotations['images'])}")
    print(f"Augmented images: {len(new_images)}")
    print(f"Total images: {len(new_images)}")
    print(f"Augmented annotations saved to: {output_annotations_path}")
    print(f"Augmented images saved to: {output_images_dir}")
    print()
    print("You can now use the augmented data in your training notebook:")
    print("  notebooks/1.8.2025/fixed_training_pipeline.ipynb")
    print()
    print(f"Update the annotation file path to: {output_annotations_path}")

if __name__ == "__main__":
    import torch
    main()

