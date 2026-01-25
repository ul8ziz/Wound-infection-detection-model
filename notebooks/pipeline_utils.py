import os
import json
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

# Try importing pycocotools
try:
    from pycocotools.coco import COCO
    HAS_COCO = True
except ImportError:
    HAS_COCO = False
    COCO = None

def set_seed(seed: int = 42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Returns the appropriate torch device.
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

class WoundDataset(Dataset):
    """
    Dataset for Wound Infection Detection.
    Expects COCO-style annotations and an image directory.
    """
    def __init__(self, root: str, annotation_file: str, transforms: Optional[A.Compose] = None):
        self.root = Path(root)
        self.transforms = transforms
        self.ann_file = annotation_file
        
        # Load annotations using COCO API if available, otherwise use dict
        if HAS_COCO:
            # Read file with UTF-8 encoding first to avoid UnicodeDecodeError
            # pycocotools.COCO uses default encoding which may fail on Windows
            with open(annotation_file, 'r', encoding='utf-8') as f:
                self.coco_json = json.load(f)
            
            # Create COCO object by monkey-patching open() to use UTF-8
            # This is necessary because pycocotools.COCO uses default encoding
            import builtins
            original_open = builtins.open
            
            def utf8_open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
                if 'r' in mode and encoding is None:
                    encoding = 'utf-8'
                return original_open(file, mode, buffering, encoding, errors, newline, closefd, opener)
            
            # Write full data to temp file with UTF-8 encoding
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.json', delete=False) as tmp_file:
                json.dump(self.coco_json, tmp_file, ensure_ascii=False)
                tmp_path = tmp_file.name
            
            try:
                # Monkey-patch open() temporarily
                builtins.open = utf8_open
                # Initialize COCO with temp file (now uses UTF-8)
                self.coco = COCO(tmp_path)
            finally:
                # Restore original open()
                builtins.open = original_open
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except:
                        pass
            
            # Ensure 'info' field exists for COCO API compatibility
            if hasattr(self.coco, 'dataset') and isinstance(self.coco.dataset, dict):
                if 'info' not in self.coco.dataset:
                    self.coco.dataset['info'] = {
                        "description": "Wound Infection Detection Dataset",
                        "version": "1.0",
                        "year": 2025
                    }
        else:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                self.coco_json = json.load(f)
            self.coco = self.coco_json
            
        # Build image and annotation mappings
        if HAS_COCO:
            self.images = {img['id']: img for img in self.coco_json['images']}
        else:
            self.images = {img['id']: img for img in self.coco['images']}
            
        self.img_to_anns = {}
        ann_list = self.coco_json['annotations'] if HAS_COCO else self.coco['annotations']
        for ann in ann_list:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
            
        self.ids = list(self.images.keys())

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_id = self.ids[index]
        img_info = self.images[img_id]
        
        # Construct image path
        # Assuming file_name in JSON is relative to root, e.g., task_0/data/image.jpg
        # If root is 'data/', and file_name is 'task_0/data/image.jpg', complete path is 'data/task_0/data/image.jpg'
        img_path = self.root / img_info['file_name']
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            # Fallback for missing images (should ideally not happen in clean dataset)
            print(f"Warning: Could not load image {img_path}. Returning empty tensor.")
            # Return a black image of expected size if possible, or error
            # For robustness, we'll return a black image of the size specified in annotations
            h = img_info.get('height', 512)
            w = img_info.get('width', 512)
            image = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get annotations
        anns = self.img_to_anns.get(img_id, [])
        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []
        
        img_h, img_w = image.shape[:2]

        for ann in anns:
            # COCO bbox: [x, y, w, h] in pixel coordinates
            x, y, w, h = ann['bbox']
            # Skip invalid boxes
            if w <= 0 or h <= 0:
                continue
            
            # Clamp bbox to image bounds first
            x = max(0.0, min(x, img_w - 1))
            y = max(0.0, min(y, img_h - 1))
            # Ensure width and height don't exceed image bounds
            w = min(w, img_w - x)
            h = min(h, img_h - y)
            
            # Skip if box is invalid after clamping
            if w <= 0 or h <= 0:
                continue
            
            # Pass boxes in absolute pixel coordinates (COCO format: [x, y, w, h])
            # Albumentations with format='coco' expects pixel coordinates, NOT normalized
            boxes.append([x, y, w, h])
            labels.append(ann['category_id'])
            areas.append(ann['area'])
            iscrowd.append(ann.get('iscrowd', 0))
            
            # Create mask
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            for seg in ann['segmentation']:
                poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [poly], 1)
            masks.append(mask)

        # Apply transforms
        if self.transforms:
            if len(boxes) > 0:
                try:
                    transformed = self.transforms(
                        image=image,
                        bboxes=boxes,
                        labels=labels,
                        masks=masks
                    )
                    image = transformed['image']
                    boxes = transformed['bboxes']
                    labels = transformed['labels']
                    masks = transformed['masks']
                except Exception as e:
                    print(f"Transform failed for {img_path}: {e}")
                    # Skip boxes that caused the error - filter out invalid ones
                    # Re-validate boxes before retrying
                    valid_boxes = []
                    valid_labels = []
                    valid_masks = []
                    valid_areas = []
                    valid_iscrowd = []
                    
                    for i, box in enumerate(boxes):
                        # Ensure box has exactly 4 values
                        if len(box) != 4:
                            continue
                        try:
                            x, y, w, h = box[:4]
                        except (ValueError, TypeError):
                            continue
                        
                        # Validate pixel coordinates (boxes are in pixel format from Albumentations)
                        if w > 0 and h > 0 and x >= 0 and y >= 0:
                            # Clamp to ensure they're valid
                            x = max(0.0, float(x))
                            y = max(0.0, float(y))
                            w = max(0.0, float(w))
                            h = max(0.0, float(h))
                            
                            if w > 0 and h > 0:
                                valid_boxes.append([x, y, w, h])
                                if i < len(labels):
                                    valid_labels.append(labels[i])
                                if i < len(masks):
                                    valid_masks.append(masks[i])
                                if i < len(areas):
                                    valid_areas.append(areas[i])
                                if i < len(iscrowd):
                                    valid_iscrowd.append(iscrowd[i])
                    
                    boxes = valid_boxes
                    labels = valid_labels
                    masks = valid_masks
                    areas = valid_areas
                    iscrowd = valid_iscrowd
                    
                    # Fallback: apply minimal transform (resize + normalize + to tensor)
                    # Get target size from the transform
                    target_size = (512, 512)
                    if hasattr(self.transforms, 'transforms') and len(self.transforms.transforms) > 0:
                        first_transform = self.transforms.transforms[0]
                        if hasattr(first_transform, 'height') and hasattr(first_transform, 'width'):
                            target_size = (first_transform.height, first_transform.width)
                    
                    fallback_transform = A.Compose([
                        A.Resize(height=target_size[0], width=target_size[1]),
                        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ToTensorV2()
                    ])
                    
                    # Apply fallback transform
                    if len(boxes) > 0:
                        try:
                            transformed = fallback_transform(
                                image=image,
                                bboxes=boxes,
                                labels=labels,
                                masks=masks
                            )
                            image = transformed['image']
                            boxes = transformed['bboxes']
                            labels = transformed['labels']
                            masks = transformed['masks']
                        except:
                            # Last resort: transform image only
                            transformed = fallback_transform(image=image)
                            image = transformed['image']
                            boxes = []
                            labels = []
                            masks = []
                    else:
                        transformed = fallback_transform(image=image)
                        image = transformed['image']
            else:
                # No boxes, just transform image
                transformed = self.transforms(
                    image=image,
                    bboxes=boxes,
                    labels=labels,
                    masks=masks
                )
                image = transformed['image']
                boxes = transformed['bboxes']
                labels = transformed['labels']
                masks = transformed['masks']

        # Convert to torch tensors and standard format
        # Image should be Tensor from ToTensorV2, but ensure it's float32
        if not isinstance(image, torch.Tensor):
            # Convert numpy array to tensor
            if isinstance(image, np.ndarray):
                if image.dtype == np.uint8:
                    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                else:
                    image = torch.from_numpy(image).permute(2, 0, 1).float()
            else:
                image = ToTensorV2()(image=image)['image']
        
        # Ensure image is float32 and in correct format [C, H, W]
        if image.dtype != torch.float32:
            if image.dtype == torch.uint8:
                image = image.float() / 255.0
            else:
                image = image.float()
        
        # Ensure image is in [C, H, W] format
        if len(image.shape) == 3 and image.shape[0] != 3:
            # If it's [H, W, C], permute to [C, H, W]
            if image.shape[2] == 3:
                image = image.permute(2, 0, 1)
        
        # Get transformed image dimensions
        if isinstance(image, torch.Tensor):
            _, new_h, new_w = image.shape
        else:
            new_h, new_w = image.shape[:2]

        target = {}
        target["image_id"] = torch.tensor([img_id])

        if len(boxes) > 0:
            # Filter out invalid boxes first
            valid_boxes_list = []
            valid_labels_list = []
            valid_masks_list = []
            valid_areas_list = []
            valid_iscrowd_list = []
            
            for i, box in enumerate(boxes):
                if len(box) != 4:
                    continue
                try:
                    x, y, w, h = box[:4]
                except (ValueError, TypeError):
                    continue
                
                # Validate pixel coordinates (Albumentations returns boxes in pixel coordinates)
                x = float(x)
                y = float(y)
                w = float(w)
                h = float(h)
                
                # Ensure valid dimensions
                if w > 0 and h > 0:
                    # Clamp to image bounds
                    x = max(0.0, min(x, new_w - 1))
                    y = max(0.0, min(y, new_h - 1))
                    w = min(w, new_w - x)
                    h = min(h, new_h - y)
                    
                    if w > 0 and h > 0:
                        valid_boxes_list.append([x, y, w, h])
                        if i < len(labels):
                            valid_labels_list.append(labels[i])
                        if i < len(masks):
                            valid_masks_list.append(masks[i])
                        if i < len(areas):
                            valid_areas_list.append(areas[i])
                        if i < len(iscrowd):
                            valid_iscrowd_list.append(iscrowd[i])
            
            boxes = valid_boxes_list
            labels = valid_labels_list
            masks = valid_masks_list
            areas = valid_areas_list
            iscrowd = valid_iscrowd_list
        
        if len(boxes) > 0:
            # Albumentations returns bboxes in 'coco' format [x, y, w, h] in PIXEL coordinates
            # Convert directly to xyxy format (no scaling needed)
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            
            # Convert xywh (COCO) -> xyxy format
            boxes_xyxy = boxes_t.clone()
            boxes_xyxy[:, 2] = boxes_t[:, 0] + boxes_t[:, 2]  # x2 = x + w
            boxes_xyxy[:, 3] = boxes_t[:, 1] + boxes_t[:, 3]  # y2 = y + h
            
            # Clamp to image bounds
            boxes_xyxy[:, 0] = boxes_xyxy[:, 0].clamp(0, new_w)
            boxes_xyxy[:, 1] = boxes_xyxy[:, 1].clamp(0, new_h)
            boxes_xyxy[:, 2] = boxes_xyxy[:, 2].clamp(0, new_w)
            boxes_xyxy[:, 3] = boxes_xyxy[:, 3].clamp(0, new_h)
            
            target["boxes"] = boxes_xyxy
            target["labels"] = torch.tensor(labels, dtype=torch.int64)
            
            if len(masks) > 0:
                 # Masks list of (H, W) -> (N, H, W)
                 # Albumentations returns list of masks (already resized)
                 if isinstance(masks, list):
                     if len(masks) > 0 and isinstance(masks[0], torch.Tensor):
                         target["masks"] = torch.stack(masks)
                     elif len(masks) > 0:
                         # Convert to tensor and ensure correct shape
                         mask_array = np.array(masks)
                         if mask_array.ndim == 2:
                             mask_array = mask_array[np.newaxis, :, :]
                         target["masks"] = torch.as_tensor(mask_array, dtype=torch.uint8)
                     else:
                          target["masks"] = torch.zeros((0, new_h, new_w), dtype=torch.uint8)
                 else:
                     # Already tensor from transforms
                     if isinstance(masks, torch.Tensor):
                         target["masks"] = masks
                     else:
                         target["masks"] = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            
            target["area"] = torch.tensor(areas, dtype=torch.float32)
            target["iscrowd"] = torch.tensor(iscrowd, dtype=torch.int64)
            
        else:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["masks"] = torch.zeros((0, new_h, new_w), dtype=torch.uint8)
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        # Final validation: ensure image is float32
        if not isinstance(image, torch.Tensor):
            raise ValueError(f"Image is not a tensor: {type(image)}")
        if image.dtype != torch.float32:
            image = image.float()
            if image.dtype == torch.uint8:
                image = image / 255.0
        
        return image, target

def get_transforms(
    train: bool = True, 
    image_size: Tuple[int, int] = (512, 512),
    use_medical_augmentation: bool = False,
    preserve_marker: bool = True,
    intensity: str = "moderate"
):
    """
    Returns albumentations transforms.
    
    Args:
        train: If True, applies training augmentations
        image_size: Target (height, width) for resizing
        use_medical_augmentation: If True, uses comprehensive medical augmentation strategy
        preserve_marker: If True, limits transforms that could distort marker geometry
        intensity: Augmentation intensity ("light", "moderate", "aggressive")
    """
    if use_medical_augmentation:
        # Use comprehensive medical augmentation strategy
        try:
            # Try importing from scripts directory
            import sys
            from pathlib import Path
            project_root = Path(__file__).parent.parent
            scripts_dir = project_root / "scripts"
            if (scripts_dir / "augmentation_strategy.py").exists():
                sys.path.insert(0, str(scripts_dir))
            from augmentation_strategy import get_medical_augmentation_pipeline
            return get_medical_augmentation_pipeline(
                train=train,
                image_size=image_size,
                preserve_marker=preserve_marker,
                intensity=intensity
            )
        except ImportError:
            print("Warning: augmentation_strategy not found, using default transforms")
            # Fall through to default
    
    # Default/legacy transforms
    if train:
        return A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(translate_percent=0.1, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
    else:
        return A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))

def create_dataset(
    root: str, 
    annotation_file: str, 
    train: bool = True, 
    image_size: Tuple[int, int] = (512, 512),
    use_medical_augmentation: bool = False,
    preserve_marker: bool = True,
    intensity: str = "moderate"
) -> Dataset:
    """
    Factory function to create the dataset.
    
    Args:
        root: Root directory for images
        annotation_file: Path to annotation file
        train: If True, applies training augmentations
        image_size: Target (height, width) for resizing
        use_medical_augmentation: If True, uses comprehensive medical augmentation strategy
        preserve_marker: If True, limits transforms that could distort marker geometry
        intensity: Augmentation intensity ("light", "moderate", "aggressive")
    """
    transforms = get_transforms(
        train=train,
        image_size=image_size,
        use_medical_augmentation=use_medical_augmentation,
        preserve_marker=preserve_marker,
        intensity=intensity
    )
    dataset = WoundDataset(root, annotation_file, transforms)
    return dataset

def collate_fn(batch):
    """
    Custom collate fn for handling batches of images and varying size targets.
    """
    return tuple(zip(*batch))

def make_dataloaders(
    train_dataset: Dataset, 
    val_dataset: Dataset, 
    batch_size: int = 4, 
    num_workers: int = 2,
    shuffle_train: bool = True
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader

def verify_dataset_sample(dataset: Dataset, num_samples: int = 1):
    """
    Prints info about the first few samples to verify correctness.
    """
    print(f"Dataset length: {len(dataset)}")
    for i in range(min(len(dataset), num_samples)):
        img, target = dataset[i]
        print(f"Sample {i}:")
        print(f"  Image shape: {img.shape}")
        print(f"  Target keys: {target.keys()}")
        if "boxes" in target:
            print(f"  Boxes shape: {target['boxes'].shape}")
        if "masks" in target:
            print(f"  Masks shape: {target['masks'].shape}")
        if "labels" in target:
            print(f"  Labels: {target['labels']}")

