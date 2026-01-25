"""
Image Renaming Script
====================

This script renames all images in data/task_*/data/ to a standardized format:
task_X_image_XXX.ext (e.g., task_0_image_001.jpg, task_1_image_001.jpg)

It also updates all JSON files (annotations.json, splits/*.json) to reflect the new names.

Usage:
    cd scripts
    python rename_all_images.py [--dry-run] [--no-backup]

Options:
    --dry-run: Show what would be renamed without actually renaming
    --no-backup: Skip creating backup files (not recommended)
"""

import sys
import os
from pathlib import Path
import json
import shutil
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rename_all_images.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Image extensions to process
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

# JSON files to update (relative to project root)
JSON_FILES_TO_UPDATE = [
    'data/annotations.json',
    'data/splits/train.json',
    'data/splits/val.json',
    'data/splits/test.json'
]


def normalize_path(path_str: str) -> str:
    """
    Normalize path string to use forward slashes (for JSON consistency).
    Windows paths use backslashes, but we'll store them with forward slashes.
    Also handles both formats for matching.
    """
    return path_str.replace('\\', '/').replace('//', '/')


def get_task_number_from_path(path: Path) -> Optional[int]:
    """
    Extract task number from path like 'data/task_0/data/image.jpg' -> 0
    """
    parts = path.parts
    for part in parts:
        if part.startswith('task_') and part[5:].isdigit():
            return int(part[5:])
    return None


def collect_all_images(data_root: Path) -> List[Tuple[Path, int]]:
    """
    Collect all image files from task_*/data/ directories.
    
    Returns:
        List of tuples (image_path, task_number)
    """
    images = []
    data_root = Path(data_root).resolve()
    
    if not data_root.exists():
        logger.error(f"Data root does not exist: {data_root}")
        return images
    
    # Find all task directories
    task_dirs = sorted([d for d in data_root.iterdir() 
                       if d.is_dir() and d.name.startswith('task_')])
    
    logger.info(f"Found {len(task_dirs)} task directories")
    
    for task_dir in task_dirs:
        task_num = get_task_number_from_path(task_dir)
        if task_num is None:
            logger.warning(f"Could not extract task number from: {task_dir}")
            continue
        
        data_dir = task_dir / 'data'
        if not data_dir.exists():
            logger.warning(f"Data directory does not exist: {data_dir}")
            continue
        
        # Find all image files
        for img_file in data_dir.iterdir():
            if img_file.is_file() and img_file.suffix in IMAGE_EXTENSIONS:
                images.append((img_file, task_num))
    
    logger.info(f"Found {len(images)} image files total")
    return images


def create_rename_mapping(images: List[Tuple[Path, int]], data_root: Path) -> Dict[str, str]:
    """
    Create mapping from old names to new names.
    
    Args:
        images: List of (image_path, task_number) tuples
        data_root: Root data directory (e.g., project_root/data)
    
    Returns:
        Dictionary mapping old_path -> new_path (both normalized with forward slashes)
    """
    mapping = {}
    data_root = Path(data_root).resolve()
    
    # Group images by task number
    task_images = {}
    for img_path, task_num in images:
        if task_num not in task_images:
            task_images[task_num] = []
        task_images[task_num].append(img_path)
    
    # Sort images within each task by original filename
    for task_num in sorted(task_images.keys()):
        task_imgs = sorted(task_images[task_num], key=lambda p: p.name)
        
        for idx, img_path in enumerate(task_imgs, start=1):
            # Get original extension
            ext = img_path.suffix.lower()
            if ext == '.jpeg':
                ext = '.jpg'  # Normalize .jpeg to .jpg for consistency
            
            # Create new name: task_X_image_XXX.ext
            new_name = f"task_{task_num}_image_{idx:03d}{ext}"
            new_path = img_path.parent / new_name
            
            # Create mapping (relative to data_root)
            try:
                old_path_rel = img_path.relative_to(data_root)
                new_path_rel = new_path.relative_to(data_root)
                
                # Normalize paths (use forward slashes)
                old_path_str = normalize_path(str(old_path_rel))
                new_path_str = normalize_path(str(new_path_rel))
                
                mapping[old_path_str] = new_path_str
            except ValueError as e:
                logger.warning(f"Could not compute relative path for {img_path}: {e}")
                continue
    
    logger.info(f"Created mapping for {len(mapping)} images")
    return mapping


def backup_json_files(json_files: List[str], backup_dir: Path) -> bool:
    """
    Create backup copies of JSON files.
    
    Returns:
        True if all backups successful, False otherwise
    """
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    success = True
    for json_file in json_files:
        json_path = Path(json_file)
        if not json_path.exists():
            logger.warning(f"JSON file does not exist: {json_path}")
            continue
        
        backup_path = backup_dir / f"{json_path.stem}_{timestamp}{json_path.suffix}"
        try:
            shutil.copy2(json_path, backup_path)
            logger.info(f"Backed up: {json_path.name} -> {backup_path.name}")
        except Exception as e:
            logger.error(f"Failed to backup {json_path}: {e}")
            success = False
    
    return success


def rename_images(mapping: Dict[str, str], data_root: Path, dry_run: bool = False) -> bool:
    """
    Rename all image files according to the mapping.
    
    Returns:
        True if all renames successful, False otherwise
    """
    data_root = Path(data_root).resolve()
    success = True
    renamed_count = 0
    
    for old_path_str, new_path_str in mapping.items():
        # Convert normalized paths back to actual paths
        old_path = data_root / old_path_str
        new_path = data_root / new_path_str
        
        if not old_path.exists():
            logger.warning(f"Image not found: {old_path}")
            continue
        
        if new_path.exists() and old_path != new_path:
            logger.warning(f"Target already exists: {new_path}")
            continue
        
        if dry_run:
            logger.info(f"[DRY RUN] Would rename: {old_path.name} -> {new_path.name}")
        else:
            try:
                # Rename in place
                old_path.rename(new_path)
                renamed_count += 1
                if renamed_count % 100 == 0:
                    logger.info(f"Renamed {renamed_count} images...")
            except Exception as e:
                logger.error(f"Failed to rename {old_path}: {e}")
                success = False
    
    if not dry_run:
        logger.info(f"Successfully renamed {renamed_count} images")
    
    return success


def update_json_file(json_path: Path, mapping: Dict[str, str], dry_run: bool = False) -> bool:
    """
    Update file_name fields in a JSON file using the mapping.
    
    Returns:
        True if update successful, False otherwise
    """
    if not json_path.exists():
        logger.warning(f"JSON file does not exist: {json_path}")
        return False
    
    try:
        # Read JSON with UTF-8 encoding
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        updated_count = 0
        
        # Update file_name in images array
        if 'images' in data:
            for img in data['images']:
                if 'file_name' in img:
                    old_file_name = normalize_path(img['file_name'])
                    if old_file_name in mapping:
                        new_file_name = mapping[old_file_name]
                        # Keep forward slashes for consistency (works on both Windows and Linux)
                        img['file_name'] = new_file_name
                        updated_count += 1
                    else:
                        # Try with backslashes if forward slashes didn't match
                        old_file_name_alt = img['file_name'].replace('/', '\\')
                        if old_file_name_alt in mapping:
                            new_file_name = mapping[old_file_name_alt]
                            img['file_name'] = new_file_name
                            updated_count += 1
        
        if dry_run:
            logger.info(f"[DRY RUN] Would update {updated_count} entries in {json_path.name}")
            return True
        
        # Write updated JSON with UTF-8 encoding
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Updated {updated_count} entries in {json_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update {json_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Rename all images to standardized format')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be renamed without actually renaming')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip creating backup files (not recommended)')
    
    args = parser.parse_args()
    
    # Get project root (parent of scripts directory)
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    data_root = project_root / 'data'
    
    logger.info("=" * 80)
    logger.info("Starting image renaming process")
    logger.info("=" * 80)
    logger.info(f"Data root: {data_root}")
    logger.info(f"Dry run mode: {args.dry_run}")
    
    # Step 1: Collect all images
    logger.info("\nStep 1: Collecting all images...")
    images = collect_all_images(data_root)
    
    if not images:
        logger.error("No images found! Exiting.")
        return 1
    
    # Step 2: Create rename mapping
    logger.info("\nStep 2: Creating rename mapping...")
    mapping = create_rename_mapping(images, data_root)
    
    # Save mapping to file
    mapping_file = project_root / 'rename_mapping.json'
    if not args.dry_run:
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved mapping to: {mapping_file}")
    
    # Step 3: Backup JSON files
    if not args.no_backup and not args.dry_run:
        logger.info("\nStep 3: Creating backups...")
        backup_dir = project_root / 'backups' / 'json_backups'
        # Convert relative paths to absolute paths
        json_files_abs = [project_root / f for f in JSON_FILES_TO_UPDATE]
        if not backup_json_files([str(f) for f in json_files_abs], backup_dir):
            logger.warning("Some backups failed, but continuing...")
    
    # Step 4: Rename images
    logger.info("\nStep 4: Renaming images...")
    if not rename_images(mapping, data_root, dry_run=args.dry_run):
        logger.error("Some image renames failed!")
        return 1
    
    # Step 5: Update JSON files
    logger.info("\nStep 5: Updating JSON files...")
    json_files = [project_root / f for f in JSON_FILES_TO_UPDATE]
    for json_file in json_files:
        if json_file.exists():
            update_json_file(json_file, mapping, dry_run=args.dry_run)
        else:
            logger.warning(f"JSON file not found (skipping): {json_file}")
    
    logger.info("\n" + "=" * 80)
    if args.dry_run:
        logger.info("DRY RUN COMPLETE - No files were actually modified")
    else:
        logger.info("Image renaming complete!")
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
