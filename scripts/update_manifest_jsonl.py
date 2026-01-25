"""
Update manifest.jsonl files with new image names
================================================

This script updates all manifest.jsonl files in data/task_*/data/ to reflect
the new standardized image names from rename_mapping.json.

Usage:
    cd scripts
    python update_manifest_jsonl.py [--dry-run]
"""

import sys
import json
from pathlib import Path
import logging
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('update_manifest_jsonl.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def normalize_path(path_str: str) -> str:
    """Normalize path string to use forward slashes."""
    return path_str.replace('\\', '/').replace('//', '/')


def load_rename_mapping(mapping_file: Path) -> dict:
    """Load rename mapping from JSON file."""
    if not mapping_file.exists():
        logger.error(f"Mapping file not found: {mapping_file}")
        return {}
    
    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    
    logger.info(f"Loaded {len(mapping)} mappings from {mapping_file}")
    return mapping


def create_reverse_mapping(mapping: dict) -> dict:
    """
    Create reverse mapping: old_name -> new_name (filename only, no path).
    
    Example:
        "task_0/data/2.jpg" -> "task_0_image_001.jpg"
        Returns: {"2.jpg": "task_0_image_001.jpg"}
    """
    reverse = {}
    for old_path, new_path in mapping.items():
        # Extract just the filename
        old_filename = Path(old_path).name
        new_filename = Path(new_path).name
        reverse[old_filename] = new_filename
    
    return reverse


def update_manifest_file(manifest_path: Path, reverse_mapping: dict, dry_run: bool = False) -> int:
    """
    Update a single manifest.jsonl file.
    
    Returns:
        Number of entries updated
    """
    if not manifest_path.exists():
        logger.warning(f"Manifest file not found: {manifest_path}")
        return 0
    
    try:
        # Read all lines
        with open(manifest_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        updated_lines = []
        updated_count = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                updated_lines.append(line)
                continue
            
            try:
                entry = json.loads(line)
                
                # Skip version and type lines
                if 'version' in entry or 'type' in entry:
                    updated_lines.append(line)
                    continue
                
                # Update name and extension if present
                if 'name' in entry and 'extension' in entry:
                    old_name = entry['name']
                    old_ext = entry['extension']
                    old_filename = f"{old_name}{old_ext}"
                    
                    # Check if we have a mapping for this filename
                    if old_filename in reverse_mapping:
                        new_filename = reverse_mapping[old_filename]
                        # Extract name and extension from new filename
                        new_name = Path(new_filename).stem
                        new_ext = Path(new_filename).suffix
                        
                        entry['name'] = new_name
                        entry['extension'] = new_ext
                        updated_count += 1
                        
                        if dry_run:
                            logger.info(f"[DRY RUN] Would update: {old_filename} -> {new_filename}")
                
                # Write updated entry
                updated_lines.append(json.dumps(entry, ensure_ascii=False))
            
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line in {manifest_path}: {e}")
                updated_lines.append(line)
                continue
        
        # Write updated file
        if not dry_run and updated_count > 0:
            with open(manifest_path, 'w', encoding='utf-8') as f:
                for line in updated_lines:
                    f.write(line + '\n')
            logger.info(f"Updated {updated_count} entries in {manifest_path.name}")
        
        return updated_count
    
    except Exception as e:
        logger.error(f"Failed to update {manifest_path}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description='Update manifest.jsonl files with new image names')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be updated without actually updating')
    
    args = parser.parse_args()
    
    # Get project root
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    data_root = project_root / 'data'
    mapping_file = project_root / 'rename_mapping.json'
    
    logger.info("=" * 80)
    logger.info("Starting manifest.jsonl update process")
    logger.info("=" * 80)
    logger.info(f"Data root: {data_root}")
    logger.info(f"Mapping file: {mapping_file}")
    logger.info(f"Dry run mode: {args.dry_run}")
    
    # Load mapping
    mapping = load_rename_mapping(mapping_file)
    if not mapping:
        logger.error("No mapping found! Exiting.")
        return 1
    
    # Create reverse mapping (filename only)
    reverse_mapping = create_reverse_mapping(mapping)
    logger.info(f"Created reverse mapping for {len(reverse_mapping)} filenames")
    
    # Find all manifest.jsonl files
    manifest_files = list(data_root.glob('task_*/data/manifest.jsonl'))
    logger.info(f"Found {len(manifest_files)} manifest.jsonl files")
    
    if not manifest_files:
        logger.warning("No manifest.jsonl files found!")
        return 1
    
    # Update each manifest file
    total_updated = 0
    files_updated = 0
    
    for manifest_file in sorted(manifest_files):
        updated_count = update_manifest_file(manifest_file, reverse_mapping, dry_run=args.dry_run)
        if updated_count > 0:
            total_updated += updated_count
            files_updated += 1
    
    logger.info("\n" + "=" * 80)
    if args.dry_run:
        logger.info(f"DRY RUN COMPLETE - Would update {total_updated} entries in {files_updated} files")
    else:
        logger.info(f"Update complete! Updated {total_updated} entries in {files_updated} files")
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
