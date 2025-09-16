"""
Filename Utilities for Dataset Management

This module provides functions for consistent filename formatting, parsing, and
management for colposcopy images and segmentation masks.
"""

import re
import pandas as pd
from pathlib import Path
from typing import Union, Tuple, List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

def generate_sequential_filenames(
    start_index: int = 1,
    count: int = 100,
    padding: int = 4,
    prefix: str = "",
    suffix: str = ".png",
    mask_suffix: str = "_mask"
) -> Tuple[List[str], List[str]]:
    """
    Generate sequential filenames for images and masks.
    
    Args:
        start_index: Starting index for filenames
        count: Number of filenames to generate
        padding: Number of zeros to pad the index
        prefix: Optional prefix for filenames
        suffix: File extension
        mask_suffix: Suffix to add for mask files
    
    Returns:
        Tuple[List[str], List[str]]: (image_filenames, mask_filenames)
    """
    image_filenames = []
    mask_filenames = []
    
    for i in range(start_index, start_index + count):
        # Format index with padding
        index_str = f"{i:0{padding}d}"
        
        # Generate image filename
        image_filename = f"{prefix}{index_str}{suffix}"
        image_filenames.append(image_filename)
        
        # Generate mask filename
        mask_filename = f"{prefix}{index_str}{mask_suffix}{suffix}"
        mask_filenames.append(mask_filename)
    
    return image_filenames, mask_filenames

def parse_filename_to_index(
    filename: Union[str, Path],
    pattern: Optional[str] = None
) -> Optional[int]:
    """
    Parse a filename to extract the sequential index.
    
    Args:
        filename: Filename to parse
        pattern: Optional regex pattern for parsing (default: r'(\d+)')
    
    Returns:
        Optional[int]: Extracted index, or None if not found
    """
    if pattern is None:
        # Default pattern: extract digits from filename
        pattern = r'(\d+)'
    
    filename_str = str(filename)
    
    match = re.search(pattern, filename_str)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            logger.warning(f"Could not convert matched digits to integer: {match.group(1)}")
            return None
    
    return None

def format_filename_with_padding(
    index: int,
    padding: int = 4,
    prefix: str = "",
    suffix: str = ".png"
) -> str:
    """
    Format an index into a padded filename.
    
    Args:
        index: Index number
        padding: Number of zeros to pad
        prefix: Optional prefix
        suffix: File extension
    
    Returns:
        str: Formatted filename
    """
    index_str = f"{index:0{padding}d}"
    return f"{prefix}{index_str}{suffix}"

def get_image_mask_pair_names(
    image_filename: Union[str, Path],
    mask_suffix: str = "_mask"
) -> Tuple[str, str]:
    """
    Get the corresponding mask filename for an image filename.
    
    Args:
        image_filename: Image filename
        mask_suffix: Suffix to add for mask files
    
    Returns:
        Tuple[str, str]: (image_filename, mask_filename)
    """
    image_path = Path(image_filename)
    
    # Extract name without extension
    name_without_ext = image_path.stem
    
    # Create mask filename
    mask_filename = f"{name_without_ext}{mask_suffix}{image_path.suffix}"
    
    return str(image_filename), mask_filename

def validate_filename_pattern(
    filename: Union[str, Path],
    expected_pattern: str = r'^\d{4}\.png$'
) -> bool:
    """
    Validate if a filename matches an expected pattern.
    
    Args:
        filename: Filename to validate
        expected_pattern: Regex pattern for expected format
    
    Returns:
        bool: True if filename matches pattern
    """
    filename_str = str(filename)
    return bool(re.match(expected_pattern, filename_str))

def rename_files_sequentially(
    source_dir: Union[str, Path],
    target_dir: Union[str, Path],
    start_index: int = 1,
    padding: int = 4,
    file_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png'),
    dry_run: bool = False
) -> Dict[str, any]:
    """
    Rename files in a directory with sequential numbering.
    
    Args:
        source_dir: Source directory containing files
        target_dir: Target directory for renamed files
        start_index: Starting index for new filenames
        padding: Number of zeros to pad
        file_extensions: File extensions to process
        dry_run: If True, don't actually rename files
    
    Returns:
        Dict: Statistics about the renaming process
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    if not source_dir.exists():
        raise ValueError(f"Source directory does not exist: {source_dir}")
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {
        'total_files': 0,
        'renamed_files': 0,
        'skipped_files': 0,
        'errors': [],
        'mapping': {}
    }
    
    # Get all files with specified extensions
    files_to_rename = []
    for ext in file_extensions:
        files_to_rename.extend(source_dir.glob(f"*{ext}"))
        files_to_rename.extend(source_dir.glob(f"*{ext.upper()}"))
    
    files_to_rename.sort()
    stats['total_files'] = len(files_to_rename)
    
    for i, file_path in enumerate(files_to_rename):
        try:
            # Generate new filename
            new_index = start_index + i
            new_filename = format_filename_with_padding(new_index, padding, suffix=file_path.suffix)
            new_path = target_dir / new_filename
            
            # Store mapping
            stats['mapping'][file_path.name] = new_filename
            
            if not dry_run:
                # Copy file to new location with new name
                import shutil
                shutil.copy2(file_path, new_path)
                logger.info(f"Renamed: {file_path.name} -> {new_filename}")
            else:
                logger.info(f"Would rename: {file_path.name} -> {new_filename}")
            
            stats['renamed_files'] += 1
            
        except Exception as e:
            error_msg = f"Error renaming {file_path.name}: {e}"
            stats['errors'].append(error_msg)
            logger.error(error_msg)
            stats['skipped_files'] += 1
    
    logger.info(f"Renaming complete:")
    logger.info(f"  Total files: {stats['total_files']}")
    logger.info(f"  Renamed files: {stats['renamed_files']}")
    logger.info(f"  Skipped files: {stats['skipped_files']}")
    logger.info(f"  Errors: {len(stats['errors'])}")
    
    return stats

def create_filename_mapping(
    old_filenames: List[Union[str, Path]],
    new_filenames: List[Union[str, Path]]
) -> Dict[str, str]:
    """
    Create a mapping between old and new filenames.
    
    Args:
        old_filenames: List of old filenames
        new_filenames: List of new filenames
    
    Returns:
        Dict[str, str]: Mapping of old filename to new filename
    """
    if len(old_filenames) != len(new_filenames):
        raise ValueError("Old and new filename lists must have the same length")
    
    mapping = {}
    for old_name, new_name in zip(old_filenames, new_filenames):
        mapping[str(old_name)] = str(new_name)
    
    return mapping

def update_csv_filenames(
    csv_path: Union[str, Path],
    filename_mapping: Dict[str, str],
    image_column: str = 'image_name',
    mask_column: str = 'mask_name',
    output_path: Optional[Union[str, Path]] = None
) -> 'pd.DataFrame':
    """
    Update CSV file with new filenames based on a mapping.
    
    Args:
        csv_path: Path to CSV file
        filename_mapping: Mapping of old to new filenames
        image_column: Name of column containing image filenames
        mask_column: Name of column containing mask filenames
        output_path: Optional path to save updated CSV
    
    Returns:
        pd.DataFrame: Updated DataFrame
    """
    
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    
    # Update image filenames
    if image_column in df.columns:
        df[image_column] = df[image_column].map(filename_mapping).fillna(df[image_column])
    
    # Update mask filenames
    if mask_column in df.columns:
        df[mask_column] = df[mask_column].map(filename_mapping).fillna(df[mask_column])
    
    # Save updated CSV
    if output_path is not None:
        output_path = Path(output_path)
        df.to_csv(output_path, index=False)
        logger.info(f"Updated CSV saved to: {output_path}")
    
    return df

def verify_filename_consistency(
    images_dir: Union[str, Path],
    masks_dir: Union[str, Path],
    expected_pattern: str = r'^\d{4}\.png$',
    mask_suffix: str = "_mask"
) -> Dict[str, any]:
    """
    Verify that image and mask filenames are consistent.
    
    Args:
        images_dir: Directory containing images
        masks_dir: Directory containing masks
        expected_pattern: Regex pattern for expected filename format
        mask_suffix: Expected suffix for mask files
    
    Returns:
        Dict: Verification statistics
    """
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    
    verification_stats = {
        'total_images': 0,
        'total_masks': 0,
        'matching_pairs': 0,
        'orphaned_images': 0,
        'orphaned_masks': 0,
        'invalid_format_images': 0,
        'invalid_format_masks': 0,
        'errors': []
    }
    
    # Get all image files
    image_files = [f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    mask_files = [f for f in masks_dir.iterdir() if f.is_file() and f.suffix.lower() == '.png']
    
    verification_stats['total_images'] = len(image_files)
    verification_stats['total_masks'] = len(mask_files)
    
    # Check each image file
    for img_file in image_files:
        # Validate format
        if not validate_filename_pattern(img_file.name, expected_pattern):
            verification_stats['invalid_format_images'] += 1
            verification_stats['errors'].append(f"Invalid image format: {img_file.name}")
            continue
        
        # Find corresponding mask
        expected_mask_name = f"{img_file.stem}{mask_suffix}{img_file.suffix}"
        expected_mask_path = masks_dir / expected_mask_name
        
        if expected_mask_path.exists():
            verification_stats['matching_pairs'] += 1
        else:
            verification_stats['orphaned_images'] += 1
            verification_stats['errors'].append(f"Missing mask for image: {img_file.name}")
    
    # Check for orphaned masks
    for mask_file in mask_files:
        # Validate format
        if not validate_filename_pattern(mask_file.name, expected_pattern.replace('.png$', f'{mask_suffix}\\.png$')):
            verification_stats['invalid_format_masks'] += 1
            verification_stats['errors'].append(f"Invalid mask format: {mask_file.name}")
            continue
        
        # Check if corresponding image exists
        expected_img_name = mask_file.name.replace(mask_suffix, '')
        expected_img_path = images_dir / expected_img_name
        
        if not expected_img_path.exists():
            verification_stats['orphaned_masks'] += 1
            verification_stats['errors'].append(f"Missing image for mask: {mask_file.name}")
    
    return verification_stats 