"""
Dataset Filtering and Validation Utilities

This module provides functions for filtering and validating colposcopy datasets,
including checks for size mismatches, empty masks, and repeat images.
"""

from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

def check_size_mismatch(
    image_path: Union[str, Path],
    mask_path: Union[str, Path],
    tolerance: int = 0
) -> Tuple[bool, Optional[Tuple[int, int]]]:
    """
    Check if image and mask have matching dimensions.
    
    Args:
        image_path: Path to the image file
        mask_path: Path to the mask file
        tolerance: Allowable difference in dimensions (pixels)
    
    Returns:
        Tuple[bool, Optional[Tuple]]: (is_mismatch, (image_size, mask_size))
    """
    try:
        with Image.open(image_path) as img:
            image_size = img.size
        
        with Image.open(mask_path) as mask:
            mask_size = mask.size
        
        # Check if sizes match within tolerance
        width_diff = abs(image_size[0] - mask_size[0])
        height_diff = abs(image_size[1] - mask_size[1])
        
        is_mismatch = width_diff > tolerance or height_diff > tolerance
        
        if is_mismatch:
            return True, (image_size, mask_size)
        else:
            return False, None
            
    except Exception as e:
        logger.error(f"Error checking size mismatch for {image_path}: {e}")
        return True, None

def check_empty_mask(
    mask_path: Union[str, Path],
    threshold: float = 0.01
) -> Tuple[bool, float]:
    """
    Check if a mask is essentially empty (very few non-zero pixels).
    
    Args:
        mask_path: Path to the mask file
        threshold: Minimum fraction of non-zero pixels required (0.01 = 1%)
    
    Returns:
        Tuple[bool, float]: (is_empty, non_zero_fraction)
    """
    try:
        with Image.open(mask_path) as mask:
            # Convert to numpy array
            mask_array = np.array(mask)
            
            # Count non-zero pixels
            total_pixels = mask_array.size
            non_zero_pixels = np.count_nonzero(mask_array)
            non_zero_fraction = non_zero_pixels / total_pixels
            
            is_empty = non_zero_fraction < threshold
            
            return is_empty, non_zero_fraction
            
    except Exception as e:
        logger.error(f"Error checking empty mask for {mask_path}: {e}")
        return True, 0.0

def check_repeat_images(
    image_paths: List[Union[str, Path]],
    similarity_threshold: float = 0.95
) -> Dict[str, List[str]]:
    """
    Check for duplicate or very similar images using perceptual hashing.
    
    Args:
        image_paths: List of image file paths to check
        similarity_threshold: Threshold for considering images similar
    
    Returns:
        Dict[str, List[str]]: Mapping of image to list of similar images
    """
    try:
        import imagehash
    except ImportError:
        logger.warning("imagehash not available, skipping repeat detection")
        return {}
    
    hash_dict = {}
    similar_groups = defaultdict(list)
    
    # Calculate perceptual hashes for all images
    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Calculate perceptual hash
                img_hash = imagehash.average_hash(img)
                hash_dict[str(img_path)] = img_hash
                
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            continue
    
    # Find similar images
    processed = set()
    for img_path1, hash1 in hash_dict.items():
        if img_path1 in processed:
            continue
            
        similar_images = [img_path1]
        
        for img_path2, hash2 in hash_dict.items():
            if img_path1 == img_path2 or img_path2 in processed:
                continue
                
            # Calculate hash similarity
            hash_diff = hash1 - hash2
            similarity = 1 - (hash_diff / 64.0)  # 64 is max difference for average_hash
            
            if similarity >= similarity_threshold:
                similar_images.append(img_path2)
        
        if len(similar_images) > 1:
            for img_path in similar_images:
                similar_groups[img_path] = similar_images
                processed.add(img_path)
    
    return dict(similar_groups)

def filter_dataset(
    csv_path: Union[str, Path],
    images_dir: Union[str, Path],
    masks_dir: Union[str, Path],
    output_csv_path: Optional[Union[str, Path]] = None,
    check_sizes: bool = True,
    check_empty_masks: bool = True,
    check_repeats: bool = False,
    size_tolerance: int = 0,
    empty_threshold: float = 0.01,
    repeat_threshold: float = 0.95
) -> Dict[str, any]:
    """
    Filter dataset based on various criteria and update CSV metadata.
    
    Args:
        csv_path: Path to the CSV file with dataset metadata
        images_dir: Directory containing images
        masks_dir: Directory containing masks
        output_csv_path: Path to save filtered CSV (if None, overwrites input)
        check_sizes: Whether to check for size mismatches
        check_empty_masks: Whether to check for empty masks
        check_repeats: Whether to check for repeat images
        size_tolerance: Tolerance for size mismatches
        empty_threshold: Threshold for empty mask detection
        repeat_threshold: Threshold for repeat image detection
    
    Returns:
        Dict: Statistics about the filtering process
    """
    csv_path = Path(csv_path)
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    
    if output_csv_path is None:
        output_csv_path = csv_path
    else:
        output_csv_path = Path(output_csv_path)
    
    # Load CSV data
    df = pd.read_csv(csv_path)
    initial_count = len(df)
    
    stats = {
        'initial_count': initial_count,
        'size_mismatches': 0,
        'empty_masks': 0,
        'repeat_images': 0,
        'missing_files': 0,
        'final_count': 0,
        'errors': []
    }
    
    # Track rows to drop
    rows_to_drop = []
    
    for idx, row in df.iterrows():
        # Get file paths
        if 'new_image_name' in row and pd.notna(row['new_image_name']):
            img_file = row['new_image_name']
            mask_file = row['new_mask_name']
        else:
            img_file = row['image_name']
            mask_file = row['mask_name']
        
        img_path = images_dir / img_file
        mask_path = masks_dir / mask_file
        
        # Check if files exist
        if not img_path.exists() or not mask_path.exists():
            stats['missing_files'] += 1
            rows_to_drop.append(idx)
            stats['errors'].append(f"Missing files for row {idx}: {img_file}, {mask_file}")
            continue
        
        # Check size mismatch
        if check_sizes:
            is_mismatch, sizes = check_size_mismatch(img_path, mask_path, size_tolerance)
            if is_mismatch:
                stats['size_mismatches'] += 1
                rows_to_drop.append(idx)
                if sizes:
                    stats['errors'].append(f"Size mismatch for {img_file}: image {sizes[0]}, mask {sizes[1]}")
        
        # Check empty mask
        if check_empty_masks:
            is_empty, fraction = check_empty_mask(mask_path, empty_threshold)
            if is_empty:
                stats['empty_masks'] += 1
                rows_to_drop.append(idx)
                stats['errors'].append(f"Empty mask for {img_file}: {fraction:.3f} non-zero pixels")
    
    # Check for repeat images (if requested)
    if check_repeats:
        image_paths = [images_dir / row['new_image_name'] if 'new_image_name' in row and pd.notna(row['new_image_name']) 
                      else images_dir / row['image_name'] for _, row in df.iterrows()]
        image_paths = [p for p in image_paths if p.exists()]
        
        repeat_groups = check_repeat_images(image_paths, repeat_threshold)
        
        # Mark repeat images for removal (keep first occurrence)
        for img_path, similar_images in repeat_groups.items():
            if len(similar_images) > 1:
                # Find rows with these images
                for similar_img in similar_images[1:]:  # Skip first (keep it)
                    similar_img_name = Path(similar_img).name
                    repeat_rows = df[df['new_image_name'] == similar_img_name].index
                    if len(repeat_rows) == 0:
                        repeat_rows = df[df['image_name'] == similar_img_name].index
                    
                    for row_idx in repeat_rows:
                        if row_idx not in rows_to_drop:
                            rows_to_drop.append(row_idx)
                            stats['repeat_images'] += 1
                            stats['errors'].append(f"Repeat image: {similar_img_name}")
    
    # Remove marked rows
    df_filtered = df.drop(rows_to_drop)
    stats['final_count'] = len(df_filtered)
    
    # Add drop column to original dataframe
    df['drop'] = 'keep'
    df.loc[rows_to_drop, 'drop'] = 'drop'
    
    # Save filtered CSV
    df.to_csv(output_csv_path, index=False)
    
    logger.info(f"Filtering complete:")
    logger.info(f"  Initial count: {stats['initial_count']}")
    logger.info(f"  Size mismatches: {stats['size_mismatches']}")
    logger.info(f"  Empty masks: {stats['empty_masks']}")
    logger.info(f"  Repeat images: {stats['repeat_images']}")
    logger.info(f"  Missing files: {stats['missing_files']}")
    logger.info(f"  Final count: {stats['final_count']}")
    
    return stats

def validate_dataset_integrity(
    csv_path: Union[str, Path],
    images_dir: Union[str, Path],
    masks_dir: Union[str, Path]
) -> Dict[str, any]:
    """
    Comprehensive validation of dataset integrity.
    
    Args:
        csv_path: Path to the CSV file
        images_dir: Directory containing images
        masks_dir: Directory containing masks
    
    Returns:
        Dict: Validation statistics
    """
    csv_path = Path(csv_path)
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    
    df = pd.read_csv(csv_path)
    
    validation_stats = {
        'total_rows': len(df),
        'valid_pairs': 0,
        'missing_images': 0,
        'missing_masks': 0,
        'size_mismatches': 0,
        'empty_masks': 0,
        'corrupted_files': 0,
        'errors': []
    }
    
    for idx, row in df.iterrows():
        # Get file paths
        if 'new_image_name' in row and pd.notna(row['new_image_name']):
            img_file = row['new_image_name']
            mask_file = row['new_mask_name']
        else:
            img_file = row['image_name']
            mask_file = row['mask_name']
        
        img_path = images_dir / img_file
        mask_path = masks_dir / mask_file
        
        # Check file existence
        if not img_path.exists():
            validation_stats['missing_images'] += 1
            validation_stats['errors'].append(f"Missing image: {img_file}")
            continue
            
        if not mask_path.exists():
            validation_stats['missing_masks'] += 1
            validation_stats['errors'].append(f"Missing mask: {mask_file}")
            continue
        
        # Check file integrity
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception as e:
            validation_stats['corrupted_files'] += 1
            validation_stats['errors'].append(f"Corrupted image {img_file}: {e}")
            continue
        
        try:
            with Image.open(mask_path) as mask:
                mask.verify()
        except Exception as e:
            validation_stats['corrupted_files'] += 1
            validation_stats['errors'].append(f"Corrupted mask {mask_file}: {e}")
            continue
        
        # Check size mismatch
        is_mismatch, _ = check_size_mismatch(img_path, mask_path)
        if is_mismatch:
            validation_stats['size_mismatches'] += 1
            validation_stats['errors'].append(f"Size mismatch: {img_file}")
            continue
        
        # Check empty mask
        is_empty, _ = check_empty_mask(mask_path)
        if is_empty:
            validation_stats['empty_masks'] += 1
            validation_stats['errors'].append(f"Empty mask: {mask_file}")
            continue
        
        validation_stats['valid_pairs'] += 1
    
    return validation_stats 