"""
Mask Binarization Utilities

This module converts RGB-labeled lesion masks to binary masks (0 = background, 1 = lesion)
for DeepLabv3+ training. It identifies specific RGB colors that represent different
lesion types and converts them to binary format.
"""

import os
from PIL import Image
import numpy as np
from pathlib import Path
import logging
from typing import List, Tuple, Union

logger = logging.getLogger(__name__)

# RGB colors representing different lesion types
LESION_COLORS = [
    (255, 0, 255),   # Purple - Aceto-white
    (255, 0, 0),     # Red - Atypical vessels  
    (97, 7, 35),     # Brown - Mosaics
]

def convert_mask_to_binary(mask_array: np.ndarray) -> np.ndarray:
    """
    Convert RGB mask array to binary mask.
    
    Args:
        mask_array: RGB mask as numpy array with shape (H, W, 3)
        
    Returns:
        Binary mask as numpy array with shape (H, W) where 1 = lesion, 0 = background
    """
    if len(mask_array.shape) != 3 or mask_array.shape[2] != 3:
        raise ValueError(f"Expected RGB mask with shape (H, W, 3), got {mask_array.shape}")
    
    binary_mask = np.zeros(mask_array.shape[:2], dtype=np.uint8)
    
    # Set pixels matching any lesion color to 1
    for color in LESION_COLORS:
        match = np.all(mask_array == color, axis=-1)
        binary_mask[match] = 1
    
    return binary_mask

def process_mask_file(mask_path: Union[str, Path], output_path: Union[str, Path]) -> bool:
    """
    Process a single mask file from RGB to binary.
    
    Args:
        mask_path: Path to input RGB mask
        output_path: Path to save binary mask
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load RGB mask
        mask = Image.open(mask_path).convert("RGB")
        mask_array = np.array(mask)
        
        # Convert to binary
        binary_mask = convert_mask_to_binary(mask_array)
        
        # Save as binary image (0 or 255 for visualization)
        binary_image = Image.fromarray(binary_mask * 255, mode='L')
        binary_image.save(output_path)
        
        logger.debug(f"Converted {mask_path} -> {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {mask_path}: {e}")
        return False

def process_split(split: str, 
                 input_dir: Union[str, Path] = None,
                 output_dir: Union[str, Path] = None) -> Tuple[int, int]:
    """
    Process all masks in a dataset split.
    
    Args:
        split: Dataset split name ('train', 'val', 'test')
        input_dir: Input directory containing RGB masks (optional)
        output_dir: Output directory for binary masks (optional)
        
    Returns:
        Tuple[int, int]: (successful_count, failed_count)
    """
    if input_dir is None:
        input_dir = Path(f"dataset/dataset_split/{split}/masks")
    if output_dir is None:
        output_dir = Path(f"dataset/dataset_split/{split}/masks_binary")
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all PNG mask files
    mask_files = list(input_dir.glob("*.png"))
    
    if not mask_files:
        logger.warning(f"No PNG files found in {input_dir}")
        return 0, 0
    
    successful_count = 0
    failed_count = 0
    
    logger.info(f"Processing {len(mask_files)} masks for {split} split...")
    
    for mask_path in mask_files:
        output_path = output_dir / mask_path.name
        
        if process_mask_file(mask_path, output_path):
            successful_count += 1
        else:
            failed_count += 1
    
    logger.info(f"{split} split: {successful_count} successful, {failed_count} failed")
    return successful_count, failed_count

def process_all_splits(splits: List[str] = None) -> dict:
    """
    Process all dataset splits.
    
    Args:
        splits: List of splits to process (default: ['train', 'val', 'test'])
        
    Returns:
        dict: Statistics for each split
    """
    if splits is None:
        splits = ["train", "val", "test"]
    
    results = {}
    
    for split in splits:
        logger.info(f"Processing {split} split...")
        successful, failed = process_split(split)
        results[split] = {
            'successful': successful,
            'failed': failed,
            'total': successful + failed
        }
    
    return results

def verify_binary_masks(split: str, 
                       masks_dir: Union[str, Path] = None) -> dict:
    """
    Verify that binary masks are correctly generated.
    
    Args:
        split: Dataset split name
        masks_dir: Directory containing binary masks
        
    Returns:
        dict: Verification statistics
    """
    if masks_dir is None:
        masks_dir = Path(f"dataset/dataset_split/{split}/masks_binary")
    
    masks_dir = Path(masks_dir)
    
    if not masks_dir.exists():
        logger.error(f"Binary masks directory not found: {masks_dir}")
        return {}
    
    stats = {
        'total_files': 0,
        'valid_binary': 0,
        'invalid_binary': 0,
        'errors': []
    }
    
    for mask_path in masks_dir.glob("*.png"):
        stats['total_files'] += 1
        
        try:
            mask = Image.open(mask_path).convert("L")
            mask_array = np.array(mask)
            
            # Check if mask contains only 0 and 255 values
            unique_values = np.unique(mask_array)
            if set(unique_values).issubset({0, 255}):
                stats['valid_binary'] += 1
            else:
                stats['invalid_binary'] += 1
                stats['errors'].append(f"{mask_path.name}: contains values {unique_values}")
                
        except Exception as e:
            stats['invalid_binary'] += 1
            stats['errors'].append(f"{mask_path.name}: {e}")
    
    logger.info(f"Binary mask verification for {split}: {stats['valid_binary']}/{stats['total_files']} valid")
    return stats


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Process all splits
    results = process_all_splits()
    
    # Print summary
    print("\n" + "="*50)
    print("BINARY MASK GENERATION SUMMARY")
    print("="*50)
    
    total_successful = 0
    total_failed = 0
    
    for split, stats in results.items():
        print(f"{split.upper()}: {stats['successful']} successful, {stats['failed']} failed")
        total_successful += stats['successful']
        total_failed += stats['failed']
    
    print(f"\nTOTAL: {total_successful} successful, {total_failed} failed")
    
    # Verify results
    print("\n" + "="*50)
    print("VERIFICATION")
    print("="*50)
    
    for split in ["train", "val", "test"]:
        verify_stats = verify_binary_masks(split)
        if verify_stats:
            print(f"{split.upper()}: {verify_stats['valid_binary']}/{verify_stats['total_files']} valid binary masks") 