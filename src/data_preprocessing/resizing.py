"""
Image and Mask Resizing Utilities

This module provides functions for resizing colposcopy images and segmentation masks
while preserving aspect ratio and maintaining pixel-wise alignment.
"""

from PIL import Image, ImageOps
import numpy as np
from pathlib import Path
from typing import Tuple, Union, Optional
import logging

logger = logging.getLogger(__name__)

def resize_with_padding(
    img: Union[Image.Image, np.ndarray, str, Path], 
    size: Tuple[int, int] = (512, 512), 
    is_mask: bool = False,
    output_path: Optional[Union[str, Path]] = None
) -> Image.Image:
    """
    Resize image/mask while preserving aspect ratio and padding.
    
    Args:
        img: Input image as PIL Image, numpy array, or path to image file
        size: Target size as (width, height) tuple
        is_mask: Whether the image is a segmentation mask (affects interpolation)
        output_path: Optional path to save the resized image
    
    Returns:
        PIL Image: Resized image with padding
        
    Raises:
        ValueError: If size is not a valid tuple
        OSError: If image file cannot be opened
    """
    if not isinstance(size, tuple) or len(size) != 2:
        raise ValueError("size must be a tuple of (width, height)")
    
    # Load image if path is provided
    if isinstance(img, (str, Path)):
        try:
            img = Image.open(img)
        except OSError as e:
            raise OSError(f"Cannot open image file: {e}")
    
    # Convert numpy array to PIL Image
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    if not isinstance(img, Image.Image):
        raise TypeError("img must be PIL Image, numpy array, or file path")
    
    # Choose interpolation method
    if is_mask:
        interpolation = Image.NEAREST  # Preserve label values for masks
    else:
        interpolation = Image.BILINEAR  # Smooth interpolation for images
    
    # Resize while preserving aspect ratio
    img_resized = ImageOps.contain(img, size, method=interpolation)
    
    # Create new image with target size and background
    if is_mask:
        # Use black background for masks
        new_img = Image.new("L", size, 0)
    else:
        # Use black background for images
        new_img = Image.new("RGB", size, (0, 0, 0))
    
    # Calculate paste position to center the image
    paste_position = (
        (size[0] - img_resized.size[0]) // 2,
        (size[1] - img_resized.size[1]) // 2
    )
    
    # Paste the resized image onto the background
    new_img.paste(img_resized, paste_position)
    
    # Save if output path is provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        new_img.save(output_path)
        logger.info(f"Saved resized image to: {output_path}")
    
    return new_img

def resize_dataset_batch(
    input_images_dir: Union[str, Path],
    input_masks_dir: Union[str, Path],
    output_images_dir: Union[str, Path],
    output_masks_dir: Union[str, Path],
    target_size: Tuple[int, int] = (512, 512),
    image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png'),
    mask_extensions: Tuple[str, ...] = ('.png',)
) -> Tuple[int, int]:
    """
    Resize all images and masks in a dataset directory.
    
    Args:
        input_images_dir: Directory containing input images
        input_masks_dir: Directory containing input masks
        output_images_dir: Directory to save resized images
        output_masks_dir: Directory to save resized masks
        target_size: Target size for resized images
        image_extensions: File extensions to process for images
        mask_extensions: File extensions to process for masks
    
    Returns:
        Tuple[int, int]: (successful_count, failed_count)
    """
    input_images_dir = Path(input_images_dir)
    input_masks_dir = Path(input_masks_dir)
    output_images_dir = Path(output_images_dir)
    output_masks_dir = Path(output_masks_dir)
    
    # Create output directories
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_images_dir.glob(f"*{ext}"))
        image_files.extend(input_images_dir.glob(f"*{ext.upper()}"))
    
    successful_count = 0
    failed_count = 0
    
    for image_path in image_files:
        # Find corresponding mask
        mask_name = f"{image_path.stem}_mask{image_path.suffix}"
        mask_path = input_masks_dir / mask_name
        
        if not mask_path.exists():
            logger.warning(f"Mask not found for {image_path.name}: {mask_path}")
            failed_count += 1
            continue
        
        try:
            # Resize image
            output_image_path = output_images_dir / image_path.name
            resize_with_padding(
                image_path, 
                size=target_size, 
                is_mask=False,
                output_path=output_image_path
            )
            
            # Resize mask
            output_mask_path = output_masks_dir / mask_path.name
            resize_with_padding(
                mask_path, 
                size=target_size, 
                is_mask=True,
                output_path=output_mask_path
            )
            
            successful_count += 1
            
            if successful_count % 10 == 0:
                logger.info(f"Processed {successful_count} pairs...")
                
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            failed_count += 1
    
    logger.info(f"Resizing complete: {successful_count} successful, {failed_count} failed")
    return successful_count, failed_count

def verify_resized_files(
    images_dir: Union[str, Path],
    masks_dir: Union[str, Path],
    expected_size: Tuple[int, int] = (512, 512)
) -> dict:
    """
    Verify that all resized files have the correct dimensions.
    
    Args:
        images_dir: Directory containing resized images
        masks_dir: Directory containing resized masks
        expected_size: Expected size for all files
    
    Returns:
        dict: Statistics about the verification
    """
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    
    stats = {
        'total_images': 0,
        'total_masks': 0,
        'correct_size_images': 0,
        'correct_size_masks': 0,
        'incorrect_size_images': 0,
        'incorrect_size_masks': 0,
        'errors': []
    }
    
    # Check images
    for img_file in images_dir.iterdir():
        if img_file.is_file() and img_file.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            stats['total_images'] += 1
            try:
                with Image.open(img_file) as img:
                    if img.size == expected_size:
                        stats['correct_size_images'] += 1
                    else:
                        stats['incorrect_size_images'] += 1
                        stats['errors'].append(f"Image {img_file.name} has size {img.size}, expected {expected_size}")
            except Exception as e:
                stats['errors'].append(f"Error checking image {img_file}: {e}")
    
    # Check masks
    for mask_file in masks_dir.iterdir():
        if mask_file.is_file() and mask_file.suffix.lower() == '.png':
            stats['total_masks'] += 1
            try:
                with Image.open(mask_file) as mask:
                    if mask.size == expected_size:
                        stats['correct_size_masks'] += 1
                    else:
                        stats['incorrect_size_masks'] += 1
                        stats['errors'].append(f"Mask {mask_file.name} has size {mask.size}, expected {expected_size}")
            except Exception as e:
                stats['errors'].append(f"Error checking mask {mask_file}: {e}")
    
    return stats 