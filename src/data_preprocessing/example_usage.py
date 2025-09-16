#!/usr/bin/env python3
"""
Example Usage of CerVAI Data Preprocessing Package

This script demonstrates how to use the modular preprocessing utilities
for the CerVAI cervical cancer detection project.
"""

import logging
import pandas as pd
from pathlib import Path

# Import the modular preprocessing functions
from .resizing import resize_with_padding, resize_dataset_batch, verify_resized_files
from .filtering import check_size_mismatch, check_empty_mask, filter_dataset
from .split_utils import perform_stratified_split, create_split_directories, copy_files_to_splits
from .swede_category import map_swede_score_to_category, add_swede_categories_to_dataframe
from .filename_utils import generate_sequential_filenames, verify_filename_consistency

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def example_complete_preprocessing_pipeline():
    """
    Example of a complete preprocessing pipeline using the modular functions.
    """
    logger.info("Starting complete preprocessing pipeline example...")
    
    # Configuration
    csv_path = "aceto_mask_check_split.csv"
    input_images_dir = "dataset/revised_dataset/images"
    input_masks_dir = "dataset/revised_dataset/masks"
    output_images_dir = "dataset/processed_dataset/images"
    output_masks_dir = "dataset/processed_dataset/masks"
    split_output_dir = "dataset/split_dataset"
    
    # Step 1: Load and prepare data
    logger.info("Step 1: Loading and preparing data...")
    df = pd.read_csv(csv_path)
    
    # Add Swede categories
    df = add_swede_categories_to_dataframe(df, 'swede_score', 'swede_category')
    
    # Step 2: Filter dataset
    logger.info("Step 2: Filtering dataset...")
    filter_stats = filter_dataset(
        csv_path=csv_path,
        images_dir=input_images_dir,
        masks_dir=input_masks_dir,
        check_sizes=True,
        check_empty_masks=True,
        check_repeats=False
    )
    
    logger.info(f"Filtering results: {filter_stats}")
    
    # Step 3: Resize images and masks
    logger.info("Step 3: Resizing images and masks...")
    resize_stats = resize_dataset_batch(
        input_images_dir=input_images_dir,
        input_masks_dir=input_masks_dir,
        output_images_dir=output_images_dir,
        output_masks_dir=output_masks_dir,
        target_size=(512, 512)
    )
    
    logger.info(f"Resizing results: {resize_stats}")
    
    # Step 4: Verify resized files
    logger.info("Step 4: Verifying resized files...")
    verification_stats = verify_resized_files(
        images_dir=output_images_dir,
        masks_dir=output_masks_dir,
        expected_size=(512, 512)
    )
    
    logger.info(f"Verification results: {verification_stats}")
    
    # Step 5: Perform stratified split
    logger.info("Step 5: Performing stratified split...")
    train_df, val_df, test_df = perform_stratified_split(
        df=df,
        test_size=0.1,
        val_size=0.1,
        random_state=42,
        stratify_column='swede_category'
    )
    
    # Step 6: Create split directories
    logger.info("Step 6: Creating split directories...")
    split_dirs = create_split_directories(split_output_dir)
    
    # Step 7: Copy files to splits
    logger.info("Step 7: Copying files to splits...")
    copy_stats = copy_files_to_splits(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        source_images_dir=output_images_dir,
        source_masks_dir=output_masks_dir,
        split_dirs=split_dirs
    )
    
    logger.info(f"Copy results: {copy_stats}")
    
    # Step 8: Verify filename consistency
    logger.info("Step 8: Verifying filename consistency...")
    consistency_stats = verify_filename_consistency(
        images_dir=output_images_dir,
        masks_dir=output_masks_dir
    )
    
    logger.info(f"Consistency results: {consistency_stats}")
    
    logger.info("Preprocessing pipeline completed successfully!")

def example_individual_functions():
    """
    Example of using individual functions from the package.
    """
    logger.info("Demonstrating individual function usage...")
    
    # Example 1: Resize a single image
    logger.info("Example 1: Resizing a single image...")
    from PIL import Image
    import numpy as np
    
    # Create a dummy image
    dummy_image = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
    dummy_image = Image.fromarray(dummy_image)
    
    # Resize with padding
    resized_image = resize_with_padding(
        img=dummy_image,
        size=(512, 512),
        is_mask=False
    )
    
    logger.info(f"Original size: {dummy_image.size}")
    logger.info(f"Resized size: {resized_image.size}")
    
    # Example 2: Check size mismatch
    logger.info("Example 2: Checking size mismatch...")
    # This would normally check actual files
    # is_mismatch, sizes = check_size_mismatch(image_path, mask_path)
    
    # Example 3: Map Swede scores
    logger.info("Example 3: Mapping Swede scores...")
    test_scores = [0, 3, 5, 7, 10, None]
    for score in test_scores:
        category = map_swede_score_to_category(score)
        logger.info(f"Swede score {score} -> Category: {category}")
    
    # Example 4: Generate sequential filenames
    logger.info("Example 4: Generating sequential filenames...")
    image_filenames, mask_filenames = generate_sequential_filenames(
        start_index=1,
        count=5,
        padding=4
    )
    
    logger.info("Image filenames:")
    for filename in image_filenames:
        logger.info(f"  {filename}")
    
    logger.info("Mask filenames:")
    for filename in mask_filenames:
        logger.info(f"  {filename}")

def example_dataframe_operations():
    """
    Example of working with DataFrames and the preprocessing functions.
    """
    logger.info("Demonstrating DataFrame operations...")
    
    # Create a sample DataFrame
    sample_data = {
        'case': ['Case 1', 'Case 1', 'Case 2', 'Case 2'],
        'image_name': ['C1Aceto (1).jpg', 'C1Aceto (2).jpg', 'C2Aceto (1).jpg', 'C2Aceto (2).jpg'],
        'mask_name': ['C1Aceto (1).png', 'C1Aceto (2).png', 'C2Aceto (1).png', 'C2Aceto (2).png'],
        'swede_score': [0, 2, 5, 7],
        'drop': ['keep', 'keep', 'keep', 'keep']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Add Swede categories
    df = add_swede_categories_to_dataframe(df, 'swede_score', 'swede_category')
    
    logger.info("Sample DataFrame with Swede categories:")
    logger.info(df.to_string())
    
    # Perform stratified split
    train_df, val_df, test_df = perform_stratified_split(
        df=df,
        test_size=0.25,
        val_size=0.25,
        random_state=42
    )
    
    logger.info(f"Train set: {len(train_df)} samples")
    logger.info(f"Validation set: {len(val_df)} samples")
    logger.info(f"Test set: {len(test_df)} samples")

if __name__ == "__main__":
    # Run examples
    logger.info("=" * 60)
    logger.info("CerVAI Data Preprocessing Package Examples")
    logger.info("=" * 60)
    
    try:
        # Example 1: Individual functions
        example_individual_functions()
        
        logger.info("\n" + "=" * 60)
        
        # Example 2: DataFrame operations
        example_dataframe_operations()
        
        logger.info("\n" + "=" * 60)
        
        # Example 3: Complete pipeline (commented out as it requires actual data)
        # example_complete_preprocessing_pipeline()
        logger.info("Complete pipeline example skipped (requires actual data files)")
        
        logger.info("\nAll examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise 