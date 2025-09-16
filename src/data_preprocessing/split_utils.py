"""
Dataset Splitting Utilities

This module provides functions for splitting datasets into train/validation/test sets
with stratification based on swede categories.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from typing import Tuple, Dict, List, Union, Optional
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def perform_stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify_column: str = 'swede_category'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform stratified split of the dataset into train/val/test sets.
    
    Args:
        df: DataFrame with dataset metadata
        test_size: Fraction of data for test set
        val_size: Fraction of data for validation set
        random_state: Random seed for reproducibility
        stratify_column: Column to use for stratification
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
    """
    if stratify_column not in df.columns:
        raise ValueError(f"Stratification column '{stratify_column}' not found in DataFrame")
    
    # Filter out dropped rows
    df_valid = df[df['drop'] != 'drop'].copy()
    
    if len(df_valid) == 0:
        raise ValueError("No valid rows found after filtering dropped entries")
    
    # Get stratification labels
    stratify_labels = df_valid[stratify_column].values
    
    # Calculate split sizes
    total_size = len(df_valid)
    test_count = int(total_size * test_size)
    val_count = int(total_size * val_size)
    train_count = total_size - test_count - val_count
    
    logger.info(f"Dataset split configuration:")
    logger.info(f"  Total valid samples: {total_size}")
    logger.info(f"  Train samples: {train_count} ({train_count/total_size:.1%})")
    logger.info(f"  Validation samples: {val_count} ({val_count/total_size:.1%})")
    logger.info(f"  Test samples: {test_count} ({test_count/total_size:.1%})")
    
    # First split: train + val vs test
    train_val_df, test_df, train_val_labels, test_labels = train_test_split(
        df_valid, stratify_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels
    )
    
    # Second split: train vs val
    train_df, val_df, train_labels, val_labels = train_test_split(
        train_val_df, train_val_labels,
        test_size=val_count / (train_count + val_count),
        random_state=random_state,
        stratify=train_val_labels
    )
    
    # Add set column
    train_df['set'] = 'train'
    val_df['set'] = 'val'
    test_df['set'] = 'test'
    
    # Log stratification distribution
    logger.info(f"Split results:")
    logger.info(f"  Train: {len(train_df)} samples")
    logger.info(f"  Validation: {len(val_df)} samples")
    logger.info(f"  Test: {len(test_df)} samples")
    
    for split_name, split_df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
        category_counts = split_df[stratify_column].value_counts().to_dict()
        logger.info(f"  {split_name} category distribution: {category_counts}")
    
    return train_df, val_df, test_df

def create_split_directories(
    base_dir: Union[str, Path],
    splits: List[str] = None
) -> Dict[str, Dict[str, Path]]:
    """
    Create directory structure for dataset splits.
    
    Args:
        base_dir: Base directory for the split dataset
        splits: List of split names (default: ['train', 'val', 'test'])
    
    Returns:
        Dict: Dictionary with directory paths for each split
    """
    if splits is None:
        splits = ['train', 'val', 'test']
    
    base_dir = Path(base_dir)
    dirs = {}
    
    for split in splits:
        split_dir = base_dir / split
        images_dir = split_dir / 'images'
        masks_dir = split_dir / 'masks'
        
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        
        dirs[split] = {
            'base': split_dir,
            'images': images_dir,
            'masks': masks_dir
        }
        
        logger.info(f"Created directory: {split_dir}")
    
    return dirs

def copy_files_to_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    source_images_dir: Union[str, Path],
    source_masks_dir: Union[str, Path],
    split_dirs: Dict[str, Dict[str, Path]]
) -> Dict[str, int]:
    """
    Copy files to their respective split directories.
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
        source_images_dir: Source images directory
        source_masks_dir: Source masks directory
        split_dirs: Dictionary with split directory paths
    
    Returns:
        Dict: Statistics about copied files
    """
    source_images_dir = Path(source_images_dir)
    source_masks_dir = Path(source_masks_dir)
    
    stats = {'train': 0, 'val': 0, 'test': 0, 'errors': 0}
    
    split_dataframes = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }
    
    for split_name, split_df in split_dataframes.items():
        logger.info(f"Copying files to {split_name} directory...")
        
        for idx, row in split_df.iterrows():
            # Get file names
            if 'new_image_name' in row and pd.notna(row['new_image_name']):
                img_file = row['new_image_name']
                mask_file = row['new_mask_name']
            else:
                img_file = row['image_name']
                mask_file = row['mask_name']
            
            # Source paths
            src_img = source_images_dir / img_file
            src_mask = source_masks_dir / mask_file
            
            # Destination paths
            dst_img = split_dirs[split_name]['images'] / img_file
            dst_mask = split_dirs[split_name]['masks'] / mask_file
            
            try:
                # Copy files
                shutil.copy2(src_img, dst_img)
                shutil.copy2(src_mask, dst_mask)
                stats[split_name] += 1
                
            except Exception as e:
                logger.error(f"Error copying files for {img_file}: {e}")
                stats['errors'] += 1
    
    logger.info(f"File copying complete:")
    logger.info(f"  Train: {stats['train']} files")
    logger.info(f"  Validation: {stats['val']} files")
    logger.info(f"  Test: {stats['test']} files")
    logger.info(f"  Errors: {stats['errors']} files")
    
    return stats

def merge_split_info_to_original_df(
    df_original: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge split information back to the original dataframe.
    
    Args:
        df_original: Original dataframe
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
    
    Returns:
        pd.DataFrame: Original dataframe with split information added
    """
    logger.info("Merging split information back to original dataframe...")
    
    # Combine all split dataframes
    split_df = pd.concat([train_df, val_df, test_df])
    
    # Create a mapping from index to set
    index_to_set = split_df['set'].to_dict()
    
    # Add set column to original dataframe
    df_original['set'] = df_original.index.map(index_to_set)
    
    # Fill NaN values (for dropped rows) with 'dropped'
    df_original['set'] = df_original['set'].fillna('dropped')
    
    # Log distribution
    set_counts = df_original['set'].value_counts().to_dict()
    logger.info(f"Final dataset distribution: {set_counts}")
    
    return df_original

def verify_split_integrity(
    split_dirs: Dict[str, Dict[str, Path]],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Dict[str, any]:
    """
    Verify that the split directories match the expected files.
    
    Args:
        split_dirs: Dictionary with split directory paths
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
    
    Returns:
        Dict: Verification statistics
    """
    verification_stats = {
        'train_files_expected': len(train_df),
        'val_files_expected': len(val_df),
        'test_files_expected': len(test_df),
        'train_files_found': 0,
        'val_files_found': 0,
        'test_files_found': 0,
        'errors': []
    }
    
    split_dataframes = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }
    
    for split_name, split_df in split_dataframes.items():
        expected_files = set()
        for _, row in split_df.iterrows():
            if 'new_image_name' in row and pd.notna(row['new_image_name']):
                img_file = row['new_image_name']
                mask_file = row['new_mask_name']
            else:
                img_file = row['image_name']
                mask_file = row['mask_name']
            
            expected_files.add(img_file)
            expected_files.add(mask_file)
        
        # Check actual files
        images_dir = split_dirs[split_name]['images']
        masks_dir = split_dirs[split_name]['masks']
        
        actual_images = set([f.name for f in images_dir.iterdir() if f.is_file()])
        actual_masks = set([f.name for f in masks_dir.iterdir() if f.is_file()])
        
        actual_files = actual_images.union(actual_masks)
        
        # Count found files
        found_count = len(actual_files.intersection(expected_files))
        verification_stats[f'{split_name}_files_found'] = found_count
        
        # Check for missing files
        missing_files = expected_files - actual_files
        if missing_files:
            verification_stats['errors'].append(f"Missing files in {split_name}: {list(missing_files)}")
        
        # Check for extra files
        extra_files = actual_files - expected_files
        if extra_files:
            verification_stats['errors'].append(f"Extra files in {split_name}: {list(extra_files)}")
    
    return verification_stats

def create_split_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Create a summary of the dataset splits.
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
        output_path: Optional path to save summary CSV
    
    Returns:
        pd.DataFrame: Summary dataframe
    """
    summary_data = []
    
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        # Count by swede category
        category_counts = split_df['swede_category'].value_counts().to_dict()
        
        # Count by case
        case_counts = split_df['case'].value_counts().to_dict()
        
        summary_data.append({
            'split': split_name,
            'total_samples': len(split_df),
            'unique_cases': len(case_counts),
            'low_category': category_counts.get('low', 0),
            'mid_category': category_counts.get('mid', 0),
            'high_category': category_counts.get('high', 0),
            'unknown_category': category_counts.get('unknown', 0)
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    if output_path is not None:
        output_path = Path(output_path)
        summary_df.to_csv(output_path, index=False)
        logger.info(f"Split summary saved to: {output_path}")
    
    return summary_df 