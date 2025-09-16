"""
Mask Open Contour Detection Utilities

This module detects unclosed lesion contours in binary masks using OpenCV.
It identifies masks with open contours that may need doctor review during active learning.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
import logging
from PIL import Image

logger = logging.getLogger(__name__)

def check_mask_open_contour(mask_path: Union[str, Path]) -> Tuple[bool, Dict]:
    """
    Check if a binary mask contains open contours.
    
    Args:
        mask_path: Path to the binary mask file
        
    Returns:
        Tuple[bool, Dict]: (is_open, details)
            - is_open: True if mask has open contours
            - details: Dictionary with contour analysis details
    """
    try:
        # Load mask as grayscale
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            logger.error(f"Could not load mask: {mask_path}")
            return True, {'error': 'Could not load mask'}
        
        # Ensure binary (0 or 255)
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # No contours found - mask is empty
            return False, {
                'contour_count': 0,
                'open_contours': 0,
                'closed_contours': 0,
                'total_area': 0
            }
        
        open_contours = 0
        closed_contours = 0
        total_area = 0
        
        for contour in contours:
            # Calculate contour area
            area = cv2.contourArea(contour)
            total_area += area
            
            # Check if contour is convex (closed)
            if len(contour) >= 3:  # Need at least 3 points for convexity
                is_convex = cv2.isContourConvex(contour)
                if is_convex:
                    closed_contours += 1
                else:
                    open_contours += 1
            else:
                # Very small contours might be noise
                if area > 10:  # Minimum area threshold
                    open_contours += 1
        
        # Determine if mask has open contours
        is_open = open_contours > 0
        
        details = {
            'contour_count': len(contours),
            'open_contours': open_contours,
            'closed_contours': closed_contours,
            'total_area': total_area,
            'is_open': is_open
        }
        
        return is_open, details
        
    except Exception as e:
        logger.error(f"Error checking mask {mask_path}: {e}")
        return True, {'error': str(e)}

def process_split_masks(split: str, 
                       masks_dir: Union[str, Path] = None) -> Dict[str, Dict]:
    """
    Process all binary masks in a dataset split.
    
    Args:
        split: Dataset split name ('train', 'val', 'test')
        masks_dir: Directory containing binary masks (optional)
        
    Returns:
        Dict[str, Dict]: Mapping of mask filename to analysis results
    """
    if masks_dir is None:
        masks_dir = Path(f"dataset/dataset_split/{split}/masks_binary")
    
    masks_dir = Path(masks_dir)
    
    if not masks_dir.exists():
        logger.warning(f"Binary masks directory not found: {masks_dir}")
        return {}
    
    results = {}
    mask_files = list(masks_dir.glob("*.png"))
    
    logger.info(f"Processing {len(mask_files)} masks for {split} split...")
    
    for mask_path in mask_files:
        is_open, details = check_mask_open_contour(mask_path)
        results[mask_path.name] = {
            'is_open': is_open,
            'details': details
        }
    
    # Log summary
    open_count = sum(1 for result in results.values() if result['is_open'])
    total_count = len(results)
    logger.info(f"{split} split: {open_count}/{total_count} masks have open contours")
    
    return results

def update_csv_with_open_status(csv_path: Union[str, Path],
                               output_csv_path: Optional[Union[str, Path]] = None) -> Dict:
    """
    Update CSV file with mask open/closed status for all splits.
    
    Args:
        csv_path: Path to the input CSV file
        output_csv_path: Path to save updated CSV (if None, overwrites input)
        
    Returns:
        Dict: Statistics about the update process
    """
    csv_path = Path(csv_path)
    
    if output_csv_path is None:
        output_csv_path = csv_path
    else:
        output_csv_path = Path(output_csv_path)
    
    # Load CSV
    df = pd.read_csv(csv_path)
    initial_count = len(df)
    
    # Initialize mask_open column
    df['mask_open'] = 'closed'  # Default to closed
    
    stats = {
        'total_samples': initial_count,
        'processed_splits': 0,
        'open_masks': 0,
        'closed_masks': 0,
        'missing_masks': 0,
        'errors': []
    }
    
    # Process each split
    for split in ['train', 'val', 'test']:
        try:
            split_results = process_split_masks(split)
            
            if split_results:
                stats['processed_splits'] += 1
                
                # Update CSV for this split
                for mask_name, result in split_results.items():
                    # Find corresponding row in CSV
                    mask_rows = df[df['new_mask_name'] == mask_name]
                    
                    if len(mask_rows) > 0:
                        # Update the mask_open column
                        mask_status = 'open' if result['is_open'] else 'closed'
                        df.loc[mask_rows.index, 'mask_open'] = mask_status
                        
                        if result['is_open']:
                            stats['open_masks'] += 1
                        else:
                            stats['closed_masks'] += 1
                    else:
                        stats['missing_masks'] += 1
                        stats['errors'].append(f"Mask {mask_name} not found in CSV")
            
        except Exception as e:
            logger.error(f"Error processing {split} split: {e}")
            stats['errors'].append(f"{split} split: {e}")
    
    # Save updated CSV
    df.to_csv(output_csv_path, index=False)
    
    logger.info(f"Updated CSV with mask open status:")
    logger.info(f"  Total samples: {stats['total_samples']}")
    logger.info(f"  Open masks: {stats['open_masks']}")
    logger.info(f"  Closed masks: {stats['closed_masks']}")
    logger.info(f"  Missing masks: {stats['missing_masks']}")
    
    return stats

def analyze_open_contours(csv_path: Union[str, Path]) -> Dict:
    """
    Analyze the distribution of open vs closed contours in the dataset.
    
    Args:
        csv_path: Path to the CSV file with mask_open column
        
    Returns:
        Dict: Analysis results
    """
    df = pd.read_csv(csv_path)
    
    if 'mask_open' not in df.columns:
        logger.error("CSV does not contain 'mask_open' column")
        return {}
    
    analysis = {
        'total_samples': len(df),
        'open_masks': len(df[df['mask_open'] == 'open']),
        'closed_masks': len(df[df['mask_open'] == 'closed']),
        'unknown_masks': len(df[df['mask_open'].isna()]),
        'by_split': {},
        'by_category': {}
    }
    
    # Analysis by split
    if 'set' in df.columns:
        for split in ['train', 'val', 'test']:
            split_df = df[df['set'] == split]
            if len(split_df) > 0:
                analysis['by_split'][split] = {
                    'total': len(split_df),
                    'open': len(split_df[split_df['mask_open'] == 'open']),
                    'closed': len(split_df[split_df['mask_open'] == 'closed'])
                }
    
    # Analysis by swede category
    if 'swede_category' in df.columns:
        for category in df['swede_category'].unique():
            if pd.notna(category):
                cat_df = df[df['swede_category'] == category]
                analysis['by_category'][category] = {
                    'total': len(cat_df),
                    'open': len(cat_df[cat_df['mask_open'] == 'open']),
                    'closed': len(cat_df[cat_df['mask_open'] == 'closed'])
                }
    
    return analysis

def visualize_open_masks(csv_path: Union[str, Path], 
                        output_dir: Union[str, Path] = None,
                        max_samples: int = 10) -> None:
    """
    Create visualizations of masks with open contours for review.
    
    Args:
        csv_path: Path to the CSV file
        output_dir: Directory to save visualizations
        max_samples: Maximum number of samples to visualize
    """
    df = pd.read_csv(csv_path)
    
    if 'mask_open' not in df.columns:
        logger.error("CSV does not contain 'mask_open' column")
        return
    
    open_masks = df[df['mask_open'] == 'open']
    
    if len(open_masks) == 0:
        logger.info("No open masks found to visualize")
        return
    
    if output_dir is None:
        output_dir = Path("open_masks_visualization")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Sample masks to visualize
    sample_masks = open_masks.head(max_samples)
    
    logger.info(f"Creating visualizations for {len(sample_masks)} open masks...")
    
    for idx, row in sample_masks.iterrows():
        try:
            # Find corresponding mask file
            mask_name = row['new_mask_name']
            mask_path = None
            
            # Look in each split directory
            for split in ['train', 'val', 'test']:
                potential_path = Path(f"dataset/dataset_split/{split}/masks_binary/{mask_name}")
                if potential_path.exists():
                    mask_path = potential_path
                    break
            
            if mask_path and mask_path.exists():
                # Load and visualize mask
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                
                # Find contours for visualization
                _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Create visualization
                vis_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                
                # Draw contours
                for contour in contours:
                    if len(contour) >= 3:
                        is_convex = cv2.isContourConvex(contour)
                        color = (0, 255, 0) if is_convex else (0, 0, 255)  # Green=closed, Red=open
                        cv2.drawContours(vis_img, [contour], -1, color, 2)
                
                # Save visualization
                output_path = output_dir / f"open_mask_{idx}_{mask_name}"
                cv2.imwrite(str(output_path), vis_img)
                
        except Exception as e:
            logger.error(f"Error visualizing mask {mask_name}: {e}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # CSV file path
    csv_path = "aceto_mask_check_split.csv"
    
    print("="*60)
    print("MASK OPEN CONTOUR DETECTION")
    print("="*60)
    
    # Update CSV with open/closed status
    stats = update_csv_with_open_status(csv_path)
    
    print(f"\nProcessing Summary:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Open masks: {stats['open_masks']}")
    print(f"  Closed masks: {stats['closed_masks']}")
    print(f"  Missing masks: {stats['missing_masks']}")
    
    # Analyze results
    analysis = analyze_open_contours(csv_path)
    
    print(f"\nAnalysis by Split:")
    for split, data in analysis.get('by_split', {}).items():
        print(f"  {split.upper()}: {data['open']} open, {data['closed']} closed")
    
    print(f"\nAnalysis by Category:")
    for category, data in analysis.get('by_category', {}).items():
        print(f"  {category.upper()}: {data['open']} open, {data['closed']} closed")
    
    # Create visualizations if open masks found
    if stats['open_masks'] > 0:
        print(f"\nCreating visualizations for open masks...")
        visualize_open_masks(csv_path)
        print(f"Visualizations saved to 'open_masks_visualization/' directory") 