"""
Enhanced Mask Border Check and Completion Utilities

This module detects open lesion contours in binary masks and attempts automatic
border-based closure when appropriate. It provides comprehensive logging and
visualization for doctor review.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
import logging
from PIL import Image
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)

def detect_open_contours(mask: np.ndarray) -> Tuple[bool, List[np.ndarray], Dict]:
    """
    Detect open contours in binary mask using arcLength comparison.
    
    Args:
        mask: Binary mask array (0 or 1)
        
    Returns:
        Tuple[bool, List, Dict]: (has_open_contours, open_contours, details)
    """
    # Find contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                   cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    open_contours = []
    details = {
        'total_contours': len(contours),
        'open_contours': 0,
        'closed_contours': 0,
        'contour_details': []
    }
    
    for i, contour in enumerate(contours):
        if len(contour) < 3:
            continue
            
        # Calculate arc length with closed=False and closed=True
        arc_length_open = cv2.arcLength(contour, closed=False)
        arc_length_closed = cv2.arcLength(contour, closed=True)
        
        # Check if contour is open (significant difference in arc lengths)
        is_open = abs(arc_length_open - arc_length_closed) > 1.0
        
        # Additional check: if contour touches border
        touches_border = check_contour_touches_border(contour, mask.shape)
        
        contour_info = {
            'index': i,
            'is_open': is_open,
            'touches_border': touches_border,
            'arc_length_open': arc_length_open,
            'arc_length_closed': arc_length_closed,
            'area': cv2.contourArea(contour)
        }
        
        details['contour_details'].append(contour_info)
        
        if is_open and touches_border:
            open_contours.append(contour)
            details['open_contours'] += 1
        else:
            details['closed_contours'] += 1
    
    return len(open_contours) > 0, open_contours, details

def check_contour_touches_border(contour: np.ndarray, mask_shape: Tuple[int, int]) -> bool:
    """
    Check if contour touches any of the image borders.
    
    Args:
        contour: Contour points
        mask_shape: Shape of the mask (height, width)
        
    Returns:
        bool: True if contour touches border
    """
    height, width = mask_shape
    
    # Check if any contour point is on the border
    for point in contour:
        x, y = point[0]
        if x == 0 or x == width - 1 or y == 0 or y == height - 1:
            return True
    
    return False

def attempt_border_completion(mask: np.ndarray, open_contours: List[np.ndarray]) -> Tuple[np.ndarray, bool, Dict]:
    """
    Attempt to complete open contours by connecting endpoints along borders.
    
    Args:
        mask: Original binary mask
        open_contours: List of open contours
        
    Returns:
        Tuple[np.ndarray, bool, Dict]: (completed_mask, was_completed, details)
    """
    completed_mask = mask.copy()
    completion_details = {
        'contours_processed': 0,
        'contours_completed': 0,
        'pixels_added': 0
    }
    
    for contour in open_contours:
        completion_details['contours_processed'] += 1
        
        # Simplify contour
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx_contour = cv2.approxPolyDP(contour, epsilon, False)
        
        if len(approx_contour) < 2:
            continue
            
        # Find endpoints (first and last points)
        endpoints = [approx_contour[0][0], approx_contour[-1][0]]
        
        # Check if both endpoints are on border
        height, width = mask.shape
        on_border = []
        
        for endpoint in endpoints:
            x, y = endpoint
            if x == 0 or x == width - 1 or y == 0 or y == height - 1:
                on_border.append(endpoint)
        
        # If both endpoints are on border, attempt to connect them
        if len(on_border) >= 2:
            # Create a temporary mask for this contour
            temp_mask = np.zeros_like(mask)
            cv2.drawContours(temp_mask, [contour], -1, 1, -1)
            
            # Connect endpoints along border
            connected_mask = connect_endpoints_along_border(temp_mask, on_border, mask.shape)
            
            if connected_mask is not None:
                # Fill the connected region
                filled_mask = fill_connected_region(connected_mask)
                
                # Merge with original mask
                completed_mask = np.logical_or(completed_mask, filled_mask).astype(np.uint8)
                completion_details['contours_completed'] += 1
                completion_details['pixels_added'] += np.sum(filled_mask)
    
    was_completed = completion_details['contours_completed'] > 0
    return completed_mask, was_completed, completion_details

def connect_endpoints_along_border(mask: np.ndarray, endpoints: List[Tuple[int, int]], 
                                 shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Connect endpoints along the image border.
    
    Args:
        mask: Binary mask
        endpoints: List of endpoint coordinates
        shape: Image shape
        
    Returns:
        Optional[np.ndarray]: Connected mask or None if failed
    """
    if len(endpoints) < 2:
        return None
    
    height, width = shape
    connected_mask = mask.copy()
    
    # Find the border path between endpoints
    border_path = find_border_path(endpoints[0], endpoints[1], width, height)
    
    if border_path:
        # Draw the border path
        for i in range(len(border_path) - 1):
            pt1 = border_path[i]
            pt2 = border_path[i + 1]
            cv2.line(connected_mask, pt1, pt2, 1, 1)
        
        return connected_mask
    
    return None

def find_border_path(pt1: Tuple[int, int], pt2: Tuple[int, int], 
                    width: int, height: int) -> List[Tuple[int, int]]:
    """
    Find the shortest path along the border between two points.
    
    Args:
        pt1: First endpoint
        pt2: Second endpoint
        width: Image width
        height: Image height
        
    Returns:
        List[Tuple[int, int]]: Border path points
    """
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Determine which borders the points are on
    borders1 = get_border_edges(x1, y1, width, height)
    borders2 = get_border_edges(x2, y2, width, height)
    
    # Find common border or shortest path
    common_borders = borders1.intersection(borders2)
    
    if common_borders:
        # Both points on same border
        border = list(common_borders)[0]
        return get_path_along_border(pt1, pt2, border, width, height)
    else:
        # Points on different borders, find corner path
        return get_corner_path(pt1, pt2, width, height)

def get_border_edges(x: int, y: int, width: int, height: int) -> set:
    """Get which border edges a point is on."""
    edges = set()
    
    if x == 0:
        edges.add('left')
    if x == width - 1:
        edges.add('right')
    if y == 0:
        edges.add('top')
    if y == height - 1:
        edges.add('bottom')
    
    return edges

def get_path_along_border(pt1: Tuple[int, int], pt2: Tuple[int, int], 
                         border: str, width: int, height: int) -> List[Tuple[int, int]]:
    """Get path along a specific border."""
    x1, y1 = pt1
    x2, y2 = pt2
    
    if border == 'top':
        return [(i, 0) for i in range(min(x1, x2), max(x1, x2) + 1)]
    elif border == 'bottom':
        return [(i, height - 1) for i in range(min(x1, x2), max(x1, x2) + 1)]
    elif border == 'left':
        return [(0, i) for i in range(min(y1, y2), max(y1, y2) + 1)]
    elif border == 'right':
        return [(width - 1, i) for i in range(min(y1, y2), max(y1, y2) + 1)]
    
    return []

def get_corner_path(pt1: Tuple[int, int], pt2: Tuple[int, int], 
                   width: int, height: int) -> List[Tuple[int, int]]:
    """Get path through corners when points are on different borders."""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Determine corner to use based on point positions
    if x1 == 0 and y2 == 0:  # pt1 on left, pt2 on top
        corner = (0, 0)
    elif x1 == width - 1 and y2 == 0:  # pt1 on right, pt2 on top
        corner = (width - 1, 0)
    elif x1 == 0 and y2 == height - 1:  # pt1 on left, pt2 on bottom
        corner = (0, height - 1)
    elif x1 == width - 1 and y2 == height - 1:  # pt1 on right, pt2 on bottom
        corner = (width - 1, height - 1)
    else:
        # Default to top-left corner
        corner = (0, 0)
    
    # Create path through corner
    path = []
    
    # Path from pt1 to corner
    if x1 == 0 or x1 == width - 1:  # Vertical border
        for y in range(min(y1, corner[1]), max(y1, corner[1]) + 1):
            path.append((x1, y))
    else:  # Horizontal border
        for x in range(min(x1, corner[0]), max(x1, corner[0]) + 1):
            path.append((x, y1))
    
    # Path from corner to pt2
    if x2 == 0 or x2 == width - 1:  # Vertical border
        for y in range(min(y2, corner[1]), max(y2, corner[1]) + 1):
            path.append((x2, y))
    else:  # Horizontal border
        for x in range(min(x2, corner[0]), max(x2, corner[0]) + 1):
            path.append((x, y2))
    
    return path

def fill_connected_region(mask: np.ndarray) -> np.ndarray:
    """
    Fill the connected region in the mask.
    
    Args:
        mask: Binary mask with contour and border path
        
    Returns:
        np.ndarray: Filled mask
    """
    # Find contours in the connected mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                   cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    filled_mask = np.zeros_like(mask)
    
    for contour in contours:
        # Check if contour is closed (area > 0)
        area = cv2.contourArea(contour)
        if area > 0:
            cv2.fillPoly(filled_mask, [contour], 1)
    
    return filled_mask

def check_mask_border_completion(mask_path: Union[str, Path]) -> Tuple[bool, bool, Dict]:
    """
    Check if a binary mask has open contours and attempt border completion.
    
    Args:
        mask_path: Path to the binary mask file
        
    Returns:
        Tuple[bool, bool, Dict]: (is_open, was_completed, details)
    """
    try:
        # Load mask
        mask = np.array(Image.open(mask_path).convert('L'))
        mask = (mask > 0).astype(np.uint8)
        
        # Detect open contours
        has_open_contours, open_contours, detection_details = detect_open_contours(mask)
        
        if not has_open_contours:
            print(f"[DEBUG] {mask_path.name} → is_open: False, was_completed: False")
            return False, False, {
                'is_open': False,
                'was_completed': False,
                'detection_details': detection_details,
                'completion_details': {}
            }
        
        # Attempt border completion
        completed_mask, was_completed, completion_details = attempt_border_completion(mask, open_contours)
        
        # Calculate pixel difference for debugging
        print(f"[DEBUG] {mask_path.name} → is_open: True, was_completed: {was_completed}")
        print(f"[DEBUG] Completed contours: {completion_details['contours_completed']}")

        pixel_diff = np.sum(completed_mask != mask)
        
        return True, was_completed, {
            'is_open': True,
            'was_completed': was_completed,
            'detection_details': detection_details,
            'completion_details': completion_details,
            'pixel_difference': pixel_diff,
            'original_mask': mask,
            'completed_mask': completed_mask
        }

    except Exception as e:
        logger.error(f"Error processing {mask_path}: {e}")
        return False, False, {'error': str(e)}


def create_border_completion_visualization(mask_path: Union[str, Path],
                                           output_path: Union[str, Path],
                                           details: Dict) -> None:
    """
    Create visualization showing original vs completed mask and difference.
    Always generates image, regardless of whether completion was successful.

    Args:
        mask_path: Path to original mask
        output_path: Path to save visualization
        details: Processing details
    """
    try:
        # Load original mask
        original_mask = np.array(Image.open(mask_path).convert('L'))
        original_mask = (original_mask > 0).astype(np.uint8)

        # Get completed mask
        completed_mask = details.get('completed_mask', original_mask)

        # Compute difference
        diff = (completed_mask != original_mask).astype(np.uint8)
        pixel_diff = int(np.sum(diff))

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original mask
        axes[0].imshow(original_mask, cmap='gray')
        axes[0].set_title('Original Mask')
        axes[0].axis('off')

        # Completed mask
        axes[1].imshow(completed_mask, cmap='gray')
        axes[1].set_title('Completed Mask')
        axes[1].axis('off')

        # Difference
        axes[2].imshow(diff, cmap='Reds')
        axes[2].set_title(f'Difference ({pixel_diff} pixels)')
        axes[2].axis('off')

        # Info text
        is_open = details.get("is_open", False)
        was_completed = details.get("was_completed", False)
        contours_completed = details.get("completion_details", {}).get("contours_completed", 0)
        pixels_added = details.get("completion_details", {}).get("pixels_added", 0)

        if not is_open:
            status = "Status: Closed (no need to complete)"
        elif was_completed:
            status = f"Status: Completed ({contours_completed} contour(s), {pixels_added} pixels added)"
        else:
            status = "Status: Open but not completed"

        fig.suptitle(f"Border Completion Analysis\n{status}", fontsize=12)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[DEBUG] Saved visualization: {output_path}, pixel_diff = {pixel_diff}")

    except Exception as e:
        logger.error(f"Error creating visualization for {mask_path}: {e}")

def process_split_with_border_completion(split: str, csv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process a dataset split and update CSV with border completion results.

    Args:
        split: Dataset split name (train/val/test)
        csv_df: DataFrame to update

    Returns:
        pd.DataFrame: Updated DataFrame
    """
    input_dir = Path(f"/Users/daidai/Documents/pythonProject_summer/CerVAI/dataset/dataset_split/{split}/masks_binary")
    vis_dir = Path(f"/Users/daidai/Documents/pythonProject_summer/CerVAI/visualizations/mask_border_completion/{split}")
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Filter rows for this split
    split_mask = csv_df['set'] == split
    split_df = csv_df[split_mask].copy()

    logger.info(f"Processing {split} split: {len(split_df)} masks")

    open_count = 0
    completed_count = 0

    for idx, row in split_df.iterrows():
        mask_name = row['new_mask_name']
        mask_path = input_dir / mask_name

        if not mask_path.exists():
            logger.warning(f"Mask not found: {mask_path}")
            continue

        # Check border completion
        is_open, was_completed, details = check_mask_border_completion(mask_path)

        # Update CSV
        csv_df.loc[idx, 'mask_open'] = 'open' if is_open else 'closed'
        csv_df.loc[idx, 'border_completed'] = 'yes' if was_completed else 'no'

        if is_open:
            open_count += 1
            if was_completed:
                completed_count += 1
                csv_df.loc[idx, 'doctor_review'] = 'auto_completed'
            else:
                csv_df.loc[idx, 'doctor_review'] = 'needed'
        else:
            csv_df.loc[idx, 'doctor_review'] = 'not_needed'

        # Create visualization for open masks
        if is_open:
            vis_path = vis_dir / f"{mask_name.replace('.png', '_bordercheck.png')}"
            create_border_completion_visualization(mask_path, vis_path, details)

    logger.info(f"{split} split: {open_count} open, {completed_count} auto-completed")
    return csv_df


def process_all_splits_with_border_completion(csv_path: str) -> None:
    """
    Process all dataset splits and update CSV with border completion results.
    
    Args:
        csv_path: Path to CSV file
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Initialize new columns if they don't exist
    if 'mask_open' not in df.columns:
        df['mask_open'] = 'unknown'
    if 'border_completed' not in df.columns:
        df['border_completed'] = 'no'
    if 'doctor_review' not in df.columns:
        df['doctor_review'] = 'unknown'
    
    # Process each split
    for split in ['train', 'val', 'test']:
        df = process_split_with_border_completion(split,df)
    
    # Save updated CSV
    df.to_csv(csv_path, index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("BORDER COMPLETION ANALYSIS SUMMARY")
    print("="*60)
    
    total_samples = len(df[df['set'].isin(['train', 'val', 'test'])])
    open_masks = len(df[df['mask_open'] == 'open'])
    completed_masks = len(df[df['border_completed'] == 'yes'])
    need_review = len(df[df['doctor_review'] == 'needed'])
    
    print(f"Total samples: {total_samples}")
    print(f"Open masks: {open_masks} ({open_masks/total_samples*100:.1f}%)")
    print(f"Auto-completed: {completed_masks} ({completed_masks/open_masks*100:.1f}% of open)")
    print(f"Need doctor review: {need_review} ({need_review/open_masks*100:.1f}% of open)")
    
    # Analysis by split
    for split in ['train', 'val', 'test']:
        split_df = df[df['set'] == split]
        split_open = len(split_df[split_df['mask_open'] == 'open'])
        split_completed = len(split_df[split_df['border_completed'] == 'yes'])
        split_review = len(split_df[split_df['doctor_review'] == 'needed'])
        
        print(f"\n{split.upper()} split:")
        print(f"  Open: {split_open}, Auto-completed: {split_completed}, Need review: {split_review}")
    
    # Analysis by category
    for category in ['low', 'mid', 'high']:
        cat_df = df[df['swede_category'] == category]
        cat_open = len(cat_df[cat_df['mask_open'] == 'open'])
        cat_completed = len(cat_df[cat_df['border_completed'] == 'yes'])
        cat_review = len(cat_df[cat_df['doctor_review'] == 'needed'])
        
        if cat_open > 0:
            print(f"\n{category.upper()} risk:")
            print(f"  Open: {cat_open}, Auto-completed: {cat_completed}, Need review: {cat_review}")

if __name__ == "__main__":
    # Process all splits
    csv_path = "/Users/daidai/Documents/pythonProject_summer/CerVAI/aceto_mask_check_split.csv"
    process_all_splits_with_border_completion(csv_path) 