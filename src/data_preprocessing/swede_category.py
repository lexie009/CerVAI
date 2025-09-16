"""
Swede Score to Category Mapping Utilities

This module provides functions for converting numerical Swede scores to categorical
labels for stratification and analysis purposes.
"""

import pandas as pd
from typing import Union, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

def map_swede_score_to_category(
    swede_score: Union[int, float],
    mapping_rules: Optional[Dict] = None
) -> str:
    """
    Map a numerical Swede score to a categorical label.
    
    Args:
        swede_score: Numerical Swede score (0-10)
        mapping_rules: Optional custom mapping rules
    
    Returns:
        str: Categorical label ('low', 'mid', 'high', or 'unknown')
    """
    if mapping_rules is None:
        # Default mapping rules
        mapping_rules = {
            'low': (0, 4),    # Scores 0-4: low risk
            'mid': (5, 6),    # Scores 5-6: mid risk  
            'high': (7, 10)   # Scores 7-10: high risk
        }
    
    # Handle NaN or invalid values
    if pd.isna(swede_score) or not isinstance(swede_score, (int, float)):
        return 'unknown'
    
    # Convert to int for comparison
    score = int(swede_score)
    
    # Apply mapping rules
    for category, (min_score, max_score) in mapping_rules.items():
        if min_score <= score <= max_score:
            return category
    
    # If no rule matches, return unknown
    return 'unknown'

def add_swede_categories_to_dataframe(
    df: pd.DataFrame,
    swede_score_column: str = 'swede_score',
    category_column: str = 'swede_category',
    mapping_rules: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Add Swede categories to a DataFrame based on Swede scores.
    
    Args:
        df: DataFrame containing Swede scores
        swede_score_column: Name of the column containing Swede scores
        category_column: Name of the column to create for categories
        mapping_rules: Optional custom mapping rules
    
    Returns:
        pd.DataFrame: DataFrame with added category column
    """
    if swede_score_column not in df.columns:
        raise ValueError(f"Swede score column '{swede_score_column}' not found in DataFrame")
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Apply mapping to each row
    df_copy[category_column] = df_copy[swede_score_column].apply(
        lambda x: map_swede_score_to_category(x, mapping_rules)
    )
    
    # Log category distribution
    category_counts = df_copy[category_column].value_counts().to_dict()
    logger.info(f"Swede category distribution: {category_counts}")
    
    return df_copy

def get_swede_score_distribution(
    df: pd.DataFrame,
    swede_score_column: str = 'swede_score'
) -> Dict[int, int]:
    """
    Get the distribution of Swede scores in the dataset.
    
    Args:
        df: DataFrame containing Swede scores
        swede_score_column: Name of the column containing Swede scores
    
    Returns:
        Dict[int, int]: Mapping of score to count
    """
    if swede_score_column not in df.columns:
        raise ValueError(f"Swede score column '{swede_score_column}' not found in DataFrame")
    
    # Filter out NaN values
    valid_scores = df[swede_score_column].dropna()
    
    # Count occurrences of each score
    score_counts = valid_scores.value_counts().sort_index().to_dict()
    
    return score_counts

def validate_swede_scores(
    df: pd.DataFrame,
    swede_score_column: str = 'swede_score',
    valid_range: tuple = (0, 10)
) -> Dict[str, any]:
    """
    Validate Swede scores in the dataset.
    
    Args:
        df: DataFrame containing Swede scores
        swede_score_column: Name of the column containing Swede scores
        valid_range: Tuple of (min_score, max_score) for valid scores
    
    Returns:
        Dict: Validation statistics
    """
    if swede_score_column not in df.columns:
        raise ValueError(f"Swede score column '{swede_score_column}' not found in DataFrame")
    
    validation_stats = {
        'total_rows': len(df),
        'valid_scores': 0,
        'invalid_scores': 0,
        'missing_scores': 0,
        'out_of_range_scores': 0,
        'errors': []
    }
    
    min_score, max_score = valid_range
    
    for idx, score in df[swede_score_column].items():
        if pd.isna(score):
            validation_stats['missing_scores'] += 1
            validation_stats['errors'].append(f"Row {idx}: Missing Swede score")
        elif not isinstance(score, (int, float)):
            validation_stats['invalid_scores'] += 1
            validation_stats['errors'].append(f"Row {idx}: Invalid score type {type(score)}")
        elif score < min_score or score > max_score:
            validation_stats['out_of_range_scores'] += 1
            validation_stats['errors'].append(f"Row {idx}: Score {score} outside valid range {valid_range}")
        else:
            validation_stats['valid_scores'] += 1
    
    return validation_stats

def create_swede_summary(
    df: pd.DataFrame,
    swede_score_column: str = 'swede_score',
    category_column: str = 'swede_category'
) -> Dict[str, any]:
    """
    Create a comprehensive summary of Swede scores and categories.
    
    Args:
        df: DataFrame containing Swede scores and categories
        swede_score_column: Name of the column containing Swede scores
        category_column: Name of the column containing categories
    
    Returns:
        Dict: Summary statistics
    """
    summary = {
        'total_samples': len(df),
        'score_distribution': {},
        'category_distribution': {},
        'category_score_ranges': {},
        'statistics': {}
    }
    
    # Score distribution
    if swede_score_column in df.columns:
        valid_scores = df[swede_score_column].dropna()
        summary['score_distribution'] = valid_scores.value_counts().sort_index().to_dict()
        summary['statistics'] = {
            'mean': valid_scores.mean(),
            'median': valid_scores.median(),
            'std': valid_scores.std(),
            'min': valid_scores.min(),
            'max': valid_scores.max()
        }
    
    # Category distribution
    if category_column in df.columns:
        summary['category_distribution'] = df[category_column].value_counts().to_dict()
        
        # Score ranges for each category
        for category in df[category_column].unique():
            if pd.notna(category) and category != 'unknown':
                category_scores = df[df[category_column] == category][swede_score_column].dropna()
                if len(category_scores) > 0:
                    summary['category_score_ranges'][category] = {
                        'min': category_scores.min(),
                        'max': category_scores.max(),
                        'mean': category_scores.mean(),
                        'count': len(category_scores)
                    }
    
    return summary

def get_stratification_weights(
    df: pd.DataFrame,
    category_column: str = 'swede_category',
    target_weights: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Calculate stratification weights for balanced sampling.
    
    Args:
        df: DataFrame containing categories
        category_column: Name of the column containing categories
        target_weights: Optional target weights for each category
    
    Returns:
        Dict[str, float]: Mapping of category to weight
    """
    if category_column not in df.columns:
        raise ValueError(f"Category column '{category_column}' not found in DataFrame")
    
    # Get current distribution
    category_counts = df[category_column].value_counts()
    total_samples = len(df)
    
    current_weights = (category_counts / total_samples).to_dict()
    
    if target_weights is None:
        # Use inverse frequency weighting for balanced sampling
        weights = {}
        for category, count in category_counts.items():
            weights[category] = 1.0 / count if count > 0 else 0.0
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    else:
        return target_weights 