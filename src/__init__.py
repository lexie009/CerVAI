"""
CerVAI Source Package

This package provides data processing and dataset management tools for cervical cancer image analysis.
"""

from .dataset import CervixDataset, MaskReviewProcessor, create_datasets

__all__ = [
    'CervixDataset',
    'MaskReviewProcessor', 
    'create_datasets'
]

__version__ = "1.0.0"
__author__ = "Lexie Dai"
__description__ = "Cervical cancer image analysis toolkit"

