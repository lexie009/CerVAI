# CerVAI Data Preprocessing Package

A modular Python package for preprocessing colposcopy images and segmentation masks for the CerVAI cervical cancer detection project.

## Overview

This package provides a comprehensive set of utilities for:
- **Resizing**: Resize images and masks while preserving aspect ratio
- **Filtering**: Validate and filter datasets for quality issues
- **Splitting**: Perform stratified train/validation/test splits
- **Swede Categories**: Map numerical Swede scores to categorical labels
- **Filename Management**: Consistent filename formatting and validation

## Package Structure

```
src/data_preprocessing/
├── __init__.py              # Package initialization and exports
├── resizing.py              # Image and mask resizing utilities
├── filtering.py             # Dataset filtering and validation
├── split_utils.py           # Train/val/test splitting with stratification
├── swede_category.py        # Swede score to category mapping
├── filename_utils.py        # Filename formatting and validation
├── example_usage.py         # Usage examples and demonstrations
└── README.md               # This documentation
```

## Installation

The package requires the following dependencies:

```bash
pip install pillow pandas numpy scikit-learn
```

Optional dependencies for advanced features:
```bash
pip install imagehash  # For repeat image detection
```

## Quick Start

### Basic Usage

```python
from src.data_preprocessing import (
    resize_with_padding,
    map_swede_score_to_category,
    perform_stratified_split
)

# Resize a single image
resized_image = resize_with_padding(
    img="path/to/image.jpg",
    size=(512, 512),
    is_mask=False
)

# Map Swede score to category
category = map_swede_score_to_category(7)  # Returns 'high'

# Perform stratified split
train_df, val_df, test_df = perform_stratified_split(
    df=your_dataframe,
    test_size=0.1,
    val_size=0.1,
    stratify_column='swede_category'
)
```

### Complete Pipeline Example

```python
import pandas as pd
from src.data_preprocessing import *

# Load data
df = pd.read_csv("aceto_mask_check_split.csv")

# Add Swede categories
df = add_swede_categories_to_dataframe(df, 'swede_score', 'swede_category')

# Filter dataset
filter_stats = filter_dataset(
    csv_path="aceto_mask_check_split.csv",
    images_dir="dataset/images",
    masks_dir="dataset/masks",
    check_sizes=True,
    check_empty_masks=True
)

# Resize images and masks
resize_stats = resize_dataset_batch(
    input_images_dir="dataset/images",
    input_masks_dir="dataset/masks",
    output_images_dir="dataset/resized/images",
    output_masks_dir="dataset/resized/masks",
    target_size=(512, 512)
)

# Perform stratified split
train_df, val_df, test_df = perform_stratified_split(
    df=df,
    test_size=0.1,
    val_size=0.1,
    stratify_column='swede_category'
)

# Create split directories and copy files
split_dirs = create_split_directories("dataset/splits")
copy_stats = copy_files_to_splits(
    train_df, val_df, test_df,
    source_images_dir="dataset/resized/images",
    source_masks_dir="dataset/resized/masks",
    split_dirs=split_dirs
)
```

## Module Documentation

### resizing.py

**Main Functions:**
- `resize_with_padding()`: Resize image/mask with aspect ratio preservation
- `resize_dataset_batch()`: Resize entire dataset in batch
- `verify_resized_files()`: Verify resized files have correct dimensions

**Example:**
```python
from src.data_preprocessing.resizing import resize_with_padding

# Resize single image
resized = resize_with_padding(
    img="image.jpg",
    size=(512, 512),
    is_mask=False,
    output_path="resized_image.png"
)
```

### filtering.py

**Main Functions:**
- `check_size_mismatch()`: Check if image and mask have matching dimensions
- `check_empty_mask()`: Check if mask is essentially empty
- `check_repeat_images()`: Detect duplicate or similar images
- `filter_dataset()`: Comprehensive dataset filtering

**Example:**
```python
from src.data_preprocessing.filtering import filter_dataset

stats = filter_dataset(
    csv_path="data.csv",
    images_dir="images/",
    masks_dir="masks/",
    check_sizes=True,
    check_empty_masks=True,
    check_repeats=True
)
```

### split_utils.py

**Main Functions:**
- `perform_stratified_split()`: Split dataset with stratification
- `create_split_directories()`: Create directory structure for splits
- `copy_files_to_splits()`: Copy files to respective split directories
- `verify_split_integrity()`: Verify split integrity

**Example:**
```python
from src.data_preprocessing.split_utils import perform_stratified_split

train_df, val_df, test_df = perform_stratified_split(
    df=df,
    test_size=0.1,
    val_size=0.1,
    stratify_column='swede_category'
)
```

### swede_category.py

**Main Functions:**
- `map_swede_score_to_category()`: Map numerical score to category
- `add_swede_categories_to_dataframe()`: Add categories to DataFrame
- `validate_swede_scores()`: Validate Swede scores
- `create_swede_summary()`: Create comprehensive summary

**Example:**
```python
from src.data_preprocessing.swede_category import map_swede_score_to_category

# Default mapping: 0-4=low, 5-6=mid, 7-10=high
category = map_swede_score_to_category(7)  # Returns 'high'

# Custom mapping
custom_rules = {'low': (0, 3), 'mid': (4, 6), 'high': (7, 10)}
category = map_swede_score_to_category(5, custom_rules)  # Returns 'mid'
```

### filename_utils.py

**Main Functions:**
- `generate_sequential_filenames()`: Generate sequential filenames
- `parse_filename_to_index()`: Extract index from filename
- `validate_filename_pattern()`: Validate filename format
- `verify_filename_consistency()`: Verify image-mask filename consistency

**Example:**
```python
from src.data_preprocessing.filename_utils import generate_sequential_filenames

image_files, mask_files = generate_sequential_filenames(
    start_index=1,
    count=100,
    padding=4
)
# Returns: ['0001.png', '0002.png', ...], ['0001_mask.png', '0002_mask.png', ...]
```

## Swede Score Categories

The package uses the following default mapping for Swede scores:

| Score Range | Category | Risk Level |
|-------------|----------|------------|
| 0-4        | low      | Low risk   |
| 5-6        | mid      | Medium risk|
| 7-10       | high     | High risk  |
| NaN/Invalid | unknown  | Unknown    |

## File Naming Convention

The package expects and generates filenames in the following format:
- **Images**: `0001.png`, `0002.png`, etc.
- **Masks**: `0001_mask.png`, `0002_mask.png`, etc.

## Error Handling

All functions include comprehensive error handling and logging:
- Invalid file paths are caught and logged
- Missing files are reported with details
- Data validation errors are captured
- Processing statistics are returned

## Logging

The package uses Python's logging module. Configure logging level as needed:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Testing

Run the example script to test the package:

```python
from src.data_preprocessing.example_usage import example_individual_functions
example_individual_functions()
```

## Contributing

When adding new functions:
1. Follow the existing docstring format
2. Include type hints
3. Add comprehensive error handling
4. Update the `__init__.py` exports
5. Add examples to `example_usage.py`

## Dependencies

- **pillow**: Image processing
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scikit-learn**: Stratified splitting
- **imagehash**: (Optional) Repeat image detection
