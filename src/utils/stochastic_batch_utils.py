# https://github.com/Minimel/StochasticBatchAL/blob/main/src/Utils/stochasticbatches_utils.py

from contextlib import AbstractContextManager
import numpy as np
from typing import Any, List, Tuple

def aggregate_group_uncertainty(groups: List[List[int]], uncertainty_values: List[float], aggregation: str) -> List[float]:
    """
    Aggregate uncertainty values over each group of indices.

    Args:
        groups: List of groups, each a list of sample indices
        uncertainty_values: List of uncertainty values per sample
        aggregation: 'mean', 'max', or 'sum'

    Returns:
        List of aggregated uncertainty values per group
    """
    if aggregation not in ['mean', 'max', 'sum']:
        raise ValueError("Invalid aggregation method. Supported methods are 'mean', 'max', and 'sum'.")

    aggregated_uncertainty = []

    for group in groups:
        group_uncertainties = [uncertainty_values[idx] for idx in group]

        if aggregation == 'mean':
            aggregated_value = np.mean(group_uncertainties)
        elif aggregation == 'max':
            aggregated_value = np.max(group_uncertainties)
        else:  # aggregation == 'sum'
            aggregated_value = np.sum(group_uncertainties)

        aggregated_uncertainty.append(aggregated_value)

    return aggregated_uncertainty

def generate_random_groups(indices: List[int], num_groups: int, group_size: int, resample: bool = False) -> List[List[int]]:
    """
    Randomly generate groups of indices.

    Args:
        indices: List of available sample indices
        num_groups: Number of groups to generate
        group_size: Size of each group
        resample: If True, samples may be reused

    Returns:
        List of groups (each a list of sample indices)
    """
    if resample:
        # Resampling allowed: we can use the same index in multiple groups
        groups = [np.random.choice(indices, group_size, replace=False).tolist() for _ in range(num_groups)]
    else:
        # Resampling not allowed: shuffle the indices and split into groups
        shuffled_indices = np.random.permutation(indices)
        groups = [shuffled_indices[i:i+group_size].tolist() for i in range(0, len(shuffled_indices), group_size)]

        # In case there are not enough indices to form num_groups, there will be less groups
        if len(groups[-1]) < group_size:
            groups.pop()

    return groups

def select_top_positions_with_highest_uncertainty(positions: List[List[int]], aggregated_uncertainty: List[float], num_groups: int) -> List[List[int]]:
    """
    Select top-k groups based on highest aggregated uncertainty.

    Args:
        positions: List of groups (each a list of sample indices)
        aggregated_uncertainty: Uncertainty score per group
        num_groups: Number of top groups to return

    Returns:
        List of top-k groups
    """
    top_indices = np.argsort(aggregated_uncertainty)[::-1][:num_groups]
    return [positions[i] for i in top_indices]
