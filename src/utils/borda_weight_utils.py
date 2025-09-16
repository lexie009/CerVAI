import numpy as np
import pandas as pd

def compute_weighted_borda_ranks(score_dicts, weights):
    """
    Compute weighted Borda count ranks for each sample.

    Args:
        score_dicts: A dict of {metric_name: [scores]}.
        weights: A dict of {metric_name: weight}. Must match score_dicts keys.

    Returns:
        np.ndarray: Array of final weighted ranks.
    """
    assert set(score_dicts.keys()) == set(weights.keys()), "Metric keys in scores and weights must match."
    n_samples = len(next(iter(score_dicts.values())))
    weighted_ranks = np.zeros(n_samples)

    for metric_name, scores in score_dicts.items():
        weight = weights[metric_name]
        ranks = pd.Series(scores).rank(ascending=False, method='average').to_numpy()
        weighted_ranks += weight * ranks

    return weighted_ranks


def select_top_k(samples, weighted_ranks, k=10):
    """
    Select top-k samples based on lowest weighted rank.

    Args:
        samples: list of sample identifiers
        weighted_ranks: np.ndarray of weighted Borda ranks
        k: number of samples to select

    Returns:
        list: top-k selected samples
    """
    top_indices = np.argsort(weighted_ranks)[:k]
    return [samples[i] for i in top_indices]


# Example usage - just for reminder
if __name__ == '__main__':
    # Suppose we have scores for 5 images on 3 metrics
    sample_ids = ["img1", "img2", "img3", "img4", "img5"]
    score_dicts = {
        'uncertainty': [0.9, 0.2, 0.6, 0.3, 0.5],
        'representativeness': [0.1, 0.3, 0.8, 0.4, 0.2],
        'texture': [0.5, 0.7, 0.3, 0.6, 0.4]
    }

    # Weight dynamically adjusted after round t
    dynamic_weights = {
        'uncertainty': 0.5,
        'representativeness': 0.3,
        'texture': 0.2
    }

    weighted_ranks = compute_weighted_borda_ranks(score_dicts, dynamic_weights)
    selected_samples = select_top_k(sample_ids, weighted_ranks, k=3)

    print("Weighted Ranks:", weighted_ranks)
    print("Selected Samples:", selected_samples)
