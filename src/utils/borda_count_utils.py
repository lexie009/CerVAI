import pandas as pd

def borda_count_ranking(score_dict: dict, top_k: int = 20, ascending: bool = False):
    """
    Compute Borda count-based ranking across multiple score types.

    Args:
        score_dict (dict): {metric_name: {sample_id: score, ...}, ...}
        top_k (int): Number of top samples to return
        ascending (bool): Whether lower score is better (True) or higher is better (False)

    Returns:
        List[str]: List of top_k sample IDs selected by Borda count
    """
    if not score_dict:
        raise ValueError("Input score_dict is empty.")

    # Convert to DataFrame
    df = pd.DataFrame(score_dict)

    if df.isnull().values.any():
        raise ValueError("Missing values detected in score_dict. Check all sample IDs are present across metrics.")

    # Compute ranks for each metric
    ranks = df.rank(ascending=ascending, method='min')  # smaller rank is better

    # Sum ranks across metrics → lower Borda score is better
    borda_scores = ranks.sum(axis=1)

    # Return sample IDs with the lowest Borda scores
    return borda_scores.sort_values().index.tolist()[:top_k]


def load_scores_from_csv(csv_path: str, metric_columns: list, id_column: str = "sample_id") -> dict:
    """
    Load sample scores from a CSV file into a score_dict.

    Args:
        csv_path (str): Path to CSV file
        metric_columns (list): List of metric column names to include (e.g., ["uncertainty", "texture", "entropy"])
        id_column (str): Column name containing sample identifiers

    Returns:
        dict: {metric_name: {sample_id: score, ...}, ...}
    """
    df = pd.read_csv(csv_path)
    score_dict = {}

    for metric in metric_columns:
        if metric not in df.columns:
            raise ValueError(f"Metric column '{metric}' not found in CSV.")
        score_dict[metric] = dict(zip(df[id_column], df[metric]))

    return score_dict
