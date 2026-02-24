import numpy as np

# CSV column name constants
CSV_COL_IMAGE = 'Image'
CSV_COL_MEAN_BAD = 'Mean Bad Scores'
CSV_COL_STD_BAD = 'Std Deviation Bad Scores'
CSV_COL_FILTER_N = 'Filter N'


def normalize_l2(features):
    """L2-normalize feature vectors along axis=1."""
    return features / np.linalg.norm(features, axis=1, keepdims=True)
