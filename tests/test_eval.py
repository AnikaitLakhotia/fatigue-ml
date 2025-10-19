# tests/test_eval.py
"""
Unit tests for evaluation helpers in src.eeg.models.eval
"""

import numpy as np
from src.eeg.models.eval import cluster_silhouette_score, cluster_temporal_contiguity


def test_silhouette_simple():
    # tiny synthetic clusters
    X = np.vstack(
        [
            np.random.randn(10, 2) + np.array([0, 0]),
            np.random.randn(10, 2) + np.array([5, 5]),
        ]
    )
    labels = np.array([0] * 10 + [1] * 10)
    s = cluster_silhouette_score(X, labels)
    assert s > 0.0


def test_temporal_contiguity():
    labels = [0, 0, 1, 1, -1, 1, 1, 0]
    order = list(range(len(labels)))
    cont = cluster_temporal_contiguity(labels, order)
    # there are transitions; contiguity should be between 0 and 1
    assert 0.0 <= cont <= 1.0
