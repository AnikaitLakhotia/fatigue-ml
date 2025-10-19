"""Microstate analysis helpers (simple KMeans microstate segmentation).

Functions:
  - compute_microstates(topographies, n_states, random_state=None)
  - microstate_coverage(labels) -> dict with occupancy/duration stats
Notes:
  - topographies: (n_samples, n_channels) instantaneous scalp maps (per-sample).
  - requires reasonable channel count and good preprocessing.
"""

from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np
from sklearn.cluster import KMeans


def compute_microstates(
    topographies: np.ndarray, n_states: int = 4, random_state: int | None = None
) -> Dict[str, Any]:
    """
    Cluster instantaneous topographies into microstate maps and return segmentation.

    Args:
        topographies: (n_samples, n_channels) per-sample scalp maps (must be pre-referenced)
        n_states: number of microstate maps to find

    Returns:
        dict with keys:
          - "maps": (n_states, n_channels) cluster centroids (normalized)
          - "labels": (n_samples,) int labels per sample
          - "coverage": dict with occupancy fraction per state
          - "mean_duration": mean duration per state in samples
    """
    if topographies.ndim != 2:
        raise ValueError("topographies must be shape (n_samples, n_channels)")
    # Normalize each map to unit norm to focus on topographic pattern (not amplitude)
    norms = np.linalg.norm(topographies, axis=1, keepdims=True) + 1e-12
    X = topographies / norms
    km = KMeans(n_clusters=n_states, random_state=random_state)
    labels = km.fit_predict(X)
    maps = km.cluster_centers_
    # Compute coverage
    coverage = {int(k): float(np.mean(labels == k)) for k in range(n_states)}
    # Mean duration: compute runs
    durations = {int(k): [] for k in range(n_states)}
    # iterate runs
    prev = labels[0]
    runlen = 1
    for l in labels[1:]:
        if l == prev:
            runlen += 1
        else:
            durations[prev].append(runlen)
            runlen = 1
            prev = l
    durations[prev].append(runlen)
    mean_duration = {
        k: float(np.mean(durations[k])) if durations[k] else 0.0 for k in durations
    }
    return {
        "maps": maps,
        "labels": labels,
        "coverage": coverage,
        "mean_duration": mean_duration,
    }
