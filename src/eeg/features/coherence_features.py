"""Connectivity measures: pairwise coherence per canonical band.

This module computes average pairwise coherence (magnitude-squared) per band
across all channel pairs. It focuses on clarity and robustness rather than
extreme micro-optimizations.
"""

from __future__ import annotations
from typing import Dict
import numpy as np
from scipy import signal

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def mean_pairwise_coherence(data: np.ndarray, sfreq: float) -> Dict[str, float]:
    """Compute mean pairwise coherence per band.

    Args:
        data: (n_channels, n_samples)
        sfreq: sampling frequency

    Returns:
        dict mapping band name -> scalar mean coherence
    """
    n_chan = data.shape[0]
    out: Dict[str, float] = {}
    for band_name, (low, high) in BANDS.items():
        vals = []
        for i in range(n_chan):
            for j in range(i + 1, n_chan):
                f, Cxy = signal.coherence(data[i], data[j], fs=sfreq, nperseg=min(256, data.shape[1]))
                idx = (f >= low) & (f <= high)
                if idx.any():
                    vals.append(float(Cxy[idx].mean()))
        out[band_name] = float(np.nanmean(vals)) if vals else 0.0
    return out
