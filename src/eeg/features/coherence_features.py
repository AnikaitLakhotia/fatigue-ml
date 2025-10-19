"""Connectivity measures: pairwise coherence per canonical band.

This module computes average pairwise magnitude-squared coherence per band
across all channel pairs. The implementation favors clarity and robustness
and avoids heavy dependencies at import time.
"""

from __future__ import annotations
from typing import Dict, List
import numpy as np
from scipy import signal

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def mean_pairwise_coherence(
    data: np.ndarray, sfreq: float, nperseg: int | None = None
) -> Dict[str, float]:
    """
    Compute mean pairwise magnitude-squared coherence (Cxy) per canonical band.

    Args:
        data: array of shape (n_channels, n_samples)
        sfreq: sampling frequency (Hz)
        nperseg: parameter passed to scipy.signal.coherence (defaults to min(256, n_samples))

    Returns:
        Mapping band name -> mean coherence (float). If no pairs/frequencies exist,
        the band value will be 0.0.
    """
    if data.ndim != 2:
        raise ValueError("data must be shape (n_channels, n_samples)")
    n_chan, n_samps = data.shape
    if nperseg is None:
        nperseg = min(256, n_samps)

    out: Dict[str, float] = {}
    # iterate bands and compute coherence for all unique pairs
    for band_name, (low, high) in BANDS.items():
        vals: List[float] = []
        for i in range(n_chan):
            xi = data[i]
            for j in range(i + 1, n_chan):
                xj = data[j]
                try:
                    f, Cxy = signal.coherence(
                        xi, xj, fs=float(sfreq), nperseg=int(nperseg)
                    )
                except Exception:
                    # if coherence fails for a pair, skip it
                    continue
                idx = (f >= low) & (f <= high)
                if idx.any():
                    vals.append(float(np.nanmean(Cxy[idx])))
        out[band_name] = float(np.nanmean(vals)) if vals else 0.0
    return out
