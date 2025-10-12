"""Tests for preprocessing utilities: epoch generation and z-score normalization.
"""

from __future__ import annotations
import numpy as np
from src.eeg.preprocessing.epoching import sliding_window_epochs_from_raw, make_epochs
from src.eeg.preprocessing.normalization import zscore_normalize_epochs


def _make_raw_like(n_channels: int = 8, n_samples: int = 1024) -> np.ndarray:
    """Create a deterministic multi-channel sinusoidal signal (no MNE included)."""
    t = np.arange(n_samples) / 256.0
    data = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        data[ch] = 100.0 * np.sin(2 * np.pi * (6.0 + ch) * t)
    return data


def test_zscore_normalize_epochs() -> None:
    """Z-scoring returns same-shaped array and per-channel mean approx. 0."""
    # create 2 epochs (n_epochs, n_channels, n_samples)
    e1 = _make_raw_like()
    e2 = _make_raw_like()
    epochs = np.stack([e1, e2])
    norm = zscore_normalize_epochs(epochs)
    assert norm.shape == epochs.shape
    # per-channel mean over time should be close to zero for each epoch
    means = norm.mean(axis=2)
    assert np.allclose(means, 0.0, atol=1e-6)
    # std along time should be close to 1
    stds = norm.std(axis=2)
    assert np.allclose(stds, 1.0, atol=1e-5)
