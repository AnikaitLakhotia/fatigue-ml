"""Tests for preprocessing modules using small synthetic examples."""

import numpy as np
from src.eeg.preprocessing.epoching import make_epochs
from src.eeg.preprocessing.normalization import zscore_normalize_epochs

def _make_raw_like(n_channels=8, n_samples=1024):
    # fake epochs array rather than MNE object for unit-level tests
    t = np.arange(n_samples) / 256.0
    data = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        data[ch] = 100.0 * np.sin(2 * np.pi * (6 + ch) * t)
    return data

def test_zscore_normalize_epochs():
    # create epochs shape (2, channels, samples)
    epochs = np.stack([_make_raw_like(), _make_raw_like()])
    norm = zscore_normalize_epochs(epochs)
    assert norm.shape == epochs.shape
    # per-channel mean approx 0 over time
    means = norm.mean(axis=2)
    assert np.allclose(means, 0.0, atol=1e-6)
