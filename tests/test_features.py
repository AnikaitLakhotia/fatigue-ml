"""Unit tests for feature extraction (Day 4)."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from src.eeg.features.extract_features import extract_features_from_epochs, _psd_welch

def _synthetic_epochs(n_epochs=3, n_channels=8, n_samples=256*4, sfreq=256):
    """Create synthetic epochs: a mixture of theta (6 Hz) and alpha (10 Hz)."""
    t = np.arange(n_samples) / sfreq
    epochs = np.zeros((n_epochs, n_channels, n_samples), dtype=np.float32)
    for e in range(n_epochs):
        for ch in range(n_channels):
            # vary amplitude slightly per channel
            epochs[e, ch] = 50.0 * np.sin(2 * np.pi * 6 * t) + 20.0 * np.sin(2 * np.pi * 10 * t) + np.random.randn(n_samples) * 1.0
    return epochs

def test_psd_shape_and_values():
    epochs = _synthetic_epochs(n_epochs=1)
    data = epochs[0]
    psd, freqs = _psd_welch(data, sfreq=256)
    assert psd.shape[0] == data.shape[0]
    assert freqs.ndim == 1
    assert np.all(psd >= 0)

def test_extract_features_dataframe():
    epochs = _synthetic_epochs(n_epochs=2)
    df = extract_features_from_epochs(epochs, sfreq=256, per_channel=False)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 2
    # check a few expected columns exist
    for col in ["theta_power_mean", "alpha_power_mean", "theta_alpha_ratio", "spec_entropy_mean"]:
        assert col in df.columns, f"Missing expected feature column: {col}"

def test_extract_features_values_reasonable():
    epochs = _synthetic_epochs(n_epochs=2)
    df = extract_features_from_epochs(epochs, sfreq=256, per_channel=False)
    # theta should be non-negative
    assert (df["theta_power_mean"] >= 0).all()
    # spectral entropy in [0,1]
    assert ((df["spec_entropy_mean"] >= 0) & (df["spec_entropy_mean"] <= 1)).all()
