"""Unit tests for core feature extraction CLI logic (lightweight)."""

from __future__ import annotations
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from src.eeg.features.extract_features import extract_features_from_epochs, _psd_welch


def _synthetic_epochs(n_epochs=2, n_channels=4, n_samples=256 * 4, sfreq=256):
    t = np.arange(n_samples) / sfreq
    epochs = np.zeros((n_epochs, n_channels, n_samples), dtype=np.float32)
    for e in range(n_epochs):
        for ch in range(n_channels):
            epochs[e, ch] = 20.0 * np.sin(2 * np.pi * (6 + ch) * t) + 5.0 * np.random.randn(n_samples)
    return epochs


def test_extract_basic_columns_and_types():
    epochs = _synthetic_epochs()
    df = extract_features_from_epochs(epochs, sfreq=256, per_channel=False, enabled=None)
    assert isinstance(df, pd.DataFrame)
    # required scalars
    for col in ["theta_power_mean", "alpha_power_mean", "theta_alpha_ratio", "spec_entropy_mean", "one_over_f_slope"]:
        assert col in df.columns
    # numeric sanity
    assert (df[["theta_power_mean", "alpha_power_mean"]] >= 0).all().all()
