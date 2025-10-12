"""Unit tests for spectrogram helpers."""

from __future__ import annotations
import numpy as np

from src.eeg.features.spectrograms import compute_spectrogram

def test_compute_spectrogram_shape():
    n_ch = 4
    sfreq = 256
    n_samples = sfreq * 2  # 2 seconds
    t = np.arange(n_samples) / sfreq
    # build a multi-channel sweep signal
    epoch = np.stack([np.sin(2 * np.pi * (5 + ch) * t) for ch in range(n_ch)], axis=0)
    S, freqs, times = compute_spectrogram(epoch, sfreq, nperseg=128, noverlap=64)
    assert S.ndim == 3
    assert S.shape[0] == n_ch
    assert freqs.ndim == 1
    assert times.ndim == 1
