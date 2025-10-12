"""Tests for spectrogram patch extraction and simple vector-quantization helpers.

These tests are lightweight and only validate:
 - patch extraction shapes
 - sampling functionality
 - codebook fit and encoding (small k)
"""

from __future__ import annotations
import numpy as np

from src.eeg.features.tf_patches import (
    extract_patches_from_spectrogram,
    sample_patches_from_epochs,
    build_vq_codebook,
    encode_patches_to_tokens,
)


def _fake_spectrogram(n_channels: int = 2, n_freq: int = 40, n_time: int = 20) -> np.ndarray:
    """Return a reproducible random spectrogram-like tensor (abs of gaussian)."""
    rng = np.random.RandomState(0)
    return np.abs(rng.randn(n_channels, n_freq, n_time))


def test_extract_and_sample_patches_and_vq() -> None:
    """End-to-end small pipeline: patches -> sample -> codebook -> tokens."""
    S = _fake_spectrogram()
    patches = extract_patches_from_spectrogram(S, freq_bins=5, time_bins=4, stride_freq=2, stride_time=2)
    assert len(patches) > 0

    sampled = sample_patches_from_epochs([S], n_patches=8, freq_bins=5, time_bins=4, seed=1)
    assert sampled.ndim == 2  # (n_samples, patch_dim)
    # If we have at least 2 samples, fit a tiny codebook and encode
    if sampled.shape[0] >= 2:
        n_clusters = min(4, sampled.shape[0])
        codebook = build_vq_codebook(sampled, n_clusters=n_clusters, random_state=0)
        tokens = encode_patches_to_tokens(sampled, codebook)
        assert tokens.shape[0] == sampled.shape[0]
