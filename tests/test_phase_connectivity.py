"""Tests for phase-based connectivity metrics: PLV, PLI and bandwise wrappers.

These tests verify:
 - shape and symmetry of PLV matrices
 - values are within [0, 1]
 - the high-level bandwise wrapper returns expected keys and shapes
"""

from __future__ import annotations
import numpy as np

from src.eeg.features.phase_connectivity import (
    plv_matrix,
    pli_matrix,
    bandwise_phase_connectivity,
)

RNG = np.random.default_rng(42)


def _sine_pair(
    n_channels: int = 4, n_samples: int = 1024, sfreq: int = 256
) -> np.ndarray:
    """
    Create a deterministic multi-channel signal where even channels share the same 10 Hz
    base (thus expected to be coherent), and odd channels vary slightly.
    """
    t = np.arange(n_samples) / float(sfreq)
    base = np.sin(2 * np.pi * 10 * t)
    data = np.zeros((n_channels, n_samples))
    for i in range(n_channels):
        if i % 2 == 0:
            data[i] = base
        else:
            data[i] = np.sin(2 * np.pi * (10.0 + i) * t) + 0.05 * RNG.normal(
                size=n_samples
            )
    return data


def test_plv_symmetry_and_range() -> None:
    """PLV matrix should be symmetric and values bounded in [0,1]."""
    data = _sine_pair()
    mat = plv_matrix(data, sfreq=256, band=(8, 12))
    assert mat.shape == (data.shape[0], data.shape[0])
    assert np.allclose(mat, mat.T, atol=1e-8)  # symmetry
    assert np.all((mat >= -1e-8) & (mat <= 1.0 + 1e-8))


def test_pli_basic_properties() -> None:
    """PLI should return a symmetric matrix of the right shape (values [0,1])."""
    data = _sine_pair()
    mat = pli_matrix(data, sfreq=256, band=(8, 12))
    assert mat.shape == (data.shape[0], data.shape[0])
    assert np.allclose(mat, mat.T, atol=1e-8)
    assert np.all((mat >= -1e-8) & (mat <= 1.0 + 1e-8))


def test_bandwise_connectivity_returns_metrics() -> None:
    """High-level wrapper returns metric keys and per-band matrices."""
    data = _sine_pair()
    out = bandwise_phase_connectivity(data, sfreq=256, metrics=("plv", "pli"))
    assert "plv" in out and "pli" in out
    # Expect at least one band produced per requested metric
    assert isinstance(out["plv"], list) and len(out["plv"]) >= 1
    assert out["plv"][0].shape == (data.shape[0], data.shape[0])
