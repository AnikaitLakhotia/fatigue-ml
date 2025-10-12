"""Tests for Phase-Amplitude Coupling (PAC) utilities.

Verifies that the comodulogram builder returns the expected shape and values in a valid range.
"""

from __future__ import annotations
import numpy as np

from src.eeg.features.pac import comodulogram_epoch

RNG = np.random.default_rng(0)


def _synthetic_epoch(n_channels: int = 4, n_samples: int = 1024, sfreq: int = 256) -> np.ndarray:
    """
    Build a synthetic multi-channel epoch containing low-frequency phase (6 Hz)
    and high-frequency amplitude (40 Hz) components plus small noise.
    """
    t = np.arange(n_samples) / float(sfreq)
    epoch = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        epoch[ch] = np.sin(2 * np.pi * 6 * t) + 0.5 * np.sin(2 * np.pi * 40 * t) + 0.02 * RNG.normal(size=n_samples)
    return epoch


def test_comodulogram_shape_and_range() -> None:
    """Comodulogram should be shape (len(phase_bands), len(amp_bands)) with values in [0,1]."""
    epoch = _synthetic_epoch()
    phase_bands = [(4, 8), (8, 12)]
    amp_bands = [(30, 45), (45, 60)]
    C = comodulogram_epoch(epoch, sfreq=256, phase_bands=phase_bands, amp_bands=amp_bands, n_bins=12)
    assert C.shape == (len(phase_bands), len(amp_bands))
    # Allow a tiny numerical slack
    assert np.all(C >= -1e-8) and np.all(C <= 1.0 + 1e-8)
