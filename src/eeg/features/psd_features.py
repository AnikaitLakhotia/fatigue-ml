"""Power spectral density and band-power helpers.

Provides:
  - compute_psd_welch: wrapper around mne.time_frequency.psd_array_welch
  - bandpower_from_psd: integrate PSD over frequency band
  - bandpowers: dictionary of canonical bands and total power
"""

from __future__ import annotations
from typing import Tuple, Dict
import numpy as np
import mne

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def compute_psd_welch(data: np.ndarray, sfreq: float, n_per_seg: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute PSD per channel using Welch.

    Args:
        data: (n_channels, n_samples)
        sfreq: sampling frequency in Hz
        n_per_seg: nperseg for Welch (defaults to min(256, n_samples))

    Returns:
        psd: shape (n_channels, n_freqs), freqs: shape (n_freqs,)
    """
    if n_per_seg is None:
        n_per_seg = min(256, data.shape[1])
    psd, freqs = mne.time_frequency.psd_array_welch(
        data, sfreq=sfreq, n_per_seg=n_per_seg, n_overlap=n_per_seg // 2, verbose=False
    )
    return psd, freqs


def bandpower_from_psd(psd: np.ndarray, freqs: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    """Integrate PSD across a frequency band per channel."""
    low, high = band
    idx = (freqs >= low) & (freqs <= high)
    if not idx.any():
        return np.zeros(psd.shape[0])
    return psd[:, idx].sum(axis=1)


def bandpowers(psd: np.ndarray, freqs: np.ndarray) -> Dict[str, np.ndarray]:
    """Return absolute band powers for canonical bands and the total band (1–45 Hz)."""
    out: Dict[str, np.ndarray] = {}
    for name, rng in BANDS.items():
        out[name] = bandpower_from_psd(psd, freqs, rng)
    out["total"] = bandpower_from_psd(psd, freqs, (1.0, 45.0))
    return out
