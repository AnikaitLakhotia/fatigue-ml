from __future__ import annotations

"""
Power spectral density and band-power helpers (robust).

This module supplies:
  - compute_psd_welch: robust wrapper around MNE's Welch PSD with sensible defaults
  - bandpower_from_psd: integrate PSD over a frequency band with guards
  - bandpowers: compute canonical band powers + total (1–45 Hz)

Notes:
  - PSD values are floored to EPS to avoid exact zeros which cause downstream divide-by-zero.
  - If mne.time_frequency.psd_array_welch fails, a simple FFT fallback is used.
"""

from typing import Tuple, Dict
import numpy as np
import mne

# canonical bands (Hz)
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# tiny floor to avoid exact zeros
EPS = 1e-12


def compute_psd_welch(data: np.ndarray, sfreq: float, n_per_seg: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute PSD per channel using Welch with robust defaults and guards.

    Args:
        data: array, shape (n_channels, n_samples)
        sfreq: sampling frequency in Hz
        n_per_seg: desired nperseg for Welch. If None, defaults to min(256, n_samples).
                   If provided and larger than n_samples, it will be reduced.

    Returns:
        psd: ndarray shape (n_channels, n_freqs)
        freqs: ndarray shape (n_freqs,)
    """
    n_samples = int(data.shape[1])
    if n_per_seg is None:
        n_per_seg = min(256, n_samples)
    else:
        n_per_seg = min(int(n_per_seg), max(2, n_samples))

    # ensure a sensible n_overlap
    n_overlap = max(0, n_per_seg // 2)

    try:
        psd, freqs = mne.time_frequency.psd_array_welch(
            data,
            sfreq=sfreq,
            n_per_seg=n_per_seg,
            n_overlap=n_overlap,
            verbose=False,
        )
    except Exception:
        # Fallback to simple FFT-based periodogram if MNE call fails
        n_fft = max(2 ** int(np.ceil(np.log2(n_samples))), 256)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sfreq)
        psd_list = []
        for ch in range(data.shape[0]):
            x = data[ch].astype(float)
            X = np.fft.rfft(x, n_fft)
            Sxx = (np.abs(X) ** 2) / n_fft
            psd_list.append(Sxx[: len(freqs)])
        psd = np.vstack(psd_list)

    # defensive cleanup: replace NaN/inf, floor tiny values to EPS
    psd = np.nan_to_num(psd, nan=0.0, posinf=np.nanmax(psd) if np.isfinite(np.nanmax(psd)) else 0.0, neginf=0.0)
    psd = np.maximum(psd, EPS)

    return psd, freqs


def bandpower_from_psd(psd: np.ndarray, freqs: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    """
    Integrate PSD across a frequency band per channel.

    Args:
        psd: ndarray (n_channels, n_freqs)
        freqs: ndarray (n_freqs,)
        band: tuple (low, high) in Hz

    Returns:
        1D ndarray (n_channels,) with integrated band power (floored at EPS)
    """
    low, high = band
    idx = (freqs >= low) & (freqs <= high)
    if not idx.any():
        return np.zeros(psd.shape[0], dtype=float)
    vals = psd[:, idx].sum(axis=1)
    vals = np.maximum(vals, EPS)
    return vals


def bandpowers(psd: np.ndarray, freqs: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute canonical band powers and a 'total' band (1–45 Hz).

    Returns:
        dict mapping band name -> ndarray (n_channels,)
    """
    out: Dict[str, np.ndarray] = {}
    for name, rng in BANDS.items():
        out[name] = bandpower_from_psd(psd, freqs, rng)
    out["total"] = bandpower_from_psd(psd, freqs, (1.0, 45.0))
    return out
