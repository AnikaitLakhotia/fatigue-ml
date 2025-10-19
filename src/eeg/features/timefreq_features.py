"""Time-frequency utilities providing STFT spectrograms and band summaries."""

from __future__ import annotations
from typing import Tuple, Dict
import numpy as np
from scipy import signal

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def stft_spectrogram(
    data: np.ndarray, sfreq: float, nperseg: int | None = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute STFT spectrograms per channel and stack into array.

    Args:
        data: (n_channels, n_samples)
        sfreq: sampling frequency
        nperseg: nperseg for spectrogram

    Returns:
        S: (n_channels, n_freqs, n_times), freqs, times
    """
    nperseg = nperseg or min(256, data.shape[1])
    S_list = []
    freqs = None
    times = None
    for ch in range(data.shape[0]):
        f, t, Sxx = signal.spectrogram(
            data[ch], fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2
        )
        if freqs is None:
            freqs = f
            times = t
        S_list.append(Sxx)
    S = np.stack(S_list, axis=0)
    return S, freqs, times


def band_mean_from_spectrogram(S: np.ndarray, freqs: np.ndarray) -> Dict[str, float]:
    """
    Compute mean power per canonical band aggregated across channels and time.

    Args:
        S: (n_channels, n_freqs, n_times)
        freqs: frequency axis

    Returns:
        Dict band->scalar mean power
    """
    out: Dict[str, float] = {}
    for name, (low, high) in BANDS.items():
        idx = (freqs >= low) & (freqs <= high)
        out[name] = float(S[:, idx, :].mean()) if idx.any() else 0.0
    return out
