"""Phase-Amplitude Coupling (PAC) utilities.

Contains:
  - comodulogram_epoch(epoch, sfreq, phase_bands, amp_bands, n_bins=18, method='tort')
  - simple MVL-based comod index fallback
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
from scipy.signal import hilbert, butter, sosfiltfilt

def _bandpass(epoch: np.ndarray, sfreq: float, low: float, high: float, order: int = 4) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt
    sos = butter(order, [low / (0.5 * sfreq), high / (0.5 * sfreq)], btype="band", output="sos")
    return sosfiltfilt(sos, epoch)


def _phase_amplitude(series, sfreq, phase_band, amp_band):
    """Return phase (1D) and amplitude envelope (1D) for a single channel."""
    ph = _bandpass(series, sfreq, phase_band[0], phase_band[1])
    am = _bandpass(series, sfreq, amp_band[0], amp_band[1])
    phase = np.angle(hilbert(ph))
    amp = np.abs(hilbert(am))
    return phase, amp


def _modulation_index_tort(phase, amp, n_bins=18):
    """
    Compute Tort's modulation index (Kullback-Leibler based).
    Implementation: bin phases, compute mean amp per bin, compute normalized KL divergence.
    """
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    idx = np.digitize(phase, bin_edges) - 1
    amp_means = np.array([amp[idx == b].mean() if np.any(idx == b) else 0.0 for b in range(n_bins)])
    amp_means = amp_means + 1e-12
    p = amp_means / amp_means.sum()
    # KL divergence to uniform
    kl = np.sum(p * np.log(p * len(p)))
    mi = kl / np.log(len(p))
    return float(mi)


def comodulogram_epoch(epoch: np.ndarray, sfreq: float, phase_bands: List[Tuple[float, float]], amp_bands: List[Tuple[float, float]], n_bins: int = 18) -> np.ndarray:
    """
    Compute comodulogram for multi-channel epoch by averaging per-channel comod indices.

    Args:
        epoch: (n_channels, n_samples)
        phase_bands: list of (low, high) for phase
        amp_bands: list of (low, high) for amplitude
        n_bins: number of phase bins used in Tort MI

    Returns:
        C: array shape (len(phase_bands), len(amp_bands)) of average MI across channels
    """
    n_ch = epoch.shape[0]
    out = np.zeros((len(phase_bands), len(amp_bands)), dtype=float)
    for i_pb, pb in enumerate(phase_bands):
        for j_ab, ab in enumerate(amp_bands):
            vals = []
            for ch in range(n_ch):
                phase, amp = _phase_amplitude(epoch[ch], sfreq, pb, ab)
                try:
                    mi = _modulation_index_tort(phase, amp, n_bins=n_bins)
                except Exception:
                    mi = 0.0
                vals.append(mi)
            out[i_pb, j_ab] = float(np.nanmean(vals)) if vals else 0.0
    return out
