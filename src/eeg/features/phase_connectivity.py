"""Phase-based connectivity metrics (PLV, PLI, imaginary coherence).

Functions:
  - plv_matrix
  - pli_matrix
  - imag_coherence_matrix
  - bandwise_phase_connectivity
"""

from __future__ import annotations
from typing import Tuple, Iterable, List, Dict, Optional
import numpy as np
from scipy.signal import hilbert
import mne

from ..utils.logger import get_logger

logger = get_logger(__name__)

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def _bandpass_signal(
    sig: np.ndarray, sfreq: float, band: Tuple[float, float]
) -> np.ndarray:
    """Bandpass a 1D signal using MNE's filter."""
    l, h = band
    return mne.filter.filter_data(sig, sfreq, l_freq=l, h_freq=h, verbose=False)


def _analytic_signal_array(
    data: np.ndarray, sfreq: float, band: Tuple[float, float]
) -> np.ndarray:
    """
    Compute analytic signals per channel after bandpass.

    Args:
        data: (n_channels, n_samples)
    """
    n_chan = data.shape[0]
    analytic = np.empty(data.shape, dtype=complex)
    for ch in range(n_chan):
        bp = _bandpass_signal(data[ch], sfreq, band)
        analytic[ch] = hilbert(bp)
    return analytic


def plv_matrix(data: np.ndarray, sfreq: float, band: Tuple[float, float]) -> np.ndarray:
    """
    Compute phase-locking value (PLV) matrix for data in a given band.

    Args:
        data: (n_channels, n_samples)
        sfreq: sampling frequency
        band: (low, high)

    Returns:
        Symmetric (n_channels, n_channels) matrix with values in [0,1].
    """
    analytic = _analytic_signal_array(data, sfreq, band)
    phases = np.angle(analytic)
    n = phases.shape[0]
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            diff = phases[i] - phases[j]
            val = np.abs(np.mean(np.exp(1j * diff)))
            mat[i, j] = mat[j, i] = float(val)
    return mat


def pli_matrix(data: np.ndarray, sfreq: float, band: Tuple[float, float]) -> np.ndarray:
    """
    Compute Phase-Lag Index (PLI) matrix.

    Args:
        data, sfreq, band as above.

    Returns:
        Symmetric PLI matrix.
    """
    analytic = _analytic_signal_array(data, sfreq, band)
    phases = np.angle(analytic)
    n = phases.shape[0]
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            diff = phases[i] - phases[j]
            val = np.abs(np.mean(np.sign(np.sin(diff))))
            mat[i, j] = mat[j, i] = float(val)
    return mat


def imag_coherence_matrix(
    data: np.ndarray, sfreq: float, band: Tuple[float, float]
) -> np.ndarray:
    """
    Compute approximate imaginary coherence matrix.

    Returns:
        Symmetric matrix.
    """
    analytic = _analytic_signal_array(data, sfreq, band)
    phases = np.angle(analytic)
    n = phases.shape[0]
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            diff = phases[i] - phases[j]
            imc = np.abs(np.mean(np.imag(np.exp(1j * diff))))
            mat[i, j] = mat[j, i] = float(imc)
    return mat


def bandwise_phase_connectivity(
    data: np.ndarray,
    sfreq: float,
    bands: Optional[Iterable[Tuple[float, float]]] = None,
    metrics: Iterable[str] = ("plv", "pli", "imag"),
) -> Dict[str, List[np.ndarray]]:
    """
    Compute connectivity matrices per band for requested metrics.

    Args:
        data: (n_channels, n_samples)
        sfreq: Sampling frequency
        bands: Iterable of (low, high). Defaults to canonical BANDS.
        metrics: Iterable of metric names: 'plv', 'pli', 'imag'

    Returns:
        Dict mapping metric -> list of matrices (one per band)
    """
    if bands is None:
        bands = list(BANDS.values())
    out: Dict[str, List[np.ndarray]] = {m: [] for m in metrics}
    for band in bands:
        if "plv" in metrics:
            try:
                out["plv"].append(plv_matrix(data, sfreq, band))
            except Exception:
                logger.exception("PLV failed for band %s", band)
                out["plv"].append(np.zeros((data.shape[0], data.shape[0])))
        if "pli" in metrics:
            try:
                out["pli"].append(pli_matrix(data, sfreq, band))
            except Exception:
                logger.exception("PLI failed for band %s", band)
                out["pli"].append(np.zeros((data.shape[0], data.shape[0])))
        if "imag" in metrics:
            try:
                out["imag"].append(imag_coherence_matrix(data, sfreq, band))
            except Exception:
                logger.exception("Imag coherence failed for band %s", band)
                out["imag"].append(np.zeros((data.shape[0], data.shape[0])))
    return out
