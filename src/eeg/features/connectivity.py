"""Connectivity (phase and amplitude) helpers.

Provides:
  - bandpass_filter(x, sfreq, band) -> filtered signal
  - compute_plv(x, y) -> plv scalar or vector over time
  - plv_matrix(data, sfreq, band) -> (n_channels, n_channels)
  - pli_matrix(...)
  - bandwise_connectivity(epoch, sfreq, bands=None, methods=('plv',)) -> dict[method] -> list of matrices
  - sliding_connectivity... (if required)
  - save_connectivity_npz(path, conn_dict, metadata)
"""

from __future__ import annotations
from typing import Dict, List, Tuple
from pathlib import Path
import numpy as np
from scipy.signal import hilbert, butter, sosfiltfilt, coherence

# Default canonical bands (reuse same as other modules)
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def _bandpass_sos(data: np.ndarray, sfreq: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Apply zero-phase bandpass filter using SOS (works on 1D arrays)."""
    sos = butter(order, [low / (0.5 * sfreq), high / (0.5 * sfreq)], btype="band", output="sos")
    return sosfiltfilt(sos, data)


def analytic_phase(signal: np.ndarray) -> np.ndarray:
    """Return instantaneous phase of 1D signal via Hilbert transform."""
    return np.angle(hilbert(signal))


def compute_plv(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Phase-Locking Value (PLV) between two analytic-phase signals.

    Args:
        x, y: 1D real-valued signals of equal length (assumed to be narrowbanded).

    Returns:
        scalar PLV in [0,1]
    """
    ph = np.angle(hilbert(x)) - np.angle(hilbert(y))
    return float(np.abs(np.mean(np.exp(1j * ph))))


def compute_pli(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Phase-Lag Index (PLI) between two signals.

    Returns:
        scalar in [0,1] (absolute value of mean sign of phase difference).
    """
    ph = np.angle(hilbert(x)) - np.angle(hilbert(y))
    return float(np.abs(np.mean(np.sign(np.sin(ph)))))


def plv_matrix(data: np.ndarray, sfreq: float, band: Tuple[float, float], nperseg: int | None = None) -> np.ndarray:
    """
    Compute pairwise PLV matrix for a single epoch (n_channels, n_samples) in given band.

    Returns:
        matrix (n_channels, n_channels) symmetric with diagonal ones.
    """
    n_ch = data.shape[0]
    low, high = band
    # bandpass each channel
    filtered = np.stack([_bandpass_sos(data[ch], sfreq, low, high) for ch in range(n_ch)], axis=0)
    phases = np.angle(hilbert(filtered))  # (n_ch, n_samples)
    mat = np.ones((n_ch, n_ch), dtype=float)
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            phdiff = phases[i] - phases[j]
            plv = float(np.abs(np.mean(np.exp(1j * phdiff))))
            mat[i, j] = mat[j, i] = plv
    return mat


def pli_matrix(data: np.ndarray, sfreq: float, band: Tuple[float, float]) -> np.ndarray:
    n_ch = data.shape[0]
    low, high = band
    filtered = np.stack([_bandpass_sos(data[ch], sfreq, low, high) for ch in range(n_ch)], axis=0)
    phases = np.angle(hilbert(filtered))
    mat = np.zeros((n_ch, n_ch), dtype=float)
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            phdiff = phases[i] - phases[j]
            pli = float(np.abs(np.mean(np.sign(np.sin(phdiff)))))
            mat[i, j] = mat[j, i] = pli
    return mat


def bandwise_phase_connectivity(data: np.ndarray, sfreq: float, metrics: Tuple[str, ...] = ("plv",), bands: Dict[str, Tuple[float, float]] | None = None) -> Dict[str, List[np.ndarray]]:
    """
    Compute connectivity per canonical band and per requested metric.

    Args:
        data: (n_channels, n_samples)
        metrics: tuple containing any of 'plv', 'pli', 'coh'
        bands: optional dict of bandname->(low,high); default BANDS

    Returns:
        dict mapping metric -> list of matrices (ordered by band_name keys)
    """
    if bands is None:
        bands = BANDS
    out: Dict[str, List[np.ndarray]] = {m: [] for m in metrics}
    for band_name, (low, high) in bands.items():
        for m in metrics:
            if m == "plv":
                out["plv"].append(plv_matrix(data, sfreq, (low, high)))
            elif m == "pli":
                out["pli"].append(pli_matrix(data, sfreq, (low, high)))
            elif m == "coh":
                # use scipy.signal.coherence averaged across pairs
                n_ch = data.shape[0]
                mat = np.zeros((n_ch, n_ch), dtype=float)
                for i in range(n_ch):
                    for j in range(i + 1, n_ch):
                        f, Cxy = coherence(data[i], data[j], fs=float(sfreq), nperseg=256)
                        idx = (f >= low) & (f <= high)
                        mat[i, j] = mat[j, i] = float(np.nanmean(Cxy[idx])) if idx.any() else 0.0
                out["coh"].append(mat)
            else:
                raise ValueError(f"Unknown connectivity metric: {m}")
    return out


def save_connectivity_npz(path: Path, conn_dict: Dict[str, List[np.ndarray]], metadata: Dict | None = None) -> None:
    """
    Save connectivity dict as compressed npz. conn_dict maps metric->list of matrices for bands.

    Args:
        path: .npz path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # convert to object arrays for storage
    np.savez_compressed(str(path), conn=conn_dict, metadata=metadata or {})
