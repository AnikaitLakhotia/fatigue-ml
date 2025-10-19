"""Spectrogram / time-frequency helpers.

Compute per-channel STFT spectrograms for epochs, apply common normalizations,
and save spectrogram tensors in a disk-efficient format.

Provided functions:
  - compute_spectrogram(epoch, sfreq, nperseg=None, noverlap=None) -> (S, freqs, times)
  - log_normalize_spectrogram(S, eps=1e-12) -> S_log
  - save_spectrograms_npz(path, specs, freqs, times, metadata)
  - save_spectrograms_zarr(path, specs, freqs, times, metadata)  # optional, uses zarr if available
"""

from __future__ import annotations
from typing import Tuple, Dict, Any
from pathlib import Path
import numpy as np
from scipy import signal

# Canonical defaults
DEFAULT_NPERSEG = 256
DEFAULT_NOOVERLAP = 128


def compute_spectrogram(
    epoch: np.ndarray,
    sfreq: float,
    nperseg: int | None = None,
    noverlap: int | None = None,
    scaling: str = "density",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute STFT spectrogram per channel.

    Args:
        epoch: array (n_channels, n_samples)
        sfreq: sampling frequency (Hz)
        nperseg: nperseg for scipy.signal.spectrogram (default min(256, n_samples))
        noverlap: number of overlapping samples (default nperseg // 2)
        scaling: 'density' or 'spectrum'

    Returns:
        S: array (n_channels, n_freqs, n_times) (power)
        freqs: 1D freq vector (Hz)
        times: 1D time centers (s)
    """
    if epoch.ndim != 2:
        raise ValueError("epoch must be shape (n_channels, n_samples)")
    n_ch, n_samples = epoch.shape
    nperseg = int(nperseg or min(DEFAULT_NPERSEG, n_samples))
    noverlap = int(noverlap if noverlap is not None else nperseg // 2)

    S_list = []
    freqs = None
    times = None
    for ch in range(n_ch):
        f, t, Sxx = signal.spectrogram(
            epoch[ch],
            fs=float(sfreq),
            nperseg=nperseg,
            noverlap=noverlap,
            scaling=scaling,
        )
        if freqs is None:
            freqs, times = f, t
        S_list.append(Sxx)
    S = np.stack(S_list, axis=0)  # (n_channels, n_freqs, n_times)
    return S, freqs, times


def log_normalize_spectrogram(S: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Log-scale spectrogram and optionally z-score normalize across time.

    Args:
        S: (n_channels, n_freqs, n_times)
        eps: small constant to avoid log(0)

    Returns:
        S_log: same shape (n_channels, n_freqs, n_times)
    """
    return np.log10(S + eps)


def save_spectrograms_npz(
    path: Path,
    specs: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    metadata: Dict[str, Any] | None = None,
) -> None:
    """
    Save spectrogram tensor as compressed NPZ along with metadata.

    Args:
        path: output .npz path
        specs: array (n_epochs, n_channels, n_freqs, n_times) or (n_channels, n_freqs, n_times)
        freqs: 1D freq vector
        times: 1D time vector
        metadata: optional dict of scalars/strings (will be saved as JSON-like small dict)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path), specs=specs, freqs=freqs, times=times, metadata=metadata or {}
    )


def save_spectrograms_zarr(
    path: Path,
    specs: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    metadata: Dict[str, Any] | None = None,
) -> None:
    """
    Save spectrograms using zarr (chunked). Falls back to NPZ if zarr not installed.

    Args:
        path: output .zarr (directory) or fallback .npz
        specs: (n_epochs, n_channels, n_freqs, n_times)
    """
    try:
        import zarr  # type: ignore
    except Exception:
        save_spectrograms_npz(path.with_suffix(".npz"), specs, freqs, times, metadata)
        return

    root = zarr.open_group(str(path), mode="w")
    # choose chunking that keeps freq/time chunked sensibly
    root.create_dataset(
        "specs",
        data=specs,
        chunks=(1, specs.shape[1], specs.shape[2], specs.shape[3]),
        compressor=zarr.Blosc(cname="zstd", clevel=3),
    )
    root.create_dataset("freqs", data=freqs)
    root.create_dataset("times", data=times)
    root.attrs["metadata"] = metadata or {}
