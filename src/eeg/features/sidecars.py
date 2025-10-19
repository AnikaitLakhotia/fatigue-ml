from __future__ import annotations

"""
Spectrogram & sliding-window connectivity helpers and sidecar writers.

Functions:
  - compute_spectrogram(epoch, sfreq, nperseg=None, noverlap=None) -> (S, freqs, times)
  - save_epoch_spectrograms(epoch, sfreq, out_path, nperseg=None, noverlap=None)
  - sliding_window_connectivity(epoch, sfreq, win_sec, step_sec, bands) -> (conn, times)
  - save_sliding_connectivity(epoch, sfreq, out_path, win_sec, step_sec, bands)
"""

from typing import Tuple, Iterable, Sequence
from pathlib import Path
import numpy as np
from scipy import signal
from src.eeg.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_NPERSEG = 256


def compute_spectrogram(
    epoch: np.ndarray,
    sfreq: float,
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute STFT spectrogram per channel for an epoch.

    Args:
        epoch: (n_channels, n_samples)
        sfreq: sampling frequency
        nperseg: STFT window length in samples (default min(256, n_samples))
        noverlap: overlap in samples (default nperseg // 2)

    Returns:
        S: spectrogram array (n_channels, n_freqs, n_times) (power)
        freqs: 1D frequency array (Hz)
        times: 1D time array (s)
    """
    nperseg = int(nperseg or min(DEFAULT_NPERSEG, epoch.shape[1]))
    noverlap = int(noverlap if noverlap is not None else (nperseg // 2))
    S_list = []
    f = None
    t = None
    for ch in range(epoch.shape[0]):
        f_ch, t_ch, Sxx = signal.spectrogram(
            epoch[ch], fs=sfreq, nperseg=nperseg, noverlap=noverlap, scaling="density"
        )
        if f is None:
            f = f_ch
            t = t_ch
        S_list.append(Sxx)
    S = np.stack(S_list, axis=0)
    return S, f, t


def save_epoch_spectrograms(
    epoch: np.ndarray,
    sfreq: float,
    out_path: str | Path,
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> Path:
    """
    Compute and save spectrogram for a single epoch as compressed .npz.

    The .npz will contain arrays:
      - 'S': spectrogram (n_channels, n_freqs, n_times)
      - 'freqs': freqs
      - 'times': times
      - 'sfreq': sfreq

    Returns:
        Path to saved .npz
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    S, freqs, times = compute_spectrogram(
        epoch, sfreq, nperseg=nperseg, noverlap=noverlap
    )
    np.savez_compressed(out_path, S=S, freqs=freqs, times=times, sfreq=sfreq)
    logger.info("Saved spectrogram sidecar %s", out_path)
    return out_path


def _band_indices(freqs: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    low, high = band
    return (freqs >= low) & (freqs <= high)


def sliding_window_connectivity(
    epoch: np.ndarray,
    sfreq: float,
    win_sec: float,
    step_sec: float,
    bands: Sequence[tuple[float, float]],
    nperseg: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute sliding-window connectivity (magnitude-squared coherence) across channel pairs and bands.

    Args:
        epoch: (n_channels, n_samples)
        sfreq: sampling rate
        win_sec: window length in seconds (for sliding window)
        step_sec: step/stride in seconds
        bands: sequence of (low, high) band tuples
        nperseg: nperseg for coherence (defaults to min(256, window_samples))

    Returns:
        conn: ndarray shape (n_windows, n_bands, n_channels, n_channels) containing average coherence per band
        times: ndarray of center time for each window (s)
    """
    n_samples = epoch.shape[1]
    win_samples = int(round(win_sec * sfreq))
    step_samples = int(round(step_sec * sfreq))
    if win_samples <= 0:
        raise ValueError("win_sec too small for given sfreq")
    if step_samples <= 0:
        step_samples = max(1, win_samples // 2)

    starts = list(range(0, max(1, n_samples - win_samples + 1), step_samples))
    times = np.array([(s + win_samples / 2) / sfreq for s in starts])
    n_windows = len(starts)
    n_bands = len(bands)
    n_ch = epoch.shape[0]

    conn = np.zeros((n_windows, n_bands, n_ch, n_ch), dtype=float)

    for wi, s in enumerate(starts):
        e = s + win_samples
        window_data = epoch[:, s:e]
        # compute pairwise coherence per pair and frequency
        # we'll compute coherence for all channel pairs via scipy.signal.coherence
        for i in range(n_ch):
            for j in range(i + 1, n_ch):
                try:
                    f, Cxy = signal.coherence(
                        window_data[i],
                        window_data[j],
                        fs=sfreq,
                        nperseg=nperseg or min(256, window_data.shape[1]),
                    )
                except Exception:
                    # fallback to zeros if coherence fails
                    Cxy = np.zeros((len(f),)) if "f" in locals() else np.zeros((1,))
                    f = f if "f" in locals() else np.array([0.0])
                for bi, band in enumerate(bands):
                    idx = _band_indices(f, band)
                    if idx.any():
                        mean_coh = float(np.mean(Cxy[idx]))
                    else:
                        mean_coh = 0.0
                    conn[wi, bi, i, j] = mean_coh
                    conn[wi, bi, j, i] = mean_coh
    return conn, times


def save_sliding_connectivity(
    epoch: np.ndarray,
    sfreq: float,
    out_path: str | Path,
    win_sec: float,
    step_sec: float,
    bands: Sequence[tuple[float, float]],
    nperseg: int | None = None,
) -> Path:
    """
    Compute sliding connectivity and save compressed .npz with arrays:
      - 'conn' : (n_windows, n_bands, n_channels, n_channels)
      - 'times' : (n_windows,)
      - 'bands' : (n_bands, 2)
      - 'sfreq' : scalar
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Computing sliding connectivity (win=%.2fs, step=%.2fs, bands=%s)",
        win_sec,
        step_sec,
        bands,
    )
    conn, times = sliding_window_connectivity(
        epoch, sfreq, win_sec, step_sec, bands, nperseg=nperseg
    )
    np.savez_compressed(
        out_path, conn=conn, times=times, bands=np.array(bands), sfreq=sfreq
    )
    logger.info("Saved connectivity sidecar %s", out_path)
    return out_path
