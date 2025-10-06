"""Epoching and normalization helpers."""

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import mne
from src.eeg.utils.logger import get_logger

logger = get_logger(__name__)


def sliding_window_epochs_from_raw(raw: mne.io.Raw, window_sec: float = 10.0, stride_sec: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Create overlapping windows from a Raw object.

    Args:
        raw: Raw mne object.
        window_sec: window length in seconds (e.g., 10.0).
        stride_sec: stride in seconds; if None use 50% overlap.

    Returns:
        epochs: np.ndarray (n_windows, n_channels, n_samples)
        starts_sec: np.ndarray start times in seconds for each window
    """
    sfreq = float(raw.info["sfreq"])
    n_samples_win = int(round(window_sec * sfreq))
    if stride_sec is None:
        stride_sec = window_sec / 2.0
    step = int(round(stride_sec * sfreq))

    data = raw.get_data()
    n_times = data.shape[1]
    starts = list(range(0, n_times - n_samples_win + 1, step))
    if not starts:
        raise ValueError("Recording shorter than the requested window length")

    epochs = np.stack([data[:, s : s + n_samples_win] for s in starts], axis=0)
    starts_sec = np.array(starts, dtype=float) / sfreq
    logger.info("Epoched into %d windows: window=%fs step=%fs", epochs.shape[0], window_sec, stride_sec)
    return epochs.astype(np.float32), starts_sec


def zscore_normalize_epochs(epochs: np.ndarray, axis_sample: int = 2) -> np.ndarray:
    """Z-score per-epoch, per-channel across time axis."""
    mean = epochs.mean(axis=axis_sample, keepdims=True)
    std = epochs.std(axis=axis_sample, keepdims=True)
    return (epochs - mean) / (std + 1e-8)
