"""Epoching utilities for sliding-window epochs.

Generates numpy epochs (n_epochs, n_channels, n_samples).
"""

from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional, Tuple
import numpy as np
from omegaconf import DictConfig, OmegaConf
from ..utils.logger import get_logger

logger = get_logger(__name__)


def sliding_window_epochs_from_raw(
    raw, window_sec: float = 10.0, stride_sec: float | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding-window epochs from an MNE Raw object.

    Args:
        raw: MNE Raw object (preloaded).
        window_sec: Window length in seconds.
        stride_sec: Step length in seconds. If None, stride = window_sec (non-overlapping).

    Returns:
        epochs: ndarray (n_epochs, n_channels, n_samples)
        starts: ndarray of start times in seconds
    """
    sfreq = raw.info["sfreq"]
    n_samples = raw.n_times
    win_samp = int(round(window_sec * sfreq))
    if stride_sec is None:
        stride_sec = window_sec
    step = int(round(stride_sec * sfreq))
    starts = list(range(0, n_samples - win_samp + 1, step))
    epochs = []
    starts_sec = []
    for s in starts:
        e = s + win_samp
        data, _ = raw[:, s:e]
        epochs.append(data.copy())
        starts_sec.append(s / sfreq)
    logger.info(
        "Created %d epochs (window=%.2fs, stride=%.2fs)",
        len(epochs),
        window_sec,
        stride_sec,
    )
    return (
        np.stack(epochs, axis=0)
        if epochs
        else np.empty((0, raw.info["nchan"], win_samp))
    ), np.array(starts_sec)


def make_epochs(
    raw, cfg: Optional[Mapping] = None
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    High-level wrapper to generate epochs and metadata.

    Args:
        raw: MNE Raw object.
        cfg: Optional config mapping. Expects cfg['epoch'] with keys 'length' and 'overlap'.

    Returns:
        (epochs, starts, meta) where:
          - epochs: ndarray (n_epochs, n_channels, n_samples)
          - starts: ndarray of start times (s)
          - meta: list of metadata dicts for each epoch
    """
    if cfg is None:
        cfg = {}
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    ep_cfg = cfg.get("epoch", {})
    win = float(ep_cfg.get("length", 10.0))
    overlap = float(ep_cfg.get("overlap", 0.5))
    stride = win * (1 - overlap)
    epochs, starts = sliding_window_epochs_from_raw(
        raw, window_sec=win, stride_sec=stride
    )
    meta = [
        {"epoch_index": int(i), "start_time_sec": float(s), "duration_sec": win}
        for i, s in enumerate(starts)
    ]
    logger.info("make_epochs: created %d epochs", len(meta))
    return epochs, starts, meta
