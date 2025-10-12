"""Epoching utilities (sliding windows).

This module wraps the lower-level epoch generation and attaches structured
metadata for downstream models (epoch index, start time, duration, session id
if available). The produced epochs are numpy arrays shaped (n_epochs, n_channels, n_samples).
"""

from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional, Tuple
import numpy as np
from omegaconf import DictConfig, OmegaConf
from src.eeg.utils.logger import get_logger

logger = get_logger(__name__)


def make_epochs(raw, cfg: Optional[Mapping] = None) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Generate sliding-window epochs from an MNE Raw object.

    Args:
        raw: MNE Raw object containing EEG data.
        cfg: Optional configuration mapping with "epoch" parameters:
            - length (float): window length in seconds.
            - overlap (float): overlap ratio between 0 and 1.

    Returns:
        Tuple of (epochs, starts, meta):
            epochs: ndarray of shape (n_epochs, n_channels, n_samples)
            starts: ndarray of start times (s)
            meta: list of metadata dictionaries
    """
    if cfg is None:
        cfg = {}
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)

    ep_cfg = cfg.get("epoch", {})
    win = float(ep_cfg.get("length", 10.0))
    overlap = float(ep_cfg.get("overlap", 0.5))
    stride = win * (1 - overlap)

    epochs, starts = sliding_window_epochs_from_raw(raw, window_sec=win, stride_sec=stride)
    meta = [
        {"epoch_index": i, "start_time_sec": float(s), "duration_sec": win}
        for i, s in enumerate(starts)
    ]

    logger.info(f"Created {len(meta)} epochs (win={win:.2f}s, stride={stride:.2f}s)")
    return epochs, starts, meta
