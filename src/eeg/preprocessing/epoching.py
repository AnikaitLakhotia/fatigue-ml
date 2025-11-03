# src/eeg/preprocessing/epoching.py
"""
Epoching utilities that preserve absolute timestamps.

Provides sliding-window epochs that read TIMESTAMP misc channel from Raw and
return epoch arrays plus metadata including start_ts, end_ts, and center_ts
(absolute timestamps in seconds since epoch).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import mne

TIMESTAMP_CHANNEL = "TIMESTAMP"


def sliding_window_epochs_from_raw(
    raw: mne.io.BaseRaw,
    window: float = 10.0,
    overlap: float = 0.5,
    picks_eeg: list | None = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Slice an MNE Raw into overlapping sliding windows and return epoch arrays
    together with per-epoch metadata including absolute timestamps.

    Args:
        raw: mne.io.BaseRaw that must contain the TIMESTAMP misc channel (or be compatible).
        window: window length in seconds.
        overlap: fraction overlap between windows in [0, 1).
        picks_eeg: optional list of channel indices (defaults to all EEG channels).

    Returns:
        epochs: np.ndarray shaped (n_epochs, n_channels, n_samples_per_epoch).
        meta: list of dicts for each epoch with keys:
            - epoch_index, start_idx, stop_idx
            - start_ts, end_ts, center_ts (seconds since epoch)
            - session_id, sfreq, n_channels, channel_names
    """
    if picks_eeg is None:
        picks_eeg = mne.pick_types(raw.info, eeg=True, misc=False)  # indices of EEG channels

    if TIMESTAMP_CHANNEL not in raw.ch_names:
        raise ValueError(f"Raw is missing required TIMESTAMP channel '{TIMESTAMP_CHANNEL}'")

    # index and array of timestamp channel
    ts_idx = raw.ch_names.index(TIMESTAMP_CHANNEL)
    ts_arr = raw.get_data(picks=[ts_idx])[0, :]

    sfreq = float(raw.info["sfreq"])
    n_samples = raw.n_times
    n_channels = len(picks_eeg)

    win_samps = int(round(window * sfreq))
    step_samps = int(round(win_samps * (1.0 - overlap)))
    if step_samps <= 0:
        raise ValueError("overlap too large; resulting step <= 0")

    epoch_list = []
    meta_list: List[Dict] = []
    start = 0
    idx = 0

    # Robust session_id extraction:
    subject_info = raw.info.get("subject_info", {})
    session_id = None
    if isinstance(subject_info, dict):
        # prefer 'his_id' (we set this in process_sessions); fallback to generic 'session_id'
        session_id = subject_info.get("his_id") or subject_info.get("session_id")

    # fallback: try parsing description for 'session_id=<value>'
    if session_id is None:
        desc = raw.info.get("description")
        if isinstance(desc, str) and "session_id=" in desc:
            try:
                session_id = desc.split("session_id=")[-1].split()[0].strip()
            except Exception:
                session_id = None

    while start + win_samps <= n_samples:
        stop = start + win_samps
        seg = raw.get_data(picks=picks_eeg, start=start, stop=stop)  # shape (n_channels, win_samps)
        epoch_list.append(seg.astype(np.float32))
        start_ts = float(ts_arr[start])
        end_ts = float(ts_arr[stop - 1])
        center_ts = float((start_ts + end_ts) / 2.0)
        meta_list.append(
            {
                "epoch_index": idx,
                "start_idx": int(start),
                "stop_idx": int(stop),
                "start_ts": start_ts,
                "end_ts": end_ts,
                "center_ts": center_ts,
                "session_id": session_id,
                "sfreq": sfreq,
                "n_channels": int(n_channels),
                "channel_names": [raw.ch_names[i] for i in picks_eeg],
            }
        )
        idx += 1
        start += step_samps

    if len(epoch_list) == 0:
        # return empty consistent shape
        return np.zeros((0, n_channels, win_samps), dtype=np.float32), meta_list

    epochs = np.stack(epoch_list, axis=0)
    return epochs, meta_list


def make_epochs(raw: mne.io.BaseRaw, window: float = 10.0, overlap: float = 0.5):
    """
    Convenience wrapper for sliding_window_epochs_from_raw.
    """
    return sliding_window_epochs_from_raw(raw, window=window, overlap=overlap)
