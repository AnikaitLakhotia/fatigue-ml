# src/eeg/scripts/process_sessions.py
"""
Convert combined CSV (per-sample rows with absolute timestamps) into per-session
MNE Raw .fif files that include a TIMESTAMP misc channel.

- Keeps original timestamps from the CSV (converted to seconds since epoch).
- Stores the session identifier safely in raw.info['subject_info']['his_id'] (MNE-validated).
- Writes a small JSON sidecar with basic metadata.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import mne

log = logging.getLogger(__name__)

ELECTRODE_NAMES = ["CP3", "C3", "F5", "PO3", "PO4", "F6", "C4", "CP4"]
TIMESTAMP_CHANNEL = "TIMESTAMP"


def _ensure_numeric_timestamps(ts_series: Sequence) -> np.ndarray:
    """
    Convert a sequence of timestamps into seconds since epoch (float numpy array).

    Args:
        ts_series: column-like sequence (ints/floats/strings) from CSV.

    Returns:
        1D numpy array of timestamps in seconds.

    Raises:
        ValueError: if non-numeric values are present.
    """
    arr = pd.to_numeric(ts_series, errors="coerce").to_numpy()
    if np.isnan(arr).any():
        raise ValueError("Timestamps contain non-numeric values")
    # Convert milliseconds -> seconds if median is very large.
    if np.median(arr) > 1e12:
        arr = arr / 1000.0
    return arr.astype(float)


def process_sessions(input_csv: str | Path, out_dir: str | Path) -> None:
    """
    Read the combined CSV and write per-session preprocessed raw .fif files.

    Each output Raw contains:
      - EEG channels in ELECTRODE_NAMES
      - TIMESTAMP misc channel (seconds since epoch)
      - raw.info['subject_info']['his_id'] = session_id

    Args:
        input_csv: path to combined CSV; must contain columns for electrodes,
                   a 'timestamp' column and a 'session_id' column.
        out_dir: directory to write per-session .fif and JSON sidecar files.

    Returns:
        None
    """
    input_csv = Path(input_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Let pandas detect delim automatically (commas, tabs, etc.)
    df = pd.read_csv(input_csv, sep=None, engine="python")
    if "session_id" not in df.columns:
        raise ValueError("Input CSV must contain a 'session_id' column.")
    if "timestamp" not in df.columns:
        raise ValueError("Input CSV must contain a 'timestamp' column.")

    for session_id, g in df.groupby("session_id"):
        g = g.reset_index(drop=True)
        missing = [c for c in ELECTRODE_NAMES if c not in g.columns]
        if missing:
            raise ValueError(f"Missing expected channels for session {session_id}: {missing}")

        # Build data array: (n_channels, n_samples)
        data = np.vstack([g[name].to_numpy(dtype=float) for name in ELECTRODE_NAMES])
        ts_abs = _ensure_numeric_timestamps(g["timestamp"])
        if len(ts_abs) < 2:
            raise ValueError(f"Session {session_id} has fewer than 2 timestamped samples")

        # Robust sampling frequency estimation (median dt)
        dt = float(np.median(np.diff(ts_abs)))
        if dt <= 0:
            raise ValueError(f"Non-increasing timestamps in session {session_id}")
        sfreq = float(round(1.0 / dt, 6))

        # TIMESTAMP channel (1 x n_samples)
        timestamp_row = ts_abs.reshape(1, -1).astype(float)
        raw_data = np.vstack([data, timestamp_row])

        ch_names = list(ELECTRODE_NAMES) + [TIMESTAMP_CHANNEL]
        ch_types = ["eeg"] * len(ELECTRODE_NAMES) + ["misc"]

        info = mne.create_info(ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(raw_data, info, verbose=False)

        # Store session id in a validated SubjectInfo field 'his_id'
        raw.info["subject_info"] = {"his_id": str(session_id)}
        # Store description as a readable fallback containing the session id
        raw.info["description"] = f"Converted from combined CSV; TIMESTAMP channel preserved; session_id={session_id}"

        out_fif = out_dir / f"{session_id}_preprocessed_raw.fif"
        raw.save(out_fif, overwrite=True)

        meta = {"session_id": session_id, "sfreq": sfreq, "n_channels": len(ELECTRODE_NAMES)}
        (out_dir / f"{session_id}_preprocessed_meta.json").write_text(json.dumps(meta))
        log.info("Wrote %s  (sfreq=%.3f Hz, n_samples=%d)", out_fif, sfreq, raw.n_times)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Convert combined CSV to per-session .fif (includes TIMESTAMP channel).")
    parser.add_argument("--input", "-i", required=True, help="Path to combined CSV")
    parser.add_argument("--out", "-o", required=True, help="Output directory for per-session .fif files")
    args = parser.parse_args()
    process_sessions(args.input, args.out)
