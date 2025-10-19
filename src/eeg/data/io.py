"""EEG I/O and preprocessing helpers.

This module contains robust, typed helpers for:
  - split_combined_csv_by_session
  - load_csv_as_raw
  - process_single_session
  - run_preprocess_stage

All public functions document parameters, returns and raised exceptions.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, List
import tempfile
import pandas as pd
import mne

from ..preprocessing.filters import apply_bandpass, apply_notch, set_reference
from ..preprocessing.artifact_removal import run_ica_and_remove
from ..utils.logger import get_logger

logger = get_logger(__name__)


def split_combined_csv_by_session(
    csv_path: str | Path, out_dir: Optional[str | Path] = None
) -> List[Path]:
    """
    Split a combined CSV into per-session CSV files grouped by 'session_id'.

    Args:
        csv_path: Path to the combined EEG CSV.
        out_dir: Optional output directory. If None, a temporary dir is created.

    Returns:
        List of Path objects pointing to written session CSV files.

    Raises:
        ValueError: if 'session_id' column is missing.
        FileNotFoundError: if csv_path does not exist.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.error("CSV not found: %s", csv_path)
        raise FileNotFoundError(csv_path)
    out_dir = Path(out_dir or tempfile.mkdtemp(prefix="sessions_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "session_id" not in df.columns:
        raise ValueError("CSV must contain a 'session_id' column")
    session_files: List[Path] = []
    for session_id, g in df.groupby("session_id"):
        p = out_dir / f"{session_id}.csv"
        g.to_csv(p, index=False)
        session_files.append(p)

    logger.info("Split complete. Sessions written: %d", len(session_files))
    return session_files


def load_csv_as_raw(session_csv: Path, sfreq: float = 256.0) -> mne.io.Raw:
    """
    Load a single-session CSV into an MNE RawArray.

    CSV must contain EEG columns (non-timestamp/session_id).

    Args:
        session_csv: Path to per-session CSV file.
        sfreq: Sampling frequency in Hz.

    Returns:
        mne.io.Raw object containing the EEG channels.

    Raises:
        ValueError: If no EEG columns are found.
        FileNotFoundError: If session_csv does not exist.
    """
    session_csv = Path(session_csv)
    if not session_csv.exists():
        logger.error("Session CSV not found: %s", session_csv)
        raise FileNotFoundError(session_csv)
    df = pd.read_csv(session_csv)
    eeg_cols = [c for c in df.columns if c not in ("timestamp", "session_id")]
    if not eeg_cols:
        raise ValueError(f"No EEG columns in {session_csv}")

    data = df[eeg_cols].T.values.astype(float)
    info = mne.create_info(ch_names=eeg_cols, sfreq=float(sfreq), ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


def process_single_session(
    session_csv: Path, out_fif: Path, sfreq: float = 256.0
) -> None:
    """
    Full processing for a single session CSV: load, filter, re-reference, ICA, and save to FIF.

    Args:
        session_csv: Path to per-session CSV file.
        out_fif: Output .fif file path.
        sfreq: Sampling frequency in Hz.

    Returns:
        None

    Raises:
        Exception: If any processing step fails (logged).
    """
    logger.info("Processing %s -> %s", session_csv.name, out_fif.name)
    raw = load_csv_as_raw(session_csv, sfreq)
    raw = apply_bandpass(raw, l_freq=1.0, h_freq=45.0)
    raw = apply_notch(raw, freqs=[50.0])
    raw = set_reference(raw, reference="average")
    raw = run_ica_and_remove(raw)
    out_fif.parent.mkdir(parents=True, exist_ok=True)
    raw.save(out_fif, overwrite=True)
    logger.info("Saved preprocessed FIF: %s", out_fif)


def run_preprocess_stage(
    input_csv: str | Path, out_dir: str | Path, sfreq: float = 256.0
) -> None:
    """
    Orchestrate preprocessing for combined CSV: split, process, save .fif files.

    Args:
        input_csv: Combined CSV path.
        out_dir: Output directory for .fif files.
        sfreq: Sampling frequency.

    Returns:
        None
    """
    session_csvs = split_combined_csv_by_session(input_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not session_csvs:
        logger.warning("No sessions found — check 'session_id' column")
        return
    for csv_file in session_csvs:
        sid = csv_file.stem
        out_fif = out_dir / f"{sid}_preprocessed.fif"
        try:
            process_single_session(csv_file, out_fif, sfreq)
        except Exception:
            logger.exception("Failed processing session %s", sid)
    logger.info("Preprocessing stage complete: %d sessions", len(session_csvs))
