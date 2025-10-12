"""EEG I/O and preprocessing helpers.

Provides:
  - split_combined_csv_by_session: split combined CSV into per-session files
  - load_csv_as_raw: load one session CSV into an MNE Raw object
  - process_single_session: run filtering, reference, ICA, and save
  - run_preprocess_stage: orchestrate full preprocessing pipeline
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, List
import tempfile
import pandas as pd
import mne

from src.eeg.preprocessing.filters import apply_bandpass, apply_notch, set_reference
from src.eeg.preprocessing.artifact_removal import run_ica_and_remove
from src.eeg.utils.logger import get_logger

logger = get_logger(__name__)


def split_combined_csv_by_session(csv_path: str | Path, out_dir: Optional[str | Path] = None) -> List[Path]:
    """Split combined CSV into per-session files grouped by `session_id`.

    Args:
        csv_path: path to combined EEG CSV
        out_dir: optional output directory (defaults to tmp)

    Returns:
        List of paths to written session CSVs
    """
    csv_path = Path(csv_path)
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

    logger.info(f"Split complete. Sessions written: {len(session_files)}")
    return session_files


def load_csv_as_raw(session_csv: Path, sfreq: float = 256.0) -> mne.io.Raw:
    """Load a single session CSV as an MNE RawArray.

    Args:
        session_csv: path to per-session CSV
        sfreq: sampling frequency (Hz)

    Returns:
        MNE Raw object with EEG channels
    """
    df = pd.read_csv(session_csv)
    eeg_cols = [c for c in df.columns if c not in ("timestamp", "session_id")]
    if not eeg_cols:
        raise ValueError(f"No EEG columns in {session_csv}")

    data = df[eeg_cols].T.values
    info = mne.create_info(ch_names=eeg_cols, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


def process_single_session(session_csv: Path, out_fif: Path, sfreq: float = 256.0) -> None:
    """Process one EEG session: load, filter, re-ref, ICA, and save.

    Args:
        session_csv: path to per-session CSV
        out_fif: output .fif path
        sfreq: sampling frequency (Hz)
    """
    logger.info(f"Processing {session_csv.name} → {out_fif.name}")

    raw = load_csv_as_raw(session_csv, sfreq)
    raw = apply_bandpass(raw, l_freq=1.0, h_freq=40.0)
    raw = apply_notch(raw, freqs=[50.0])
    raw = set_reference(raw, reference="average")
    raw = run_ica_and_remove(raw)

    out_fif.parent.mkdir(parents=True, exist_ok=True)
    raw.save(out_fif, overwrite=True)
    logger.info(f"Saved {out_fif.name}")


def run_preprocess_stage(input_csv: str | Path, out_dir: str | Path, sfreq: float = 256.0) -> None:
    """Preprocess all sessions from combined EEG CSV.

    Args:
        input_csv: combined dataset path
        out_dir: output directory for preprocessed files
        sfreq: sampling frequency (Hz)
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
        except Exception as e:
            logger.error(f"Failed session {sid}: {e}")

    logger.info(f"Preprocessing complete: {len(session_csvs)} sessions")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Preprocess EEG sessions from combined CSV.")
    p.add_argument("--input", required=True, help="Path to combined_dataset.csv")
    p.add_argument("--out", required=True, help="Output directory for .fif files")
    p.add_argument("--sfreq", type=float, default=256.0, help="Sampling frequency (Hz)")
    a = p.parse_args()

    run_preprocess_stage(a.input, a.out, a.sfreq)
