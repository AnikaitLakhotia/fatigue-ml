"""Session-wise CSV ingestion and processing orchestration.

Splits `data/raw/combined_dataset.csv` by session_id (streaming, chunk-safe),
loads per-session CSV to an MNE RawArray, applies filtering/ICA/interpolation
and writes per-session FIF files into `data/interim/`.

Design goals:
- Fail fast on malformed input
- Stream large CSVs with pandas chunksize
- Produce one .fif per session for reproducibility
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional, Dict
import shutil
import tempfile

import pandas as pd
import numpy as np
import mne

from src.eeg.utils.logger import get_logger
from src.eeg.data.filters import apply_bandpass, apply_notch, set_reference
from src.eeg.data.artifacts import find_bad_channels_by_stats, run_ica_and_remove

logger = get_logger(__name__)

# Channel ordering expected in dataset
CANONICAL_CHANNELS = ["CP3", "C3", "F5", "PO3", "PO4", "F6", "C4", "CP4"]
DEFAULT_SFREQ = 256.0


def _make_tmpdir(base: Optional[Path] = None) -> Path:
    if base is None:
        return Path(tempfile.mkdtemp())
    base.mkdir(parents=True, exist_ok=True)
    return base


def split_combined_csv_by_session(
    combined_csv: Path,
    out_dir: Path,
    chunk_size: int = 1_000_000,
    session_col: str = "session_id",
) -> List[Path]:
    """Stream `combined_csv` and write per-session CSV files into `out_dir`.

    Args:
        combined_csv: input combined CSV path.
        out_dir: folder where per-session CSVs are written.
        chunk_size: number of rows per chunk to stream.
        session_col: name of session id column.

    Returns:
        List of per-session CSV Path objects.

    Raises:
        ValueError: if session_col missing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_files: Dict[str, Path] = {}
    logger.info("Splitting %s into per-session files in %s", combined_csv, out_dir)

    for chunk in pd.read_csv(combined_csv, chunksize=chunk_size):
        if session_col not in chunk.columns:
            raise ValueError(f"Session column '{session_col}' not found in combined CSV")
        for sid, grp in chunk.groupby(session_col):
            safe = str(sid).replace("/", "_")
            p = out_dir / f"session_raw_{safe}.csv"
            if not p.exists():
                grp.to_csv(p, index=False, mode="w")
                temp_files[safe] = p
            else:
                grp.to_csv(p, index=False, header=False, mode="a")
    logger.info("Split complete. Sessions: %d", len(temp_files))
    return list(temp_files.values())


def load_session_csv_to_raw(
    session_csv: Path,
    channels: Iterable[str] = CANONICAL_CHANNELS,
    timestamp_col: str = "timestamp",
    target_sfreq: Optional[float] = DEFAULT_SFREQ,
) -> tuple[mne.io.RawArray, Dict]:
    """Load a per-session CSV into an MNE RawArray with ordering & validation.

    Args:
        session_csv: path to session CSV.
        channels: expected channel names (subset of CSV cols).
        timestamp_col: name of timestamp column in ms.
        target_sfreq: expected sampling frequency (Hz). If None, infer.

    Returns:
        (raw, meta) where raw is RawArray and meta contains info like n_samples, sfreq.

    Raises:
        ValueError for missing columns or severe timestamp issues.
    """
    logger.info("Loading session file %s", session_csv)
    df = pd.read_csv(session_csv)
    missing = [c for c in channels if c not in df.columns]
    if missing:
        raise ValueError(f"Missing channels in {session_csv}: {missing}")
    if timestamp_col not in df.columns:
        raise ValueError(f"Missing timestamp column '{timestamp_col}' in {session_csv}")

    # order by timestamp
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    ts = df[timestamp_col].to_numpy(dtype=np.float64)
    diffs = np.diff(ts)
    if diffs.size == 0:
        raise ValueError(f"Session {session_csv} has no samples")
    if (diffs <= 0).any():
        logger.warning("Non-increasing timestamps detected in %s", session_csv)

    inferred_ms = float(np.median(diffs))
    inferred_sfreq = 1000.0 / inferred_ms if inferred_ms > 0 else None

    if target_sfreq is None:
        if inferred_sfreq is None:
            raise ValueError("Cannot infer sampling rate and no target provided")
        target_sfreq = inferred_sfreq
    else:
        if inferred_sfreq is not None and abs(inferred_sfreq - float(target_sfreq)) > 0.5:
            logger.warning("Inferred sfreq %.3f differs from expected %.3f for %s", inferred_sfreq, target_sfreq, session_csv)

    data = df[list(channels)].to_numpy(dtype=np.float32)  # (n_times, n_channels)
    data = data.T.copy()  # (n_channels, n_times)
    info = mne.create_info(ch_names=list(channels), sfreq=float(target_sfreq), ch_types=["eeg"] * len(channels))
    raw = mne.io.RawArray(data, info)
    meta = {"n_samples": raw.n_times, "sfreq": raw.info["sfreq"], "session_rows": len(df)}
    logger.info("Loaded %s: samples=%d sfreq=%.3f", session_csv.name, meta["n_samples"], meta["sfreq"])
    return raw, meta


def process_single_session(
    session_csv: Path,
    out_fif: Path,
    channels: Iterable[str] = CANONICAL_CHANNELS,
    target_sfreq: Optional[float] = DEFAULT_SFREQ,
    l_freq: float = 0.5,
    h_freq: float = 45.0,
    notch_freqs: Optional[Iterable[float]] = (50.0,),
    reference: Optional[str] = "average",
    ica_n_components: Optional[int] = None,
    ica_method: str = "fastica",
) -> None:
    """Process one session: load -> filter -> reference -> bad channel handling -> ICA -> save FIF."""
    raw, meta = load_session_csv_to_raw(session_csv, channels=channels, target_sfreq=target_sfreq)

    # filtering
    apply_bandpass(raw, l_freq=l_freq, h_freq=h_freq)
    if notch_freqs:
        apply_notch(raw, freqs=notch_freqs)

    # reference
    if reference is not None:
        set_reference(raw, reference)

    # bad channel detection & interpolation
    bads = find_bad_channels_by_stats(raw, flat_threshold=1e-8, noisy_z=6.0)
    if bads:
        logger.info("Marking bad channels: %s", bads)
        raw.info["bads"].extend(bads)
        try:
            raw.interpolate_bads(reset_bads=False)
        except Exception:
            logger.warning("Interpolation failed for %s", session_csv)

    # ICA
    try:
        raw_clean, ica = run_ica_and_remove(raw, n_components=ica_n_components, method=ica_method)
    except Exception:
        logger.exception("ICA failed for %s — proceeding with filtered data", session_csv)
        raw_clean = raw

    out_fif.parent.mkdir(parents=True, exist_ok=True)
    raw_clean.save(str(out_fif), overwrite=True)
    logger.info("Saved processed session to %s", out_fif)


def process_all_sessions_entrypoint(
    combined_csv: Path = Path("data/raw/combined_dataset.csv"),
    interim_dir: Path = Path("data/interim/"),
    tmp_dir: Optional[Path] = None,
) -> List[Path]:
    """Top-level helper to split and process all sessions. Cleans up temp files afterwards."""
    tmp = _make_tmpdir(tmp_dir)
    try:
        per_session = split_combined_csv_by_session(combined_csv, tmp)
        out_list: List[Path] = []
        for p in per_session:
            session_id = p.stem.replace("session_raw_", "")
            out_fif = interim_dir / f"{session_id}_preprocessed.fif"
            logger.info("Processing %s -> %s", p.name, out_fif.name)
            process_single_session(p, out_fif)
            out_list.append(out_fif)
        return out_list
    finally:
        try:
            shutil.rmtree(tmp)
            logger.info("Removed temporary dir %s", tmp)
        except Exception:
            logger.debug("Temporary cleanup failed for %s", tmp)
