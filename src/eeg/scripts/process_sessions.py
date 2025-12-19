# src/eeg/scripts/process_sessions.py
"""
Convert combined CSV (per-sample rows with absolute timestamps) into per-session
MNE Raw .fif files that include a TIMESTAMP misc channel.

Keeps original timestamps from the CSV (converted to seconds since epoch).
Stores the session identifier safely in raw.info['subject_info']['his_id'] (MNE-validated).
Writes a small JSON sidecar with basic metadata.

Improvements in this commit:
- Avoid direct mutation of MNE private attribute `raw._data`. Use `apply_function`
  to cast buffers safely.
- Log and write metadata flag `sfreq_inferred` to explicitly indicate whether
  sampling frequency was inferred from timestamps or defaulted.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import mne

logger = logging.getLogger(__name__)


def _infer_sfreq_from_timestamps(ts: pd.Series) -> Optional[float]:
    """
    Infer sampling frequency (Hz) from a pandas Series of timestamps (datetime-like or numeric seconds).
    Returns None on failure.
    """
    try:
        if ts.size < 2:
            return None

        if np.issubdtype(ts.dtype, np.datetime64) or ts.dtype == object:
            ts_dt = pd.to_datetime(ts, errors="coerce")
            if not ts_dt.isna().any():
                secs = (ts_dt.view("int64") / 1e9).astype(float)
                diffs = np.diff(secs)
                med = float(np.median(diffs[np.nonzero(diffs)])) if np.any(diffs != 0) else None
            else:
                numeric = pd.to_numeric(ts, errors="coerce").astype(float)
                numeric = numeric[~np.isnan(numeric)]
                if numeric.size < 2:
                    return None
                diffs = np.diff(numeric)
                med = float(np.median(diffs[np.nonzero(diffs)])) if np.any(diffs != 0) else None
        else:
            numeric = pd.to_numeric(ts, errors="coerce").astype(float)
            numeric = numeric[~np.isnan(numeric)]
            if numeric.size < 2:
                return None
            if np.median(np.abs(numeric)) > 1e6:
                numeric = numeric / 1000.0
            diffs = np.diff(numeric)
            med = float(np.median(diffs[np.nonzero(diffs)])) if np.any(diffs != 0) else None
    except Exception:
        return None

    if med is None or med <= 0:
        return None
    return 1.0 / med


def write_session_fif(
    df_session: pd.DataFrame,
    out_dir: Path,
    session_id: str,
    ch_names: Optional[List[str]] = None,
    resample: bool = True,
    resample_sfreq: int = 128,
    dtype: str = "float32",
    overwrite: bool = True,
) -> Path:
    """
    Write a per-session .fif file.

    Args:
        df_session: pd.DataFrame with channel columns and optionally 'timestamp'.
        out_dir: Path where outputs will be saved.
        session_id: Identifier for the session (used in filenames and metadata).
        ch_names: Optional list of channel names to include (defaults to all non-reserved columns).
        resample: Whether to resample to `resample_sfreq` when original sfreq is higher.
        resample_sfreq: Target sampling frequency when resampling.
        dtype: 'float32' or 'float64' for saved FIF internal buffer.
        overwrite: If False and file exists, skip writing.

    Returns:
        Path to the written .fif file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reserved = {"timestamp", "session_id"}
    if ch_names is None:
        ch_list = [c for c in df_session.columns if c not in reserved]
    else:
        ch_list = [c for c in ch_names if c in df_session.columns]
    if not ch_list:
        raise ValueError("No channel columns found in session dataframe")

    # build data array (n_ch, n_samples)
    data = df_session.loc[:, ch_list].to_numpy(dtype=float).T  # (n_ch, n_samples)
    n_ch, n_samples = data.shape

    # infer sampling rate from timestamp if available
    sfreq = None
    sfreq_inferred = False
    ts_seconds = None
    if "timestamp" in df_session.columns:
        ts_col = df_session["timestamp"]
        try:
            ts_seconds_candidate = None
            if np.issubdtype(ts_col.dtype, np.datetime64) or ts_col.dtype == object:
                ts_dt = pd.to_datetime(ts_col, errors="coerce")
                if not ts_dt.isna().any():
                    ts_seconds_candidate = (ts_dt - ts_dt.iloc[0]).view("int64") / 1e9
                else:
                    numeric = pd.to_numeric(ts_col, errors="coerce").astype(float)
                    if not numeric.isna().any():
                        ts_seconds_candidate = numeric
            else:
                numeric = pd.to_numeric(ts_col, errors="coerce").astype(float)
                if not numeric.isna().any():
                    ts_seconds_candidate = numeric
            if ts_seconds_candidate is not None:
                ts_seconds = np.asarray(ts_seconds_candidate, dtype=float)
                if np.median(np.abs(ts_seconds)) > 1e6:
                    ts_seconds = ts_seconds / 1000.0
                ts_seconds = ts_seconds - float(ts_seconds[0])
                sfreq = _infer_sfreq_from_timestamps(pd.Series(ts_seconds))
                if sfreq is not None:
                    sfreq_inferred = True
        except Exception:
            sfreq = None
            ts_seconds = None

    if sfreq is None or not np.isfinite(sfreq) or sfreq <= 0:
        # Default fallback - explicit warning and metadata flag to indicate inference failed
        sfreq = 256.0
        sfreq_inferred = False
        logger.warning(
            "Could not infer sampling frequency for session %s; defaulting to %.1f Hz. "
            "Please check timestamps for this session.",
            session_id,
            float(sfreq),
        )

    if ts_seconds is None:
        ts_seconds = np.arange(n_samples, dtype=float) / float(sfreq)

    # append TIMESTAMP channel
    ts_channel = ts_seconds.reshape(1, -1)
    data_with_ts = np.vstack([data, ts_channel])
    final_ch_names = list(ch_list) + ["TIMESTAMP"]
    ch_types = ["eeg"] * len(ch_list) + ["misc"]

    info = mne.create_info(final_ch_names, sfreq, ch_types=ch_types)
    # Create RawArray with float64 (MNE expects float) then safely cast below
    raw = mne.io.RawArray(data_with_ts.astype(np.float64), info)

    # store session id in subject_info 'his_id' so downstream tools may recover it
    raw.info["subject_info"] = {"his_id": str(session_id)}
    # store description fallback
    raw.info["description"] = f"Converted from combined CSV; session_id={session_id}"

    # resample if beneficial
    if resample and sfreq > resample_sfreq:
        try:
            raw.resample(resample_sfreq, npad="auto")
            sfreq = resample_sfreq
        except Exception as exc:
            logger.warning("Resample failed for session %s: %s", session_id, exc)

    # Cast internal buffer to desired dtype using MNE public API (avoid private _data mutation).
    try:
        # apply_function will apply per-channel transformations in-place
        if dtype == "float32":
            # ensure underlying data is cast to float32 safely
            raw.apply_function(lambda arr: arr.astype(np.float32), picks="all")
        else:
            raw.apply_function(lambda arr: arr.astype(np.float64), picks="all")
    except Exception:
        logger.exception("Failed to cast raw data dtype using apply_function; proceeding with current dtype")

    out_path = out_dir / f"{session_id}_preprocessed_raw.fif"
    if out_path.exists() and not overwrite:
        logger.info("Skipping existing %s", out_path)
        return out_path

    raw.save(out_path, overwrite=overwrite)

    # write a small JSON sidecar for easy consumption
    meta = {
        "session_id": str(session_id),
        "sfreq": float(sfreq),
        "sfreq_inferred": bool(sfreq_inferred),
        "n_channels": len(ch_list),
    }
    (out_dir / f"{session_id}_preprocessed_meta.json").write_text(json.dumps(meta))

    logger.info(
        "Wrote %s (sfreq=%.3f Hz, n_samples=%d, channels=%s, sfreq_inferred=%s)",
        out_path,
        sfreq,
        raw.n_times,
        final_ch_names,
        sfreq_inferred,
    )
    return out_path


def process_sessions(
    csv_path: Path,
    out_dir: Path,
    *,
    ch_names: Optional[Iterable[str]] = None,
    resample: bool = True,
    resample_sfreq: int = 128,
    dtype: str = "float32",
    overwrite: bool = True,
) -> None:
    """Read combined CSV and write a .fif per unique session_id.

    Args:
        csv_path: Path to the combined CSV.
        out_dir: Directory to write per-session FIFs.
        ch_names: Optional iterable of channel names to select.
        resample: Whether to resample sessions.
        resample_sfreq: Target sampling rate if resampling.
        dtype: Float dtype to save internal buffers as.
        overwrite: Whether to overwrite existing outputs.

    Returns:
        None
    """
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    if "session_id" not in df.columns:
        raise ValueError("Input CSV must contain a 'session_id' column.")

    ch_list = list(ch_names) if ch_names else None

    for session_id, grp in df.groupby("session_id"):
        logger.info("Processing session: %s (rows=%d)", session_id, len(grp))
        write_session_fif(
            grp.reset_index(drop=True),
            out_dir,
            session_id=str(session_id),
            ch_names=ch_list,
            resample=resample,
            resample_sfreq=resample_sfreq,
            dtype=dtype,
            overwrite=overwrite,
        )


def _build_parser() -> "argparse.ArgumentParser":
    import argparse

    p = argparse.ArgumentParser(description="Convert combined CSV -> per-session .fif files.")
    p.add_argument("--input", "-i", required=True, help="Input combined CSV")
    p.add_argument("--out", "-o", required=True, help="Output directory for .fif files")
    p.add_argument("--channels", "-c", default=None, help="Comma-separated channel list (optional)")
    p.add_argument("--no-resample", action="store_true", help="Do not resample (preserve original sfreq)")
    p.add_argument("--resample-sfreq", default=128, type=int, help="Target sfreq when resampling (Hz)")
    p.add_argument("--dtype", default="float32", choices=("float32", "float64"), help="Numeric dtype for saved FIF")
    p.add_argument("--no-overwrite", action="store_true", help="Do not overwrite existing files")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    return p


def main():
    args = _build_parser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ch_names = None
    if args.channels:
        ch_names = [c.strip() for c in args.channels.split(",")]

    process_sessions(
        Path(args.input),
        Path(args.out),
        ch_names=ch_names,
        resample=not args.no_resample,
        resample_sfreq=args.resample_sfreq,
        dtype=args.dtype,
        overwrite=not args.no_overwrite,
    )


if __name__ == "__main__":
    main()