# src/eeg/scripts/extract_features.py
"""
CLI to extract epoch-level features from preprocessed .fif files.

This script supports:
  - processing a single .fif (positional argument), or
  - processing all .fif files in a directory with --input / -i.

For each input .fif the script:
  - ensures a TIMESTAMP channel exists (synthesizes one if missing, for tests),
  - creates sliding-window epochs,
  - extracts features (via extract_features_from_epochs),
  - writes a parquet named <session_id>_features.parquet into the output directory.

Example usages:
  # process a single file
  python -m src.eeg.scripts.extract_features data/interim/6Yx..._preprocessed_raw.fif --out data/features

  # process all .fif files in a directory
  python -m src.eeg.scripts.extract_features --input data/interim --out data/features --window 10 --overlap 0.5
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

import mne
import numpy as np

from src.eeg.preprocessing.epoching import make_epochs
from src.eeg.features.extract_features import extract_features_from_epochs

log = logging.getLogger(__name__)


def _synthesize_timestamp_channel_if_missing(raw: mne.io.BaseRaw) -> None:
    """
    If TIMESTAMP is missing, synthesize a TIMESTAMP misc channel using sample indices and sfreq.
    This produces relative timestamps (seconds starting at 0.0) — intended for unit-test compatibility.
    """
    if "TIMESTAMP" in raw.ch_names:
        return
    sfreq = float(raw.info["sfreq"])
    ts = (np.arange(raw.n_times) / sfreq).astype(float).reshape(1, -1)
    ts_info = mne.create_info(["TIMESTAMP"], sfreq=sfreq, ch_types=["misc"])
    ts_raw = mne.io.RawArray(ts, ts_info, verbose=False)
    raw.add_channels([ts_raw], force_update_info=True)
    log.debug("Synthesized TIMESTAMP channel for %s (relative seconds).", raw.filenames if hasattr(raw, "filenames") else "raw")


def process_single_fif(
    fif_path: str | Path,
    out_dir: str | Path,
    window: float = 10.0,
    overlap: float = 0.5,
    per_channel: bool = True,
) -> Optional[Path]:
    """
    Process one .fif file: epoch, extract features, write parquet.

    Args:
        fif_path: path to a preprocessed .fif file.
        out_dir: directory where the output parquet will be written.
        window: epoch length in seconds.
        overlap: epoch overlap fraction (0 <= overlap < 1).
        per_channel: whether to compute per-channel features.

    Returns:
        Path to the written parquet file, or None if processing failed.
    """
    fif_path = Path(fif_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Processing %s", fif_path)
    try:
        raw = mne.io.read_raw_fif(str(fif_path), verbose=False)

        # Ensure TIMESTAMP channel exists (synthesize if missing for tests)
        _synthesize_timestamp_channel_if_missing(raw)

        epochs, meta = make_epochs(raw, window=window, overlap=overlap)
        if epochs.shape[0] == 0:
            log.warning("No epochs produced for %s (window=%s, overlap=%s)", fif_path, window, overlap)
            return None

        df = extract_features_from_epochs(epochs, epoch_meta=meta, per_channel=per_channel)

        # derive session_id and write parquet
        session_id = meta[0].get("session_id") if meta and meta[0].get("session_id") else fif_path.stem
        out_pq = out_dir / f"{session_id}_features.parquet"
        df.to_parquet(out_pq)
        log.info("Wrote features %s (n_epochs=%d)", out_pq, len(df))
        return out_pq
    except Exception as exc:
        log.exception("Failed to process %s: %s", fif_path, exc)
        return None


def _find_fif_files_in_dir(d: Path) -> List[Path]:
    """
    Return a sorted list of .fif files in directory d (non-recursive).
    """
    d = Path(d)
    return sorted([p for p in d.glob("*.fif") if p.is_file()])


def _process_many(files: Iterable[Path], out_dir: Path, window: float, overlap: float, per_channel: bool) -> List[Path]:
    """
    Process multiple files and return list of successfully written parquet paths.
    """
    written = []
    for f in files:
        res = process_single_fif(f, out_dir, window=window, overlap=overlap, per_channel=per_channel)
        if res:
            written.append(res)
    return written


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Extract epoch-level features from .fif (file or directory).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", "-i", help="Input directory containing .fif files (processes all *.fif)", metavar="DIR")
    group.add_argument("fif", nargs="?", help="Single input .fif file to process (positional)")

    parser.add_argument("--out", "-o", required=True, help="Output directory for parquet files")
    parser.add_argument("--window", type=float, default=10.0, help="Epoch window length in seconds (default: 10.0)")
    parser.add_argument("--overlap", type=float, default=0.5, help="Epoch overlap fraction in [0,1) (default: 0.5)")
    parser.add_argument("--per-channel", action="store_true", default=True, help="Compute per-channel features (default: True)")

    args = parser.parse_args()

    out_dir = Path(args.out)
    window = float(args.window)
    overlap = float(args.overlap)
    per_channel = bool(args.per_channel)

    if args.input:
        d = Path(args.input)
        if not d.exists() or not d.is_dir():
            parser.error(f"--input path {d} does not exist or is not a directory")
        files = _find_fif_files_in_dir(d)
        if not files:
            log.warning("No .fif files found in %s", d)
        written = _process_many(files, out_dir, window=window, overlap=overlap, per_channel=per_channel)
        log.info("Completed processing. Wrote %d parquet files.", len(written))
    else:
        if not args.fif:
            parser.error("Either provide --input DIR or a single positional fif file")
        fif_path = Path(args.fif)
        if not fif_path.exists():
            parser.error(f"fif file {fif_path} does not exist")
        res = process_single_fif(fif_path, out_dir, window=window, overlap=overlap, per_channel=per_channel)
        if res:
            log.info("Completed processing file: %s", res)
        else:
            log.error("Processing failed for %s", fif_path)
