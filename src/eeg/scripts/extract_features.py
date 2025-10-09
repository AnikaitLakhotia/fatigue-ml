"""CLI: Extract features from preprocessed per-session .fif files.

Usage (from repo root):
    python -m src.eeg.scripts.extract_features --input data/interim --out data/features --window 10 --overlap 0.5

This script:
- finds *.fif files in the input dir
- for each file: loads with mne, epochs with sliding windows, extracts features,
  and writes a per-session parquet: {out}/{session_id}_features.parquet
- logs progress using the project logger
"""

from __future__ import annotations
from pathlib import Path
import argparse
import mne
import pandas as pd
from tqdm import tqdm

from src.eeg.utils.logger import get_logger
from src.eeg.data.epoch import sliding_window_epochs_from_raw, zscore_normalize_epochs
from src.eeg.features.extract_features import extract_features_from_epochs

logger = get_logger(__name__)


def process_fif_file(fif_path: Path, out_dir: Path, window_sec: float, overlap: float, per_channel: bool):
    logger.info("Loading FIF: %s", fif_path)
    raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
    epochs, starts = sliding_window_epochs_from_raw(raw, window_sec=window_sec, stride_sec=window_sec * (1 - overlap))
    # optionally normalize epochs
    epochs = zscore_normalize_epochs(epochs)
    logger.info("Extracting features for %d epochs", epochs.shape[0])
    df = extract_features_from_epochs(epochs, sfreq=float(raw.info["sfreq"]), per_channel=per_channel, verbose=True)
    session_id = fif_path.stem
    out_path = out_dir / f"{session_id}_features.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Saved features to %s (rows=%d cols=%d)", out_path, df.shape[0], df.shape[1])
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extract features from interim .fif files.")
    parser.add_argument("--input", type=str, default="data/interim", help="Directory with .fif files")
    parser.add_argument("--out", type=str, default="data/features", help="Directory to write feature parquet files")
    parser.add_argument("--window", type=float, default=10.0, help="Epoch window length in seconds")
    parser.add_argument("--overlap", type=float, default=0.5, help="Epoch overlap fraction (0-1)")
    parser.add_argument("--per-channel", action="store_true", help="Include per-channel flattened features")
    args = parser.parse_args(argv)

    in_dir = Path(args.input)
    out_dir = Path(args.out)
    files = sorted(in_dir.glob("*.fif"))
    if not files:
        logger.warning("No .fif files found in %s", in_dir)
        return

    logger.info("Found %d .fif files to process", len(files))
    for f in tqdm(files):
        try:
            process_fif_file(f, out_dir, window_sec=args.window, overlap=args.overlap, per_channel=args.per_channel)
        except Exception as e:
            logger.exception("Failed to process %s: %s", f, e)

if __name__ == "__main__":
    main()
