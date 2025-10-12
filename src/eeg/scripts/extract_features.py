"""
extract_features.py

Feature extraction pipeline for preprocessed EEG sessions (.fif → .parquet).
Computes Welch PSD, canonical band powers, and summary statistics per window.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from src.eeg.features.psd_features import compute_psd_welch, bandpowers
from src.eeg.utils.logger import get_logger

logger = get_logger(__name__)


def extract_features_from_session(raw_path: Path, out_dir: Path, window: float = 10.0, overlap: float = 0.5) -> None:
    """
    Compute features from a single preprocessed .fif session.

    Args:
        raw_path: Path to preprocessed .fif file.
        out_dir: Output directory for feature parquet.
        window: Window length in seconds.
        overlap: Fractional overlap (0–1).
    """
    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
    sfreq = raw.info["sfreq"]
    n_samples = raw.n_times
    step = int(window * (1 - overlap) * sfreq)
    n_win = max(1, (n_samples - int(window * sfreq)) // step)

    logger.info(f"Extracting features from {raw_path.name} — {n_win} windows, {len(raw.ch_names)} channels")

    features = []

    for w in range(n_win):
        start = w * step
        stop = start + int(window * sfreq)
        data, _ = raw[:, start:stop]
        psd, freqs = compute_psd_welch(data, sfreq)
        bp = bandpowers(psd, freqs)
        row = {f"{band}_{ch}": bp[band][i] for band in bp for i, ch in enumerate(raw.ch_names)}
        row["window_start"] = start / sfreq
        row["window_end"] = stop / sfreq
        features.append(row)

    if not features:
        logger.warning(f"No valid windows found for {raw_path.name}; skipping.")
        return

    df = pd.DataFrame(features)
    out_path = out_dir / f"{raw_path.stem}_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    logger.info(f"Saved features → {out_path} ({df.shape[0]} rows × {df.shape[1]} cols)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract EEG features from preprocessed .fif files")
    parser.add_argument("--input", type=str, required=True, help="Input folder with .fif files")
    parser.add_argument("--out", type=str, required=True, help="Output folder for features")
    parser.add_argument("--window", type=float, default=10.0, help="Window length (s)")
    parser.add_argument("--overlap", type=float, default=0.5, help="Window overlap fraction")
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.out)

    fif_files = sorted(in_dir.glob("*.fif"))
    if not fif_files:
        logger.warning(f"No .fif files found in {in_dir}")
        return

    for f in fif_files:
        extract_features_from_session(f, out_dir, args.window, args.overlap)

    logger.info(f"Feature extraction complete: processed {len(fif_files)} sessions")


if __name__ == "__main__":
    main()
