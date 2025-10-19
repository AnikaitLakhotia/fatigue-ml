from __future__ import annotations

"""
Command-line feature extraction driver.

This module loads preprocessed .fif files, handles flat-channel detection &
interpolation, constructs epochs, calls the feature registry, and writes
parquet + optional sidecars.

Exports:
  - process_single_fif(raw_path: Path, out_dir: Path, ...) -> None
  - CLI via main()
"""

import argparse
from pathlib import Path
import json
from typing import Optional, List, Dict, Any, Sequence, Tuple

import numpy as np
import pandas as pd
import mne

from src.eeg.utils.logger import get_logger
from src.eeg.preprocessing.epoching import make_epochs

# import the feature extraction functions from the features module
from src.eeg.features.extract_features import extract_all_features
from src.eeg.features.sidecars import save_epoch_spectrograms, save_sliding_connectivity

logger = get_logger(__name__)

DEFAULT_BANDS = [
    ("delta", (1.0, 4.0)),
    ("theta", (4.0, 8.0)),
    ("alpha", (8.0, 12.0)),
    ("beta", (13.0, 30.0)),
    ("gamma", (30.0, 45.0)),
]


def _detect_flat_or_bad_channels(
    raw: mne.io.BaseRaw, std_thresh: float = 1e-12
) -> List[str]:
    try:
        data = raw.get_data(picks="eeg")
    except Exception:
        data = raw.get_data()
    ch_stds = np.std(data, axis=1)
    bads = []
    for i, s in enumerate(ch_stds):
        name = raw.ch_names[i]
        if not np.isfinite(s) or s <= std_thresh:
            bads.append(name)
    return bads


def process_single_fif(
    raw_path: Path,
    out_dir: Path,
    window: float,
    overlap: float,
    per_channel: bool = False,
    save_spectrograms: bool = False,
    save_connectivity: bool = False,
    connectivity_win: Optional[float] = None,
    connectivity_step: Optional[float] = None,
    connectivity_bands: Optional[Sequence[Tuple[float, float]]] = None,
) -> None:
    """
    Process a single .fif into features and optional sidecars.

    See src.eeg.features.extract_features.extract_all_features for feature behavior.
    """
    logger.info("Processing %s", raw_path)
    raw = mne.io.read_raw_fif(raw_path, preload=False, verbose=False)

    raw.load_data()
    bads = _detect_flat_or_bad_channels(raw)
    if bads:
        logger.warning("Detected flat/bad channels: %s", bads)
        try:
            montage = raw.get_montage()
        except Exception:
            montage = None

        if montage is not None and montage.get_positions():
            for b in bads:
                if b not in raw.info["bads"]:
                    raw.info["bads"].append(b)
            try:
                raw.interpolate_bads(reset_bads=True)
                logger.info("Interpolated bad channels: %s", bads)
            except Exception:
                logger.exception("Interpolation failed for %s, dropping instead", bads)
                for b in bads:
                    if b in raw.ch_names:
                        raw.drop_channels([b])
        else:
            logger.warning("No montage: dropping bad channels: %s", bads)
            for b in bads:
                if b in raw.ch_names:
                    raw.drop_channels([b])

    sfreq = float(raw.info.get("sfreq", 256.0))
    cfg = {"epoch": {"length": float(window), "overlap": float(overlap)}}
    try:
        epochs, starts, _meta = make_epochs(raw, cfg=cfg)
    except Exception:
        logger.exception("make_epochs failed; falling back to manual sliding windows")
        data = raw.get_data()
        n_samples = data.shape[1]
        win_samples = int(round(window * sfreq))
        step_samples = max(1, int(round(win_samples * (1 - overlap))))
        starts = list(range(0, max(1, n_samples - win_samples + 1), step_samples))
        epochs = np.stack([data[:, s : s + win_samples] for s in starts], axis=0)

    if epochs.size == 0:
        logger.warning("No epochs extracted for %s — skipping.", raw_path)
        return

    try:
        df = extract_all_features(epochs, sfreq, per_channel=per_channel)
    except TypeError:
        df = extract_all_features(epochs, sfreq)

    # metadata + QC
    df["session_id"] = raw_path.stem
    df["sfreq"] = sfreq
    df["n_channels"] = len(raw.ch_names)
    df["channel_names"] = repr(raw.info.get("ch_names", []))
    total_cols = [c for c in df.columns if c.startswith("total_")]
    if total_cols:
        df["num_channels_with_zero_total"] = df[total_cols].apply(
            lambda r: int((r == 0).sum()), axis=1
        )
        df["has_any_zero_total"] = df["num_channels_with_zero_total"] > 0
    else:
        df["num_channels_with_zero_total"] = 0
        df["has_any_zero_total"] = False

    # sanitize
    if df.isin([np.inf, -np.inf]).any().any() or df.isnull().any().any():
        n_inf = int(df.replace([np.inf, -np.inf], np.nan).isnull().sum().sum())
        logger.warning("Found %d inf/NaN values; filling with 0.0", n_inf)
        df = df.fillna(0.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    parq_path = out_dir / f"{raw_path.stem}_features.parquet"
    df.to_parquet(parq_path, index=False)
    logger.info("Saved %s", parq_path)

    manifest = {"session": raw_path.stem, "n_rows": int(df.shape[0]), "columns": []}
    for c in df.columns:
        manifest["columns"].append({"name": c, "dtype": str(df[c].dtype)})
    manifest_path = out_dir / f"{raw_path.stem}_feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Saved manifest %s", manifest_path)

    # optional sidecars
    if save_spectrograms or save_connectivity:
        sidecar_dir = out_dir / f"{raw_path.stem}_sidecars"
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        bands = (
            [rng for _, rng in DEFAULT_BANDS]
            if connectivity_bands is None
            else connectivity_bands
        )
        connectivity_win = connectivity_win or window
        connectivity_step = connectivity_step or (window * (1 - overlap))

        for i in range(epochs.shape[0]):
            epoch = epochs[i]
            prefix = sidecar_dir / f"epoch_{i:04d}"
            if save_spectrograms:
                try:
                    save_epoch_spectrograms(
                        epoch, sfreq, str(prefix) + "_spectrogram.npz"
                    )
                except Exception:
                    logger.exception("Failed spectrogram for epoch %d", i)
            if save_connectivity:
                try:
                    save_sliding_connectivity(
                        epoch,
                        sfreq,
                        str(prefix) + "_connectivity.npz",
                        win_sec=connectivity_win,
                        step_sec=connectivity_step,
                        bands=bands,
                    )
                except Exception:
                    logger.exception("Failed connectivity for epoch %d", i)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="extract_features")
    parser.add_argument("--input", required=True, help="Directory with .fif files")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--window", type=float, default=10.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--per-channel", action="store_true")
    parser.add_argument("--save-spectrograms", action="store_true")
    parser.add_argument("--save-connectivity", action="store_true")
    parser.add_argument("--conn-win", type=float, default=None)
    parser.add_argument("--conn-step", type=float, default=None)
    parser.add_argument("--conn-bands", type=str, default=None)
    args = parser.parse_args(argv)

    in_dir = Path(args.input)
    out_dir = Path(args.out)
    fif_files = sorted(in_dir.glob("*.fif"))
    if not fif_files:
        logger.warning("No .fif files found in %s", in_dir)
        return

    conn_bands = None
    if args.conn_bands:
        parts = [p.strip() for p in args.conn_bands.split(";") if p.strip()]
        lst = []
        for p in parts:
            a, b = p.split("-")
            lst.append((float(a), float(b)))
        conn_bands = lst

    for f in fif_files:
        try:
            process_single_fif(
                f,
                out_dir,
                window=args.window,
                overlap=args.overlap,
                per_channel=args.per_channel,
                save_spectrograms=args.save_spectrograms,
                save_connectivity=args.save_connectivity,
                connectivity_win=args.conn_win,
                connectivity_step=args.conn_step,
                connectivity_bands=conn_bands,
            )
        except Exception:
            logger.exception("Failed processing %s", f)


if __name__ == "__main__":
    main()
