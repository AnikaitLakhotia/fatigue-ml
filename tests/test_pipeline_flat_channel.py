"""Integration test: extractor handles a flat channel without producing zero totals."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import mne

from src.eeg.scripts.extract_features import process_single_fif


def test_pipeline_handles_flat_channel(tmp_path):
    """
    Create synthetic .fif with one flat (all-zero) channel. Run process_single_fif and
    assert that remaining channels have non-zero totals in the produced parquet.
    """
    sfreq = 256.0
    n_ch = 4
    n_samples = int(sfreq * 4)  # 4 s of data
    ch_names = [f"EEG{i}" for i in range(n_ch)]
    # random data, channel 2 is flat
    rng = np.random.RandomState(42)
    data = rng.randn(n_ch, n_samples) * 1e-6
    data[2, :] = 0.0  # flat channel
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info)
    # save to temp file
    fif_path = Path(tmp_path) / "synthetic_flat.fif"
    raw.save(fif_path, overwrite=True)
    out_dir = Path(tmp_path) / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run the extractor with per-channel features
    process_single_fif(fif_path, out_dir, window=2.0, overlap=0.5, per_channel=True)

    # find output parquet and check totals
    pqs = list(out_dir.glob("*_features.parquet"))
    assert len(pqs) == 1, "Expected exactly one features parquet produced"
    df = pd.read_parquet(pqs[0])
    total_cols = [c for c in df.columns if c.startswith("total_")]
    # Assert not all totals are zero (channels other than the flat one should have power)
    sums = df[total_cols].sum(axis=0)
    # at least one channel total should be > 0
    assert (sums > 0).any(), "All channel totals are zero — extractor failed to handle flat channel"
