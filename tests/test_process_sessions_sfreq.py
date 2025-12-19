# tests/test_process_sessions_sfreq.py
"""Tests for write_session_fif sfreq inference behavior and sidecar metadata."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from src.eeg.scripts.process_sessions import write_session_fif


def test_write_session_fif_defaults_and_meta(tmp_path: Path):
    """If timestamps are absent or cannot be inferred, write_session_fif should default sfreq and record flag."""
    # build a minimal synthetic dataframe without timestamp column
    n_samples = 128
    n_ch = 4
    ch_names = [f"ch{i}" for i in range(n_ch)]
    data = (np.random.RandomState(0).randn(n_samples, n_ch) * 1e-6).astype(float)
    df = pd.DataFrame(data, columns=ch_names)
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    session_id = "test_sess_default_sfreq"
    out_path = write_session_fif(df, out_dir, session_id=session_id, ch_names=ch_names, resample=False, dtype="float32", overwrite=True)

    # check file exists and meta written
    assert out_path.exists()
    meta_path = out_dir / f"{session_id}_preprocessed_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert "sfreq" in meta
    assert "sfreq_inferred" in meta
    assert meta["sfreq_inferred"] is False
    assert float(meta["sfreq"]) == 256.0