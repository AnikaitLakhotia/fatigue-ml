# tests/test_data_schema.py
"""Unit tests for data schema validation helper."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.eeg.data.validation import validate_parquet

def _make_epoch_df(n_epochs: int = 3):
    rows = []
    for i in range(n_epochs):
        rows.append(
            {
                "epoch_index": int(i),
                "session_id": "synth",
                "sfreq": 128.0,
                "n_channels": 4,
                "start_ts": float(i * 10.0),
                "end_ts": float((i + 1) * 10.0),
                "center_ts": float(i * 10.0 + 5.0),
                "theta_power_mean": float(abs(np.random.randn())),
                "alpha_power_mean": float(abs(np.random.randn())),
                "theta_alpha_ratio": float(np.random.rand()),
                "spec_entropy_mean": float(np.random.rand()),
                "one_over_f_slope": float(np.random.randn()),
            }
        )
    return pd.DataFrame(rows)


def test_validate_parquet_succeeds(tmp_path: Path):
    df = _make_epoch_df()
    p = tmp_path / "epoch_features.parquet"
    df.to_parquet(p)
    schema = Path(__file__).resolve().parents[1] / "schemas" / "epoch_features.schema.json"
    # Validate (will use GE if available else fallback)
    res = validate_parquet(p, schema_path=str(schema))
    assert res["success"] is True
    assert res["engine"] in ("pandas_fallback", "great_expectations")


def test_validate_parquet_fails_on_missing_required(tmp_path: Path):
    df = _make_epoch_df()
    # drop a required column
    df = df.drop(columns=["session_id"])
    p = tmp_path / "epoch_features_bad.parquet"
    df.to_parquet(p)
    schema = Path(__file__).resolve().parents[1] / "schemas" / "epoch_features.schema.json"
    try:
        validate_parquet(p, schema_path=str(schema))
        assert False, "Expected validation to raise RuntimeError for missing required column"
    except RuntimeError:
        # expected
        pass
