"""
Unit and smoke tests for preprocessing functions in src.eeg.data.io.

This test file verifies that session CSVs can be loaded into memory
correctly, produce an MNE RawArray, and metadata is returned. 

Author: Anikait Lakhotia
"""

from pathlib import Path
import pytest

from src.eeg.data.io import load_session_csv_to_raw, split_combined_csv_by_session


def test_load_session_csv_to_raw_smoke():
    """
    Smoke test for `load_session_csv_to_raw`.
    Verifies that a session CSV can be loaded, and returns a RawArray and metadata dictionary.
    """
    session_csv = Path("data/raw/combined_dataset.csv")

    raw, meta = load_session_csv_to_raw(session_csv)

    # Smoke checks
    assert raw is not None, "RawArray should not be None"
    assert meta is not None, "Meta dict should not be None"

    # Check expected channels
    expected_channels = ["CP3", "C3", "F5", "PO3", "PO4", "F6", "C4", "CP4"]
    for ch in expected_channels:
        assert ch in raw.ch_names, f"Missing expected channel: {ch}"

    # Check metadata
    assert meta["n_samples"] > 0, "Number of samples should be positive"
    assert abs(meta["sfreq"] - 256.0) < 1.0, "Sampling frequency mismatch"


def test_split_combined_csv_by_session_smoke(tmp_path):
    """
    Smoke test for `split_combined_csv_by_session`.
    Verifies that the combined CSV can be split into per-session CSV files.
    """
    combined_csv = Path("data/raw/combined_dataset.csv")
    out_dir = tmp_path / "sessions"

    session_files = split_combined_csv_by_session(combined_csv, out_dir)

    assert session_files, "No session files were created"
    for f in session_files:
        assert f.exists(), f"Session file does not exist: {f}"
        df = f.read_text()
        assert df, f"Session file is empty: {f}"


@pytest.mark.parametrize(
    "missing_column",
    ["session_id", "timestamp", "CP3"]
)
def test_load_session_csv_missing_column_raises(missing_column, tmp_path):
    """
    Parametrized test: ensures function raises ValueError if required column is missing.
    """
    session_csv = tmp_path / "broken.csv"
    # Create a CSV without the missing column
    cols = ["CP3", "C3", "F5", "PO3", "PO4", "F6", "C4", "CP4", "timestamp", "session_id"]
    cols.remove(missing_column)
    df_content = ",".join(cols) + "\n"
    session_csv.write_text(df_content)

    from src.eeg.data.io import load_session_csv_to_raw
    with pytest.raises(ValueError):
        load_session_csv_to_raw(session_csv)
