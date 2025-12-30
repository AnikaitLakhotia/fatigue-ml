# tests/test_mlflow_integration.py
"""
Lightweight MLflow integration test.

This test is skipped if mlflow, torch, or pytorch_lightning are not installed.
It runs a tiny one-epoch training in fast mode and verifies the mlruns
directory was created at the provided MLflow tracking URI.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pytest

# Skip if MLflow or training libs are not available
pytest.importorskip("mlflow")
pytest.importorskip("torch")
pytest.importorskip("pytorch_lightning")

from src.eeg.scripts.train_ssl_tf import main as train_main


def _write_tiny_npy_files(tmp_path: Path, n_files: int = 2, n_ch: int = 3, n_t: int = 64):
    paths = []
    for i in range(n_files):
        arr = np.random.randn(n_ch, n_t).astype(np.float32)
        p = tmp_path / f"sample_{i}.npy"
        np.save(str(p), arr)
        paths.append(str(p))
    return paths


def test_mlflow_tracking_creates_mlruns(tmp_path: Path):
    # prepare data
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    files = _write_tiny_npy_files(data_dir)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # Use file-based MLflow tracking to tmp path
    mlruns_dir = tmp_path / "mlruns"
    tracking_uri = f"file://{mlruns_dir}"

    argv = [
        "--data",
        *files,
        "--out_dir",
        str(out_dir),
        "--epochs",
        "1",
        "--batch_size",
        "2",
        "--num_workers",
        "0",
        "--gpus",
        "0",
        "--precision",
        "32",
        "--fast",  # ensure fast_dev_run behavior
        "--mlflow",
        "--mlflow_experiment",
        "test_experiment",
        "--mlflow_tracking_uri",
        tracking_uri,
    ]

    # run training; should complete and create mlruns under mlruns_dir
    train_main(argv)

    # assert mlruns folder was created and contains at least one experiment/run directory
    assert mlruns_dir.exists(), f"Expected mlruns dir at {mlruns_dir}"
    # There should be at least one subdirectory (experiment id)
    entries = list(mlruns_dir.glob("*"))
    assert len(entries) > 0, f"No experiment folders found inside {mlruns_dir}"