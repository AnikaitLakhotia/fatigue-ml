# tests/test_train_ssl_integration.py
"""
Integration test for src.eeg.scripts.train_ssl wrapper.

This test will be skipped if torch or pytorch_lightning are not installed.
It creates tiny synthetic EEG files (.npy), runs the wrapper with
epochs=1 and num_workers=0 and asserts that the out_dir is created.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("pytorch_lightning")

from src.eeg.scripts import train_ssl  


def _write_tiny_npy_files(tmp_path: Path, n_files: int = 2, n_ch: int = 3, n_t: int = 64):
    paths = []
    for i in range(n_files):
        arr = np.random.randn(n_ch, n_t).astype(np.float32)
        p = tmp_path / f"sample_{i}.npy"
        np.save(str(p), arr)
        paths.append(str(p))
    return paths


def test_train_ssl_runs_quick(tmp_path: Path):
    # Enable fast mode (small run, avoids full training)
    os.environ["PYTEST_FAST"] = "1"

    out_dir = tmp_path / "out_run"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    files = _write_tiny_npy_files(data_dir)

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
    ]

    train_ssl.main(argv)

    assert out_dir.exists()
    # trainer writes tb logs into out_dir by default (train_ssl_tf uses TensorBoardLogger)
    assert any(out_dir.iterdir()), "Expected training artifacts in out_dir"
