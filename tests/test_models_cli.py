# tests/test_models_cli.py
"""
Smoke test for modeling CLI.
Creates tiny synthetic features and runs:
 - embeddings subcommand
 - unsupervised subcommand (KMeans)
This test avoids heavy optional dependencies (UMAP/HDBSCAN/Torch).
"""

from pathlib import Path
import subprocess
import sys
import pandas as pd
import numpy as np


def _make_small_epoch_parquet(p: Path, n_epochs: int = 8, n_channels: int = 4):
    rows = []
    ch_names = [f"ch{i}" for i in range(n_channels)]
    for e in range(n_epochs):
        for band in ["delta", "theta", "alpha"]:
            for ch in ch_names:
                rows.append(
                    {
                        f"{band}_{ch}": float(
                            abs(np.random.randn())
                            + (1.0 if band == "theta" and e > n_epochs // 2 else 0.2)
                        ),
                        "session_id": "s1",
                        "sfreq": 128,
                    }
                )
    # Create a pivot: each epoch row
    # Simpler: create n_epochs rows with same columns
    cols = {}
    for band in ["delta", "theta", "alpha"]:
        for ch in ch_names:
            cols[f"{band}_{ch}"] = np.random.rand(n_epochs) + (
                1.0 if band == "theta" else 0.1
            )
    df = pd.DataFrame(cols)
    df["session_id"] = "synthetic"
    df["sfreq"] = 128
    df.to_parquet(p)


def test_cli_smoke(tmp_path: Path):
    feat_dir = tmp_path / "features"
    feat_dir.mkdir()
    p = feat_dir / "synthetic_features.parquet"
    _make_small_epoch_parquet(p)

    emb_out = tmp_path / "session_embeddings.parquet"
    # run embeddings
    cmd = [
        sys.executable,
        "-m",
        "src.eeg.models.cli",
        "embeddings",
        "--in",
        str(feat_dir),
        "--out",
        str(emb_out),
    ]
    subprocess.run(cmd, check=True, timeout=20)

    assert emb_out.exists()

    # run unsupervised (kmeans)
    out_dir = tmp_path / "models"
    cmd2 = [
        sys.executable,
        "-m",
        "src.eeg.models.cli",
        "unsupervised",
        "--features",
        str(emb_out),
        "--out",
        str(out_dir),
        "--pca",
        "2",
        "--umap",
        "0",  # explicitly skip UMAP by passing 0
        "--cluster",
        "kmeans",
    ]
    subprocess.run(cmd2, check=True, timeout=20)

    # check artifacts
    assert (out_dir / "embeddings.parquet").exists()
    assert (out_dir / "cluster_labels.npy").exists()
