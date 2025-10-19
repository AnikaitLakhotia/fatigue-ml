# tests/test_modeling_pipeline.py
"""
Fast smoke test for modeling pipeline:
 - creates minimal synthetic per-epoch feature parquet
 - runs session embedding creation
 - runs unsupervised pipeline (PCA + KMeans)
 - verifies outputs exist and labels assigned
"""

import numpy as np
import pandas as pd
from pathlib import Path
from src.eeg.models.embeddings import make_session_embeddings
from src.eeg.models.unsupervised import run_unsupervised_pipeline


def _make_synthetic_epoch_features(n_epochs=6, n_channels=4):
    # create small synthetic DF with band powers and a proxy
    rows = []
    ch_names = [f"ch{i}" for i in range(n_channels)]
    for e in range(n_epochs):
        row = {}
        # some band columns (delta,...)
        for band in ["delta", "theta", "alpha", "beta"]:
            for ch in ch_names:
                row[f"{band}_{ch}"] = float(
                    np.abs(np.random.randn())
                    + (1.0 if band == "theta" and e > n_epochs // 2 else 0.1)
                )
        row["session_id"] = "synthetic_sess"
        row["sfreq"] = 128
        rows.append(row)
    return pd.DataFrame(rows)


def test_modeling_smoke(tmp_path: Path):
    feat_dir = tmp_path / "features"
    feat_dir.mkdir()
    p = feat_dir / "synthetic_features.parquet"
    df = _make_synthetic_epoch_features()
    df.to_parquet(p)

    # embeddings
    out_emb = tmp_path / "embeddings.parquet"
    emb_df = make_session_embeddings(feat_dir, out_emb)
    assert out_emb.exists()
    assert not emb_df.empty

    # run unsupervised (use kmeans to avoid hdbscan dependency)
    out_dir = tmp_path / "model_out"
    res = run_unsupervised_pipeline(
        out_emb,
        out_dir,
        pca_components=2,
        umap_components=None,
        cluster_method="kmeans",
        cluster_kwargs={"n_clusters": 2},
    )
    assert "labels" in res
    assert len(res["labels"]) == len(emb_df)
