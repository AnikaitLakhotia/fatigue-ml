# src/eeg/models/__init__.py
"""
Modeling utilities package.

This package contains utilities for:
 - creating session-level embeddings from per-epoch feature tables,
 - running unsupervised modeling pipelines (PCA / UMAP / clustering),
 - evaluation helpers for unsupervised outputs,
 - a simple autoencoder baseline (PyTorch) for representation learning.

All modules are intentionally small, testable and CLI-friendly.
"""
from .embeddings import make_session_embeddings  # noqa: F401
from .unsupervised import run_unsupervised_pipeline  # noqa: F401
from .eval import (
    cluster_silhouette_score,
    cluster_temporal_contiguity,
    cluster_stability_bootstrap,
    correlate_with_proxy,
)  # noqa: F401
from .autoencoder import (
    Autoencoder,
    TrainAutoencoderConfig,
    train_autoencoder,
)  # noqa: F401
