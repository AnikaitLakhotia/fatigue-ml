# src/eeg/models/unsupervised.py
from __future__ import annotations

"""
Unsupervised modeling pipeline (robust version).

This module implements an end-to-end unsupervised feature modeling pipeline that:
 - loads precomputed features (session or epoch level),
 - standardizes numeric features,
 - optionally applies PCA and/or UMAP for dimensionality reduction,
 - runs clustering (HDBSCAN / GMM / KMeans),
 - saves embeddings, labels, and fitted models.

Now includes defensive handling for small or degenerate datasets:
 - Automatically reduces PCA components if n_samples < n_components.
 - Skips PCA/UMAP/clustering gracefully if data is too small.
 - Assigns trivial cluster labels for single-sample inputs.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import joblib

from src.eeg.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import umap  # type: ignore
except Exception:
    umap = None

try:
    import hdbscan  # type: ignore
except Exception:
    hdbscan = None


def _numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return numeric-only dataframe (drop non-numeric metadata)."""
    return df.select_dtypes(include="number")


def _fit_cluster_algo(X: np.ndarray, method: str = "hdbscan", **kwargs) -> Tuple[np.ndarray, Any]:
    """
    Fit clustering algorithm and return (labels, fitted_model).

    Args:
        X: numeric data matrix
        method: 'hdbscan' | 'gmm' | 'kmeans'
        kwargs: optional params for the model
    """
    method = method.lower()
    if X.shape[0] <= 1:
        # trivial fallback
        logger.warning("Insufficient samples for clustering (%d); assigning label 0.", X.shape[0])
        return np.zeros(X.shape[0], dtype=int), None

    if method == "hdbscan":
        if hdbscan is None:
            raise RuntimeError("hdbscan not installed; please `pip install hdbscan`")
        model = hdbscan.HDBSCAN(**kwargs)
        labels = model.fit_predict(X)
    elif method == "gmm":
        model = GaussianMixture(**kwargs)
        labels = model.fit_predict(X)
    elif method == "kmeans":
        # cap clusters to available samples
        n_clusters = int(kwargs.get("n_clusters", 2))
        n_clusters = min(n_clusters, X.shape[0])
        model = KMeans(n_clusters=n_clusters, random_state=42, **{k: v for k, v in kwargs.items() if k != "n_clusters"})
        labels = model.fit_predict(X)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    return labels, model


def run_unsupervised_pipeline(
    features_path: str | Path,
    out_dir: str | Path,
    *,
    standardize: bool = True,
    pca_components: Optional[int] = 10,
    umap_components: Optional[int] = 2,
    cluster_method: str = "hdbscan",
    cluster_kwargs: Optional[Dict[str, Any]] = None,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Run an end-to-end unsupervised modeling pipeline.

    Args:
        features_path: path to parquet (session or epoch-level features)
        out_dir: directory to write outputs (embeddings, labels, models)
        standardize: whether to z-score numeric features
        pca_components: number of PCA components (auto-reduced if needed)
        umap_components: number of UMAP components (optional)
        cluster_method: clustering backend ('hdbscan'|'gmm'|'kmeans')
        cluster_kwargs: kwargs for clustering algorithm
        save_artifacts: whether to write parquet/npy/joblib outputs

    Returns:
        dict: results containing DataFrame, pipeline, labels, embeddings, model
    """
    features_path = Path(features_path)
    out_dir = Path(out_dir)
    cluster_kwargs = cluster_kwargs or {}

    if not features_path.exists():
        raise FileNotFoundError(features_path)

    df = pd.read_parquet(features_path)
    if df.empty:
        raise ValueError("Empty features input")

    numeric = _numeric_df(df)
    if numeric.shape[1] == 0:
        raise ValueError("No numeric columns available for modeling")

    X = numeric.to_numpy(dtype=float)
    n_samples, n_features = X.shape
    logger.info("Loaded %s with shape %s", features_path, X.shape)

    # Handle trivial cases early
    if n_samples == 0:
        logger.warning("No samples found; returning empty result.")
        return {"df": df, "labels": np.array([]), "pipeline": None, "model": None}
    if n_samples == 1:
        logger.warning("Single-sample dataset; assigning single label 0.")
        labels = np.zeros(1, dtype=int)
        emb_df = pd.DataFrame({"idx": [0], "label": [0]})
        if save_artifacts:
            out_dir.mkdir(parents=True, exist_ok=True)
            emb_df.to_parquet(out_dir / "embeddings.parquet")
            np.save(out_dir / "cluster_labels.npy", labels)
        return {"df": df, "labels": labels, "embeddings": emb_df, "pipeline": None, "model": None}

    # Standardization
    steps = []
    if standardize:
        scaler = StandardScaler()
        steps.append(("scaler", scaler))

    # PCA: adjust safely
    if pca_components and pca_components > 0:
        n_comp = min(int(pca_components), n_samples, n_features)
        if n_comp < pca_components:
            logger.info("Reducing PCA components from %d → %d (samples=%d, features=%d)",
                        pca_components, n_comp, n_samples, n_features)
        pca = PCA(n_components=n_comp, random_state=42)
        steps.append(("pca", pca))

    pipeline = Pipeline(steps) if steps else None
    X_trans = pipeline.fit_transform(X) if pipeline else X

    # UMAP embedding
    umap_emb = None
    if umap_components and umap_components > 0:
        if umap is None:
            logger.warning("UMAP not available; skipping.")
        elif n_samples <= 2:
            logger.warning("Too few samples (%d) for UMAP; skipping.", n_samples)
        else:
            reducer = umap.UMAP(n_components=int(umap_components), random_state=42)
            umap_emb = reducer.fit_transform(X_trans)
            logger.info("UMAP reduced to %d dims.", umap_components)

    # Clustering
    labels, model = _fit_cluster_algo(X_trans, method=cluster_method, **cluster_kwargs)

    # Embeddings DataFrame
    emb_df = pd.DataFrame({"idx": np.arange(n_samples), "label": labels})
    if pipeline and "pca" in pipeline.named_steps:
        for i in range(pipeline.named_steps["pca"].n_components_):
            emb_df[f"pc{i+1}"] = X_trans[:, i]
    if umap_emb is not None:
        for i in range(umap_emb.shape[1]):
            emb_df[f"umap{i+1}"] = umap_emb[:, i]

    # attach metadata (if any)
    for meta_col in ("session_id", "sfreq", "n_channels", "channel_names", "source_file"):
        if meta_col in df.columns:
            emb_df[meta_col] = df[meta_col].values

    # Save artifacts
    if save_artifacts:
        out_dir.mkdir(parents=True, exist_ok=True)
        emb_path = out_dir / "embeddings.parquet"
        labels_path = out_dir / "cluster_labels.npy"
        model_path = out_dir / "model_pipeline.joblib"

        emb_df.to_parquet(emb_path)
        np.save(labels_path, labels)
        joblib.dump({"pipeline": pipeline, "cluster_model": model}, model_path)

        logger.info("Saved embeddings → %s", emb_path)
        logger.info("Saved labels → %s", labels_path)
        logger.info("Saved model pipeline → %s", model_path)

    return {
        "df": df,
        "X": X,
        "pipeline": pipeline,
        "embeddings": emb_df,
        "labels": labels,
        "model": model,
    }
