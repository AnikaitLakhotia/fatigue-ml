# src/eeg/models/eval.py
from __future__ import annotations

"""
Evaluation helpers for unsupervised clustering on EEG features.

Functions:
 - cluster_silhouette_score(X, labels)
 - cluster_temporal_contiguity(labels, epoch_indices)
 - cluster_stability_bootstrap(df, pipeline_runner, n_iter=10)
 - correlate_with_proxy(df, proxy_name)
"""

from typing import Sequence, Callable, Dict, Any
import numpy as np
import pandas as pd

from src.eeg.utils.logger import get_logger

logger = get_logger(__name__)


def cluster_silhouette_score(X: np.ndarray, labels: Sequence[int]) -> float:
    """
    Compute silhouette score for clustering (ignores noise label -1).

    Args:
        X: feature matrix (n_samples, n_features)
        labels: cluster labels (n_samples,)

    Returns:
        silhouette score (float). Returns -1.0 if not computable.
    """
    # Lazy import due to sklearn import cost on some platforms.
    try:
        from sklearn.metrics import silhouette_score  # type: ignore
    except Exception:
        logger.warning("sklearn.metrics.silhouette_score not available; returning -1.0")
        return -1.0

    if len(set(labels)) <= 1 or len(X) < 2:
        return -1.0
    # remove noise label -1 (HDBSCAN)
    mask = np.array(labels) != -1
    if mask.sum() < 2:
        return -1.0
    try:
        return float(silhouette_score(X[mask], np.array(labels)[mask]))
    except Exception:
        logger.exception("Error computing silhouette score")
        return -1.0


def cluster_temporal_contiguity(
    labels: Sequence[int], epoch_order: Sequence[int]
) -> float:
    """
    Compute a simple temporal contiguity score:
    Fraction of transitions that are within the same cluster.

    Args:
        labels: cluster label for each epoch, ordered by time.
        epoch_order: index/order (0..n-1) corresponding to time.

    Returns:
        contiguity fraction in [0,1]
    """
    if len(labels) < 2:
        return 1.0
    # assume epoch_order is same as order; compute fraction of consecutive pairs with same label
    n = len(labels)
    same = 0
    total = 0
    for i in range(1, n):
        if labels[i] == -1 or labels[i - 1] == -1:
            # ignore transitions involving noise
            continue
        total += 1
        if labels[i] == labels[i - 1]:
            same += 1
    return float(same / total) if total > 0 else 0.0


def cluster_stability_bootstrap(
    df: pd.DataFrame,
    pipeline_runner: Callable[[pd.DataFrame], Sequence[int]],
    n_iter: int = 10,
    sample_frac: float = 0.8,
):
    """
    Estimate cluster stability by bootstrap-resampling rows and comparing labels.

    Args:
        df: original feature dataframe (rows=epochs or sessions)
        pipeline_runner: function that accepts df and returns labels array
        n_iter: bootstrap iterations
        sample_frac: fraction of rows to sample each iteration

    Returns:
        dict with 'mean_ari' (mean adjusted rand index) and 'ari_list'
    """
    from sklearn.metrics import adjusted_rand_score  # lazy import

    labels_ref = pipeline_runner(df)
    ari_list = []
    n = len(df)
    for _ in range(n_iter):
        sample_idx = np.random.choice(n, size=int(n * sample_frac), replace=False)
        sub = df.iloc[sample_idx]
        labels_sub = pipeline_runner(sub)
        # need to align indices: map sub indices to positions in original ref
        # simple approach: compute ARI between labels of sub vs labels_ref at sample_idx
        ari = adjusted_rand_score(labels_ref[sample_idx], labels_sub)
        ari_list.append(float(ari))
    return {"ari_list": ari_list, "mean_ari": float(np.mean(ari_list))}


def correlate_with_proxy(df: pd.DataFrame, proxy_name: str) -> Dict[str, float]:
    """
    Correlate cluster labels or continuous embedding with a proxy feature (e.g. theta_alpha_ratio).

    Args:
        df: DataFrame containing columns 'label' and proxy_name
        proxy_name: string for proxy column

    Returns:
        dict with pearson_r and spearman_rho (floats). If not computable returns NaNs.
    """
    if "label" not in df.columns or proxy_name not in df.columns:
        logger.warning("Missing label or proxy column")
        return {"pearson_r": float("nan"), "spearman_r": float("nan")}
    try:
        # lazy import scipy.stats to avoid import-time costs
        from scipy.stats import pearsonr, spearmanr  # type: ignore

        # cluster-wise mean proxy values
        mask = df["label"] != -1
        if mask.sum() == 0:
            return {"pearson_r": float("nan"), "spearman_r": float("nan")}
        # correlate proxy with numeric label (not perfect but useful as quick check)
        labs = df.loc[mask, "label"].astype(float).values
        proxy = df.loc[mask, proxy_name].astype(float).values
        pr, _ = pearsonr(labs, proxy)
        sr, _ = spearmanr(labs, proxy)
        return {"pearson_r": float(pr), "spearman_r": float(sr)}
    except Exception:
        logger.exception("Error computing correlations")
        return {"pearson_r": float("nan"), "spearman_r": float("nan")}