"""Time-frequency patch extraction and vector-quantization helpers.

Exports:
  - extract_patches_from_spectrogram(S, freq_bins, time_bins, stride_freq, stride_time)
  - sample_patches_from_epochs(list_of_S, n_patches, ...)
  - build_vq_codebook(patches, n_clusters)
  - encode_patches_to_tokens(patches, kmeans)
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
from sklearn.cluster import KMeans


def extract_patches_from_spectrogram(
    S: np.ndarray,
    freq_bins: int,
    time_bins: int,
    stride_freq: int = 1,
    stride_time: int = 1,
) -> np.ndarray:
    """
    Extract patches from spectrogram S (n_channels, n_freqs, n_times).

    Returns:
        patches: (n_patches, patch_height * patch_width * n_channels)
    """
    if S.ndim != 3:
        raise ValueError("S must be shape (n_channels, n_freqs, n_times)")
    n_ch, n_freq, n_time = S.shape
    patches = []
    for f0 in range(0, n_freq - freq_bins + 1, stride_freq):
        for t0 in range(0, n_time - time_bins + 1, stride_time):
            patch = S[:, f0 : f0 + freq_bins, t0 : t0 + time_bins]
            patches.append(patch.ravel())
    if not patches:
        return np.zeros((0, n_ch * freq_bins * time_bins))
    return np.stack(patches, axis=0)


def sample_patches_from_epochs(
    S_list: List[np.ndarray],
    n_patches: int = 100,
    freq_bins: int = 8,
    time_bins: int = 8,
    stride_freq: int = 4,
    stride_time: int = 4,
    seed: int | None = None,
) -> np.ndarray:
    """
    Randomly sample patches across multiple spectrograms.

    Returns:
        sampled_patches: (k, patch_dim)
    """
    rng = np.random.default_rng(seed)
    all_patches = []
    for S in S_list:
        p = extract_patches_from_spectrogram(
            S, freq_bins, time_bins, stride_freq, stride_time
        )
        if p.shape[0] > 0:
            all_patches.append(p)
    if not all_patches:
        return np.zeros(
            (0, freq_bins * time_bins * (S_list[0].shape[0] if S_list else 1))
        )
    all_p = np.vstack(all_patches)
    if all_p.shape[0] <= n_patches:
        return all_p
    idx = rng.choice(all_p.shape[0], size=n_patches, replace=False)
    return all_p[idx]


def build_vq_codebook(
    patches: np.ndarray, n_clusters: int = 256, random_state: int | None = None
) -> KMeans:
    """
    Fit a KMeans codebook on patch vectors.

    Returns:
        fitted sklearn.cluster.KMeans
    """
    if patches.ndim != 2:
        raise ValueError("patches must be 2D array (n_patches, patch_dim)")
    k = KMeans(n_clusters=n_clusters, random_state=random_state)
    k.fit(patches)
    return k


def encode_patches_to_tokens(patches: np.ndarray, kmeans: KMeans) -> np.ndarray:
    """
    Encode patches to integer tokens using fitted KMeans.

    Returns:
        tokens: (n_patches,) integer labels
    """
    return kmeans.predict(patches)
