"""Entropy and complexity features."""

from __future__ import annotations
import numpy as np


def spectral_entropy(psd: np.ndarray) -> np.ndarray:
    """
    Compute normalized spectral entropy per channel.

    Args:
        psd: (n_channels, n_freqs)

    Returns:
        1D array (n_channels,) with normalized entropy in [0,1].
    """
    p = psd / (psd.sum(axis=1, keepdims=True) + 1e-12)
    ent = -np.nansum(p * np.log2(p + 1e-12), axis=1)
    max_ent = np.log2(p.shape[1]) if p.shape[1] > 0 else 1.0
    return ent / (max_ent + 1e-12)


def sample_entropy(x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    A simple Sample Entropy (SampEn) implementation for short windows.

    Args:
        x: 1D signal
        m: embedding dimension
        r: tolerance fraction of sd

    Returns:
        SampEn scalar
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n <= m + 1:
        return 0.0
    sd = x.std(ddof=0)
    if sd == 0:
        return 0.0
    r_val = r * sd

    def _count(m_):
        count = 0
        for i in range(n - m_):
            xi = x[i : i + m_]
            for j in range(i + 1, n - m_ + 1):
                xj = x[j : j + m_]
                if np.max(np.abs(xi - xj)) <= r_val:
                    count += 1
        return count

    B = _count(m)
    A = _count(m + 1)
    if B == 0:
        return 0.0
    return -np.log((A + 1e-12) / (B + 1e-12))
