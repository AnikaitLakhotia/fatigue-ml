"""Nonlinear complexity estimators (Permutation entropy, Higuchi FD).

Small, tested, and readable implementations suitable for epoch-level features.
"""

from __future__ import annotations
import numpy as np


def permutation_entropy(x: np.ndarray, m: int = 3, delay: int = 1) -> float:
    """Compute Bandt-Pompe permutation entropy for time series x."""
    n = len(x)
    if n < m * delay:
        return 0.0
    perms = {}
    for i in range(n - delay * (m - 1)):
        seq = x[i : i + delay * m : delay]
        ranks = tuple(np.argsort(np.argsort(seq)))
        perms[ranks] = perms.get(ranks, 0) + 1
    p = np.array(list(perms.values()), dtype=float)
    p /= p.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


def higuchi_fd(x: np.ndarray, k_max: int = 10) -> float:
    """Estimate Higuchi fractal dimension (simple and robust)."""
    n = len(x)
    if n < 10:
        return 0.0
    x = np.asarray(x, dtype=float)
    L = []
    for k in range(1, min(k_max, n//2) + 1):
        Lk = []
        for m in range(k):
            idxs = np.arange(m, n, k)
            if idxs.size < 2:
                continue
            Lm = (np.sum(np.abs(np.diff(x[idxs]))) * (n - 1) / (((n - m) // k) * k)) / k
            Lk.append(Lm)
        if Lk:
            L.append(np.mean(Lk))
    if len(L) < 2:
        return 0.0
    k_arr = np.arange(1, len(L) + 1)
    coef = np.polyfit(np.log(k_arr), np.log(L), 1)
    return float(-coef[0])
