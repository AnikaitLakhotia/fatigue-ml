"""Minimal wrapper for source reconstruction (MNE-based).

This module provides convenience wrappers around MNE inverse operators. These
helpers assume the user supplies an MNE Forward/Inverse operator or BEM model.
They are intentionally small and guard for missing mne dependency.
"""

from __future__ import annotations
from typing import Any
import numpy as np

def compute_minimum_norm_source_time_courses(raw, inv, start=0, stop=None, return_stc=False):
    """
    Compute source time courses from a preloaded MNE Raw using an inverse operator.

    Args:
        raw: mne.io.BaseRaw (preprocessed)
        inv: mne.minimum_norm.inverse_operator (constructed externally)
        start, stop: sample indices or None (defaults read from raw)
        return_stc: if True, return MNE SourceEstimate (requires mne)

    Returns:
        If return_stc: mne.SourceEstimate
        Else: ndarray (n_sources, n_times)
    """
    try:
        import mne  # type: ignore
    except Exception as exc:
        raise RuntimeError("mne required for source reconstruction") from exc

    if stop is None:
        stop = raw.n_times
    # pick data
    data, _ = raw[:, start:stop]
    # compute evoked-like object if needed (mne expects evoked or epochs for apply_inverse)
    # Convert to 2D (n_channels, n_samples)
    method = "dSPM"
    stc = mne.minimum_norm.apply_inverse_raw(raw, inv, lambda2=1.0 / 9.0, method=method, verbose=False)
    if return_stc:
        return stc
    return stc.data  # (n_sources, n_times)
