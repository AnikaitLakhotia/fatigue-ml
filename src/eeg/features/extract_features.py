from __future__ import annotations

"""
Unified feature extraction registry and helpers.

Provides:
  - FEATURE_REGISTRY decorator-based registration
  - several built-in features (psd_bandpowers, coherence, timefreq_summary, nonlinear)
  - extract_all_features(...) -> pd.DataFrame (one row per epoch)
  - extract_features_from_epochs(...) compatibility wrapper
  - _psd_welch(...) test helper (thin wrapper over psd_features.compute_psd_welch)
"""

from typing import Callable, Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.eeg.utils.logger import get_logger
from src.eeg.features.psd_features import compute_psd_welch, bandpowers
from src.eeg.features.coherence_features import mean_pairwise_coherence
from src.eeg.features.entropy_features import spectral_entropy, sample_entropy
from src.eeg.features.timefreq_features import stft_spectrogram, band_mean_from_spectrogram
from src.eeg.features.nonlinear_features import permutation_entropy, higuchi_fd

logger = get_logger(__name__)

# Registry for feature extractor callables:
FEATURE_REGISTRY: Dict[str, Callable[[np.ndarray, float, bool], Dict[str, Any]]] = {}

# small EPS for numerical stability (also used in PSD module)
EPS = 1e-12


def register_feature(name: str):
    """
    Decorator to register a feature function under `name`.
    Registered functions must accept (epoch: np.ndarray, sfreq: float, per_channel: bool)
    and return a dict mapping feature name -> scalar.
    """
    def decorator(func: Callable[[np.ndarray, float, bool], Dict[str, Any]]):
        FEATURE_REGISTRY[name] = func
        return func
    return decorator


@register_feature("psd_bandpowers")
def feat_psd_bandpowers(epoch: np.ndarray, sfreq: float, per_channel: bool = False) -> Dict[str, Any]:
    """
    PSD-based band powers, ratios, PAF, spectral entropy and 1/f slope.

    Args:
        epoch: ndarray (n_channels, n_samples)
        sfreq: sampling frequency
        per_channel: if True, include flattened per-channel band power features

    Returns:
        Dict[name, scalar]
    """
    features: Dict[str, Any] = {}
    try:
        psd, freqs = compute_psd_welch(epoch, sfreq)
        bp = bandpowers(psd, freqs)
        # bandwise mean/std across channels
        for band, vals in bp.items():
            arr = np.asarray(vals)
            features[f"{band}_power_mean"] = float(np.mean(arr))
            features[f"{band}_power_std"] = float(np.std(arr))

        # common ratios (safe with EPS inside bandpowers)
        theta = float(np.mean(bp.get("theta", np.zeros(epoch.shape[0]))))
        alpha = float(np.mean(bp.get("alpha", np.ones(epoch.shape[0])))) + EPS
        beta = float(np.mean(bp.get("beta", np.ones(epoch.shape[0])))) + EPS
        features["theta_alpha_ratio"] = theta / alpha
        features["theta_beta_ratio"] = theta / beta
        features["(theta+alpha)/beta"] = (theta + alpha) / beta

        if per_channel:
            for ch in range(epoch.shape[0]):
                for band in ("delta", "theta", "alpha", "beta", "gamma"):
                    val = float(bp.get(band, np.zeros(epoch.shape[0]))[ch]) if band in bp else 0.0
                    features[f"ch{ch}_{band}_power"] = val

        # Peak alpha frequency (PAF) aggregated across channels
        idx_alpha = (freqs >= 8.0) & (freqs <= 12.0)
        if idx_alpha.any():
            try:
                pk_idx = np.argmax(psd[:, idx_alpha], axis=1)
                pk_freqs = freqs[idx_alpha][pk_idx]
                features["paf_mean"] = float(np.mean(pk_freqs))
            except Exception:
                features["paf_mean"] = 0.0
                logger.debug("PAF computation failed", exc_info=True)
        else:
            features["paf_mean"] = 0.0

        # spectral entropy (per-channel -> mean)
        try:
            features["spec_entropy_mean"] = float(np.mean(spectral_entropy(psd)))
        except Exception:
            features["spec_entropy_mean"] = 0.0
            logger.debug("spectral_entropy failed", exc_info=True)

        # 1/f slope estimation (robust log-log linear fit)
        # compute per-channel slope on log10(freq) vs log10(psd) in 1-45 Hz
        try:
            f_idx = (freqs >= 1.0) & (freqs <= 45.0)
            slopes = []
            if f_idx.any():
                xf = np.log10(freqs[f_idx])
                for ch in range(psd.shape[0]):
                    yf = np.log10(psd[ch, f_idx] + EPS)  # floor to EPS to avoid -inf
                    # require at least a few points with variance
                    if np.isfinite(yf).all() and np.ptp(yf) > 1e-6:
                        coef = np.polyfit(xf, yf, 1)
                        slopes.append(float(coef[0]))
                if slopes:
                    # take median as a robust estimate
                    features["one_over_f_slope"] = float(np.median(slopes))
                else:
                    features["one_over_f_slope"] = 0.0
            else:
                features["one_over_f_slope"] = 0.0
        except Exception:
            logger.exception("one_over_f_slope estimation failed")
            features["one_over_f_slope"] = 0.0

    except Exception:
        logger.exception("feat_psd_bandpowers failed")
    return features


@register_feature("coherence")
def feat_coherence(epoch: np.ndarray, sfreq: float, per_channel: bool = False) -> Dict[str, Any]:
    """
    Mean pairwise coherence per canonical band aggregated across channel pairs.
    """
    try:
        coh = mean_pairwise_coherence(epoch, sfreq)
        return {f"coh_{k}": float(v) for k, v in coh.items()}
    except Exception:
        logger.exception("feat_coherence failed")
        return {f"coh_{k}": 0.0 for k in ("delta", "theta", "alpha", "beta", "gamma")}


@register_feature("timefreq_summary")
def feat_timefreq(epoch: np.ndarray, sfreq: float, per_channel: bool = False) -> Dict[str, Any]:
    """
    STFT-derived band-power summary aggregated across channels/time.
    """
    try:
        S, freqs, times = stft_spectrogram(epoch, sfreq)
        bmean = band_mean_from_spectrogram(S, freqs)
        return {f"tf_{k}_mean": float(v) for k, v in bmean.items()}
    except Exception:
        logger.exception("feat_timefreq_failed")
        return {f"tf_{k}_mean": 0.0 for k in ("delta", "theta", "alpha", "beta", "gamma")}


@register_feature("nonlinear")
def feat_nonlinear(epoch: np.ndarray, sfreq: float, per_channel: bool = False) -> Dict[str, Any]:
    """
    Permutation entropy, Higuchi FD and sample entropy aggregated across channels.
    """
    try:
        perm_list, hfd_list, samp_list = [], [], []
        for ch in range(epoch.shape[0]):
            sig = epoch[ch]
            perm_list.append(permutation_entropy(sig))
            hfd_list.append(higuchi_fd(sig))
            samp_list.append(sample_entropy(sig))
        return {
            "perm_entropy_mean": float(np.mean(perm_list)),
            "higuchi_fd_mean": float(np.mean(hfd_list)),
            "sampen_mean": float(np.mean(samp_list)),
        }
    except Exception:
        logger.exception("feat_nonlinear failed")
        return {"perm_entropy_mean": 0.0, "higuchi_fd_mean": 0.0, "sampen_mean": 0.0}


def extract_all_features(
    epochs: np.ndarray,
    sfreq: float,
    enabled: Optional[List[str]] = None,
    per_channel: bool = False,
    meta: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Run all registered features on each epoch and return a tidy DataFrame.

    Args:
        epochs: ndarray (n_epochs, n_channels, n_samples)
        sfreq: sampling frequency (Hz)
        enabled: optional list of feature keys to run (defaults to all registry keys)
        per_channel: whether to include per-channel flattened features (if feature supports it)
        meta: optional metadata dict (session_id, channel_names, etc.) to attach to output

    Returns:
        pd.DataFrame with one row per epoch and feature columns
    """
    if enabled is None:
        enabled = list(FEATURE_REGISTRY.keys())
    rows: List[Dict[str, Any]] = []
    n_epochs = int(getattr(epochs, "shape", (0,))[0])
    for i in range(n_epochs):
        row: Dict[str, Any] = {"epoch_index": int(i)}
        epoch = epochs[i]
        for key in enabled:
            fn = FEATURE_REGISTRY.get(key)
            if fn is None:
                logger.warning("Unknown feature requested: %s", key)
                continue
            try:
                feats = fn(epoch, sfreq, per_channel=per_channel)
                for k, v in feats.items():
                    if isinstance(v, (np.generic,)):
                        v = v.item()
                    row[k] = v
            except Exception:
                logger.exception("Feature %s failed on epoch %d", key, i)
        rows.append(row)
    df = pd.DataFrame(rows)

    # Attach metadata if provided
    if meta:
        if "session_id" in meta and "session_id" not in df.columns:
            df["session_id"] = meta.get("session_id")
        if "channel_names" in meta and "channel_names" not in df.columns:
            df["channel_names"] = repr(meta.get("channel_names"))
        if "sfreq" in meta and "sfreq" not in df.columns:
            df["sfreq"] = float(meta.get("sfreq"))

    # Ensure epoch_index is first column if present
    if "epoch_index" in df.columns:
        cols = ["epoch_index"] + [c for c in df.columns if c != "epoch_index"]
        df = df.loc[:, cols]

    # Basic QC: count channels with total==0 if totals present
    total_cols = [c for c in df.columns if c.startswith("total_")]
    if total_cols:
        df["num_channels_with_zero_total"] = df[total_cols].apply(lambda r: int((r == 0).sum()), axis=1)
        df["has_any_zero_total"] = df["num_channels_with_zero_total"] > 0
    else:
        df["num_channels_with_zero_total"] = 0
        df["has_any_zero_total"] = False

    return df


def extract_features_from_epochs(
    epochs: np.ndarray, sfreq: float, enabled: Optional[List[str]] = None, per_channel: bool = False
) -> pd.DataFrame:
    """
    Backwards-compatible wrapper around extract_all_features.
    """
    return extract_all_features(epochs=epochs, sfreq=sfreq, enabled=enabled, per_channel=per_channel)


def _psd_welch(data: np.ndarray, sfreq: float, n_per_seg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Test helper: thin wrapper exposing compute_psd_welch for tests.
    """
    return compute_psd_welch(data, sfreq, n_per_seg)


__all__ = ["extract_all_features", "extract_features_from_epochs", "_psd_welch", "FEATURE_REGISTRY"]
