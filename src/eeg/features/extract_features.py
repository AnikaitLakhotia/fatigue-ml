"""
Unified feature extraction module with backward-compatible helpers.

Provides:
  - extract_all_features(epochs, sfreq, enabled=None, per_channel=False) -> pd.DataFrame
  - extract_features_from_epochs(...) compatibility wrapper
  - _psd_welch(...) test helper
  - FEATURE_REGISTRY mapping of registered features
"""

from __future__ import annotations

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

FEATURE_REGISTRY: Dict[str, Callable[[np.ndarray, float, bool], Dict[str, Any]]] = {}


def register_feature(name: str):
    """Decorator to register a feature function under `name`."""
    def decorator(func: Callable[[np.ndarray, float, bool], Dict[str, Any]]):
        FEATURE_REGISTRY[name] = func
        return func
    return decorator


@register_feature("psd_bandpowers")
def feat_psd_bandpowers(epoch: np.ndarray, sfreq: float, per_channel: bool = False) -> Dict[str, Any]:
    """PSD-based band powers, ratios, PAF and spectral entropy."""
    features: Dict[str, Any] = {}
    try:
        psd, freqs = compute_psd_welch(epoch, sfreq)
        bp = bandpowers(psd, freqs)
        for band, vals in bp.items():
            arr = np.asarray(vals)
            features[f"{band}_power_mean"] = float(np.mean(arr))
            features[f"{band}_power_std"] = float(np.std(arr))
        theta = float(np.mean(bp.get("theta", np.zeros(epoch.shape[0]))))
        alpha = float(np.mean(bp.get("alpha", np.ones(epoch.shape[0])))) + 1e-12
        beta = float(np.mean(bp.get("beta", np.ones(epoch.shape[0])))) + 1e-12
        features["theta_alpha_ratio"] = theta / alpha
        features["theta_beta_ratio"] = theta / beta
        features["(theta+alpha)/beta"] = (theta + alpha) / beta
        if per_channel:
            for ch in range(epoch.shape[0]):
                for band in ("delta", "theta", "alpha", "beta", "gamma"):
                    val = float(bp.get(band, np.zeros(epoch.shape[0]))[ch]) if band in bp else 0.0
                    features[f"ch{ch}_{band}_power"] = val
        idx_alpha = (freqs >= 8.0) & (freqs <= 12.0)
        if idx_alpha.any():
            pk_idx = np.argmax(psd[:, idx_alpha], axis=1)
            pk_freqs = freqs[idx_alpha][pk_idx]
            features["paf_mean"] = float(np.mean(pk_freqs))
        else:
            features["paf_mean"] = 0.0
        try:
            features["spec_entropy_mean"] = float(np.mean(spectral_entropy(psd)))
        except Exception:
            features["spec_entropy_mean"] = 0.0
            logger.debug("spectral_entropy failed", exc_info=True)
    except Exception:
        logger.exception("feat_psd_bandpowers failed")
    return features


@register_feature("coherence")
def feat_coherence(epoch: np.ndarray, sfreq: float, per_channel: bool = False) -> Dict[str, Any]:
    """Mean pairwise coherence per band aggregated across channel pairs."""
    try:
        coh = mean_pairwise_coherence(epoch, sfreq)
        return {f"coh_{k}": float(v) for k, v in coh.items()}
    except Exception:
        logger.exception("feat_coherence failed")
        return {f"coh_{k}": 0.0 for k in ("delta", "theta", "alpha", "beta", "gamma")}


@register_feature("timefreq_summary")
def feat_timefreq(epoch: np.ndarray, sfreq: float, per_channel: bool = False) -> Dict[str, Any]:
    """STFT-derived band-power summary aggregated across channels/time."""
    try:
        S, freqs, times = stft_spectrogram(epoch, sfreq)
        bmean = band_mean_from_spectrogram(S, freqs)
        return {f"tf_{k}_mean": float(v) for k, v in bmean.items()}
    except Exception:
        logger.exception("feat_timefreq failed")
        return {f"tf_{k}_mean": 0.0 for k in ("delta", "theta", "alpha", "beta", "gamma")}


@register_feature("nonlinear")
def feat_nonlinear(epoch: np.ndarray, sfreq: float, per_channel: bool = False) -> Dict[str, Any]:
    """Permutation entropy, Higuchi FD and sample entropy aggregated across channels."""
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
) -> pd.DataFrame:
    """Run registered features on all epochs and return tidy DataFrame (one row per epoch)."""
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
    if "epoch_index" in df.columns:
        cols = ["epoch_index"] + [c for c in df.columns if c != "epoch_index"]
        df = df.loc[:, cols]
    return df


# Backwards-compatible helper expected by some tests/scripts
def extract_features_from_epochs(
    epochs: np.ndarray, sfreq: float, enabled: Optional[List[str]] = None, per_channel: bool = False
) -> pd.DataFrame:
    """Thin compatibility wrapper around extract_all_features."""
    return extract_all_features(epochs=epochs, sfreq=sfreq, enabled=enabled, per_channel=per_channel)


def _psd_welch(data: np.ndarray, sfreq: float, n_per_seg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Expose compute_psd_welch for tests (returns (psd, freqs))."""
    return compute_psd_welch(data, sfreq, n_per_seg)


__all__ = ["extract_all_features", "extract_features_from_epochs", "_psd_welch", "FEATURE_REGISTRY"]
