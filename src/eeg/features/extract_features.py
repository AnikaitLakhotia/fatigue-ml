"""
Feature extraction utilities for fatigue-ml.

Provides an API for turning preprocessed windows/epochs into
feature vectors suitable for ML or unsupervised analysis.

Primary entrypoints:
- extract_features_from_epochs(epochs, sfreq, ...)
- save_features_df(df, path)

Epochs expected shape: (n_epochs, n_channels, n_samples)

Dependencies: numpy, scipy, pandas, mne, sklearn (all in pyproject)
"""

from __future__ import annotations
from typing import Tuple, List, Iterable, Optional, Dict, Any
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.linear_model import HuberRegressor
import mne

from src.eeg.utils.logger import get_logger

logger = get_logger(__name__)

# canonical bands for fatigue analysis
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def _psd_welch(
    data: np.ndarray,
    sfreq: float,
    n_per_seg: Optional[int] = None,
    n_overlap: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute PSD for multichannel data via Welch.

    Args:
        data: (n_channels, n_samples)
        sfreq: sampling frequency
        n_per_seg: if None uses min(256, n_samples)
        n_overlap: if None uses n_per_seg//2

    Returns:
        psds: (n_channels, n_freqs), freqs: (n_freqs,)
    """
    n_channels, n_samples = data.shape
    if n_per_seg is None:
        n_per_seg = min(256, n_samples)
    if n_per_seg > n_samples:
        n_per_seg = n_samples
    if n_overlap is None:
        n_overlap = n_per_seg // 2

    # mne wrapper is convenient and returns (n_channels, n_freqs)
    psds, freqs = mne.time_frequency.psd_array_welch(
    data,
    sfreq=sfreq,
    n_fft=n_per_seg,      
    n_per_seg=n_per_seg,
    n_overlap=n_overlap,
    fmin=0.0,
    fmax=np.inf,
    verbose=False,
    )

    # psds are in V^2/Hz (if input in microvolts or raw units)
    return psds, freqs


def bandpower_from_psd(psd: np.ndarray, freqs: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
    """Compute band power per channel by integrating PSD over band."""
    low, high = band
    idx = np.logical_and(freqs >= low, freqs <= high)
    bp = psd[:, idx].sum(axis=1)
    return bp


def relative_bandpower(psd: np.ndarray, freqs: np.ndarray, band: Tuple[float, float], total_band: Tuple[float, float] = (1.0, 45.0)) -> np.ndarray:
    """Return relative band power per channel."""
    band_p = bandpower_from_psd(psd, freqs, band)
    total_p = bandpower_from_psd(psd, freqs, total_band)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = band_p / (total_p + 1e-12)
    return rel


def peak_alpha_frequency(psd: np.ndarray, freqs: np.ndarray, band: Tuple[float, float] = (8.0, 12.0)) -> np.ndarray:
    """Return peak frequency within alpha band per channel."""
    low, high = band
    idx = np.logical_and(freqs >= low, freqs <= high)
    band_freqs = freqs[idx]
    band_psd = psd[:, idx]
    # find argmax along freq axis
    argmax = np.argmax(band_psd, axis=1)
    pk = band_freqs[argmax]
    return pk


def spectral_entropy_from_psd(psd: np.ndarray, freqs: np.ndarray, band: Tuple[float, float] = (1.0, 45.0)) -> np.ndarray:
    """Compute spectral entropy for each channel over given band (Shannon entropy normalized)."""
    low, high = band
    idx = np.logical_and(freqs >= low, freqs <= high)
    psd_band = psd[:, idx]
    # normalize to probability
    psd_sum = psd_band.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = psd_band / (psd_sum + 1e-12)
    # Shannon entropy
    ent = -np.nansum(p * np.log2(p + 1e-12), axis=1)
    # normalize by log2(n_bins)
    max_ent = np.log2(p.shape[1]) if p.shape[1] > 0 else 1.0
    return ent / (max_ent + 1e-12)


def hjorth_parameters(epoch: np.ndarray) -> Tuple[float, float, float]:
    """Compute Hjorth activity, mobility, complexity for a 1D signal (time series)."""
    x = epoch.astype(np.float64)
    activity = np.var(x)
    dx = np.diff(x)
    mobility = np.sqrt(np.var(dx) / (activity + 1e-12))
    ddx = np.diff(dx)
    with np.errstate(divide="ignore", invalid="ignore"):
        complexity = np.sqrt(np.var(ddx) / (np.var(dx) + 1e-12)) / (mobility + 1e-12)
    return float(activity), float(mobility), float(complexity)


def sample_entropy(signal_ts: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Compute sample entropy (SampEn) for 1D time series.

    Args:
        signal_ts: 1D array
        m: embedding dimension
        r: tolerance (as fraction of std)

    Returns:
        sampen (float)
    """
    x = np.asarray(signal_ts, dtype=float)
    n = len(x)
    if n <= m + 1:
        return 0.0
    sd = np.std(x, ddof=0)
    if sd == 0:
        return 0.0
    r *= sd
    def _phi(m_):
        count = 0
        for i in range(n - m_):
            xi = x[i : i + m_]
            for j in range(i + 1, n - m_ + 1):
                xj = x[j : j + m_]
                if np.max(np.abs(xi - xj)) <= r:
                    count += 1
        return count
    B = _phi(m)
    A = _phi(m + 1)
    if B == 0:
        return 0.0
    return -np.log((A + 1e-12) / (B + 1e-12))


def slope_1f(freqs: np.ndarray, psd: np.ndarray, fit_range: Tuple[float, float] = (2.0, 40.0)) -> np.ndarray:
    """Estimate 1/f slope per channel using robust linear fit on log-log PSD.

    Returns slope (negative) per channel.
    """
    low, high = fit_range
    idx = np.logical_and(freqs >= low, freqs <= high)
    xs = np.log10(freqs[idx] + 1e-12)
    results = []
    for row in psd:
        ys = np.log10(row[idx] + 1e-12)
        # robust linear fit
        model = HuberRegressor().fit(xs.reshape(-1, 1), ys)
        slope = float(model.coef_[0])
        results.append(slope)
    return np.array(results)


def mean_coherence_per_band(
    data: np.ndarray, sfreq: float, bands: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, float]:
    """Compute average pairwise coherence across channels per band.

    data: (n_channels, n_samples)
    Returns dict band -> mean coherence (scalar)
    """
    if bands is None:
        bands = BANDS
    n_chan = data.shape[0]
    band_means: Dict[str, float] = {}
    # compute pairwise coherence for each pair and average
    for band_name, band_range in bands.items():
        low, high = band_range
        cohs = []
        for i in range(n_chan):
            for j in range(i + 1, n_chan):
                f, Cxy = signal.coherence(data[i], data[j], fs=sfreq, nperseg=min(256, data.shape[1]))
                idx = np.logical_and(f >= low, f <= high)
                if np.any(idx):
                    cohs.append(np.mean(Cxy[idx]))
        band_means[band_name] = float(np.nanmean(cohs)) if cohs else 0.0
    return band_means


def _features_from_single_epoch(
    epoch: np.ndarray, sfreq: float, per_channel: bool = False
) -> Dict[str, Any]:
    """Compute a rich set of features for a single epoch array (n_channels, n_samples)."""
    # PSD
    psd, freqs = _psd_welch(epoch, sfreq)
    # compute band powers per channel
    band_pows = {name: bandpower_from_psd(psd, freqs, rng) for name, rng in BANDS.items()}
    total_power = bandpower_from_psd(psd, freqs, (1.0, 45.0))
    rel_pows = {f"rel_{name}": relative_bandpower(psd, freqs, rng) for name, rng in BANDS.items()}

    features: Dict[str, Any] = {}
    # per-channel features (flattened) if requested
    if per_channel:
        for ch_idx, ch_name in enumerate(range(epoch.shape[0])):  # numeric indices as suffix
            for bname in BANDS.keys():
                features[f"ch{ch_idx}_{bname}_power"] = float(band_pows[bname][ch_idx])
                features[f"ch{ch_idx}_rel_{bname}"] = float(rel_pows[f"rel_{bname}"][ch_idx])
            # Hjorth
            activity, mobility, complexity = hjorth_parameters(epoch[ch_idx])
            features[f"ch{ch_idx}_hjorth_activity"] = activity
            features[f"ch{ch_idx}_hjorth_mobility"] = mobility
            features[f"ch{ch_idx}_hjorth_complexity"] = complexity
            # sample entropy
            features[f"ch{ch_idx}_sampen"] = float(sample_entropy(epoch[ch_idx]))
            # peak alpha
            features[f"ch{ch_idx}_paf"] = float(peak_alpha_frequency(psd, freqs)[ch_idx])
    # aggregate features across channels (mean/std)
    for bname in BANDS.keys():
        arr = band_pows[bname]
        features[f"{bname}_power_mean"] = float(np.mean(arr))
        features[f"{bname}_power_std"] = float(np.std(arr))
        rel_arr = rel_pows[f"rel_{bname}"]
        features[f"rel_{bname}_mean"] = float(np.mean(rel_arr))
        features[f"rel_{bname}_std"] = float(np.std(rel_arr))

    # ratios known to be useful
    theta = np.mean(band_pows["theta"])
    alpha = np.mean(band_pows["alpha"])
    beta = np.mean(band_pows["beta"]) if "beta" in band_pows else 1e-12
    features["theta_alpha_ratio"] = float(theta / (alpha + 1e-12))
    features["theta_beta_ratio"] = float(theta / (beta + 1e-12))
    features["(theta+alpha)/beta"] = float((theta + alpha) / (beta + 1e-12))

    # spectral entropy across channels (mean/std)
    spec_ent = spectral_entropy_from_psd(psd, freqs)
    features["spec_entropy_mean"] = float(np.mean(spec_ent))
    features["spec_entropy_std"] = float(np.std(spec_ent))

    # 1/f slope (mean across channels)
    slopes = slope_1f(freqs, psd)
    features["slope_1f_mean"] = float(np.mean(slopes))
    features["slope_1f_std"] = float(np.std(slopes))

    # coherence features
    coh = mean_coherence_per_band(epoch, sfreq)
    for k, v in coh.items():
        features[f"coh_{k}"] = float(v)

    return features


def extract_features_from_epochs(
    epochs: np.ndarray,
    sfreq: float,
    per_channel: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """Extract features for each epoch.

    Args:
        epochs: (n_epochs, n_channels, n_samples)
        sfreq: sampling frequency
        per_channel: if True, include flattened per-channel features (increases dimensionality)
        verbose: log progress

    Returns:
        pd.DataFrame: one row per epoch, columns = features
    """
    n_epochs = epochs.shape[0]
    rows = []
    for i in range(n_epochs):
        if verbose and i % 50 == 0:
            logger.info("Extracting features: epoch %d / %d", i + 1, n_epochs)
        feat = _features_from_single_epoch(epochs[i], sfreq, per_channel=per_channel)
        feat["epoch_index"] = i
        rows.append(feat)
    df = pd.DataFrame(rows)
    # sensible ordering: epoch_index first
    cols = ["epoch_index"] + [c for c in df.columns if c != "epoch_index"]
    return df[cols]


def save_features_df(df: pd.DataFrame, path: str) -> None:
    """Save features dataframe to parquet (fast, compressed)."""
    p = pd.Path(path) if hasattr(pd, "Path") else None
    # use pandas to_parquet
    df.to_parquet(path, index=False)
    logger.info("Saved features -> %s", path)
