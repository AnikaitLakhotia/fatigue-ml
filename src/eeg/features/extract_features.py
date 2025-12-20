# src/eeg/features/extract_features.py
"""
Feature extraction helpers.

Primary function: extract_features_from_epochs(...) which accepts epochs plus
per-epoch metadata (start_ts/end_ts/center_ts) and returns a pandas DataFrame
with per-channel features (band powers, ratios, PAF, spectral entropy,
one-over-f slope, nonlinear metrics) along with epoch timestamps and session info.

This file is backward-compatible with older callers that passed legacy kwargs
such as `enabled`, `save_spectrograms`, `backend`, etc. Unknown/unused kwargs
are accepted and ignored to maintain API stability.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Iterable, Any

import numpy as np
import pandas as pd
from scipy.signal import spectrogram, welch, coherence
from numpy.linalg import lstsq

# canonical bands
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 80.0),
}


def _psd_welch(x: np.ndarray, sfreq: float, nperseg: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Welch PSD (freqs, Pxx).
    """
    freqs, Pxx = welch(x, fs=sfreq, nperseg=nperseg or min(256, len(x)))
    return freqs, Pxx


def _spectrogram(x: np.ndarray, sfreq: float, nperseg: int | None = None, noverlap: int | None = None):
    """
    Short-time spectrogram: returns freqs, times, Sxx (power).
    Sxx shape: (n_freqs, n_times)
    """
    nperseg = int(nperseg or min(128, x.size))
    noverlap = int(noverlap if noverlap is not None else nperseg // 2)
    freqs, times, Sxx = spectrogram(x, fs=sfreq, nperseg=nperseg, noverlap=noverlap, scaling="density", mode="psd")
    return freqs, times, Sxx


def band_power_time_series_from_spectrogram(freqs: np.ndarray, Sxx: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
    """
    Given spectrogram arrays, compute power time-series for the given band
    by integrating across frequency bins for each time slice.
    Returns: 1D array length n_times.
    """
    idx = (freqs >= band[0]) & (freqs < band[1])
    if not idx.any():
        return np.zeros(Sxx.shape[1], dtype=float)
    # integrate power over frequency axis for each time column
    band_power_ts = np.trapz(Sxx[idx, :], freqs[idx], axis=0)
    return band_power_ts


def spectral_entropy_from_psd(Pxx: np.ndarray) -> float:
    """
    Shannon spectral entropy (bits).
    """
    p = Pxx / (np.sum(Pxx) + 1e-12)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def peak_alpha_freq_from_psd(freqs: np.ndarray, Pxx: np.ndarray, band=(8.0, 13.0)) -> float:
    idx = (freqs >= band[0]) & (freqs <= band[1])
    if not idx.any():
        return float("nan")
    subf = freqs[idx]
    subp = Pxx[idx]
    return float(subf[np.argmax(subp)])


def one_over_f_slope(freqs: np.ndarray, Pxx: np.ndarray, fit_range=(1.0, 40.0)) -> float:
    """
    Fit a line to log10(freq) vs log10(power) in fit_range and return slope (1/f exponent).
    """
    idx = (freqs >= fit_range[0]) & (freqs <= fit_range[1]) & (Pxx > 0)
    if idx.sum() < 2:
        return float("nan")
    x = np.log10(freqs[idx]).reshape(-1, 1)
    y = np.log10(Pxx[idx])
    A = np.hstack([x, np.ones_like(x)])
    m, c = lstsq(A, y, rcond=None)[0]
    return float(m)


# small/fast estimators for nonlinear metrics (placeholders; can be replaced)
def permutation_entropy(x: np.ndarray, order: int = 3) -> float:
    # quick heuristic: normalized sign-diff entropy on short sequence
    if len(x) < order + 1:
        return float("nan")
    diffs = np.sign(np.diff(x))
    p_pos = np.mean(diffs > 0)
    p_neg = np.mean(diffs < 0)
    ps = np.array([p_pos, p_neg])
    ps = ps[ps > 0]
    return float(-np.sum(ps * np.log2(ps))) if ps.size > 0 else 0.0


def higuchi_fd(x: np.ndarray, kmax: int = 10) -> float:
    # relatively fast approximate Higuchi FD
    N = len(x)
    if N < 10:
        return float("nan")
    L = []
    ks = np.arange(1, min(kmax, N // 2) + 1)
    for k in ks:
        Lk = 0.0
        for m in range(k):
            idxs = np.arange(m, N, k)
            if idxs.size <= 1:
                continue
            Lm = np.sum(np.abs(np.diff(x[idxs])))
            Lk += Lm * (N - 1) / (len(idxs) * k)
        L.append(Lk / k if k > 0 else 0.0)
    L = np.array(L)
    if np.any(L <= 0):
        return float("nan")
    slope = np.polyfit(np.log(ks), np.log(L), 1)[0]
    return float(slope)


def sample_entropy(x: np.ndarray) -> float:
    if len(x) < 20:
        return float("nan")
    return float(np.std(x) / (np.mean(np.abs(x)) + 1e-12))


def _bandwise_pairwise_coherence(epoch: np.ndarray, sfreq: float, bands: dict = BANDS, nperseg: int | None = 256) -> Dict[str, float]:
    """
    Compute average coherence per-band for the epoch (pairwise across channels).
    Returns mapping band_name -> average_coherence (scalar).
    """
    n_channels = epoch.shape[0]
    out = {}
    # precompute coherence for each pair
    for band_name, band_range in bands.items():
        vals = []
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                f, Cxy = coherence(epoch[i], epoch[j], fs=sfreq, nperseg=min(int(nperseg or 256), epoch.shape[1]))
                # average coherence over band_range
                idx = (f >= band_range[0]) & (f < band_range[1])
                if idx.any():
                    vals.append(float(np.mean(Cxy[idx])))
        out[band_name] = float(np.nanmean(vals)) if len(vals) > 0 else float("nan")
    return out


def extract_features_from_epochs(
    epochs: np.ndarray,
    epoch_meta: Optional[List[Dict]] = None,
    sfreq: Optional[float] = None,
    per_channel: bool = True,
    channel_names: Optional[List[str]] = None,
    *,
    # legacy / compatibility kwargs accepted (ignored if unused)
    enabled: Optional[Iterable[str]] = None,
    save_spectrograms: bool = False,
    save_connectivity: bool = False,
    save_ssl: bool = False,
    backend: str = "numpy",
    device: str = "cpu",
    connectivity_mode: str = "full",
    max_pairs: Optional[int] = None,
    nperseg: int | None = None,
    noverlap: int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Extract per-epoch features from epoch arrays.

    Args:
        epochs: array shape (n_epochs, n_channels, n_samples) of floats.
        epoch_meta: optional list of per-epoch dicts returned by epoching.make_epochs().
                    Each dict should contain 'start_ts', 'end_ts', 'center_ts', 'session_id', 'sfreq', 'channel_names'.
                    If None, a synthetic meta will be created (legacy / unit-test compatibility).
        sfreq: optional sampling frequency (Hz). If epoch_meta provided, its sfreq is used by default.
        per_channel: if True compute per-channel features (default True).
        channel_names: optional list of channel names; prefers epoch_meta[0]['channel_names'] if omitted.

    Returns:
        pandas.DataFrame: rows per epoch. Columns include:
            epoch_index, start_ts, end_ts, center_ts, session_id, sfreq, n_channels, channel_names,
            per-channel features named e.g. 'delta_CP3_mean', 'theta_CP3_std', 'paf_CP3', 'spec_entropy_CP3', ...
            aggregate columns like 'theta_power_mean', 'total_power_mean', 'coh_alpha', 'perm_entropy_mean', etc.

    Notes:
        - This function accepts many legacy/compatibility kwargs (enabled, save_spectrograms, backend, etc.)
          to avoid breaking callers. Most are ignored here unless they directly control nperseg/noverlap.
    """
    n_epochs = epochs.shape[0]
    if n_epochs == 0:
        return pd.DataFrame()

    # legacy compatibility: if epoch_meta is not provided, create synthetic meta and require sfreq/channel_names
    if epoch_meta is None:
        if sfreq is None:
            raise ValueError("sfreq must be specified if epoch_meta is not provided")
        if channel_names is None:
            # construct generic channel names ch0..chN
            channel_names = [f"ch{i}" for i in range(epochs.shape[1])]
        epoch_meta = []
        for i in range(n_epochs):
            epoch_meta.append(
                {
                    "epoch_index": i,
                    "start_ts": float(i * (epochs.shape[2] / sfreq)),
                    "end_ts": float((i + 1) * (epochs.shape[2] / sfreq)),
                    "center_ts": float(((i * (epochs.shape[2] / sfreq)) + ((i + 1) * (epochs.shape[2] / sfreq))) / 2.0),
                    "session_id": None,
                    "sfreq": float(sfreq),
                    "n_channels": epochs.shape[1],
                    "channel_names": channel_names,
                }
            )

    # fill sfreq and channel_names from meta if not supplied
    if sfreq is None:
        sfreq = float(epoch_meta[0].get("sfreq", None))
    if channel_names is None:
        channel_names = list(epoch_meta[0].get("channel_names", [f"ch{i}" for i in range(epochs.shape[1])]))

    rows = []
    for i in range(n_epochs):
        e = epochs[i]  # shape (n_channels, n_samples)
        meta = epoch_meta[i]
        row: Dict[str, object] = {
            "epoch_index": int(meta.get("epoch_index", i)),
            "start_ts": float(meta["start_ts"]),
            "end_ts": float(meta["end_ts"]),
            "center_ts": float(meta["center_ts"]),
            "session_id": meta.get("session_id"),
            "sfreq": float(sfreq),
            "n_channels": int(meta.get("n_channels", e.shape[0])),
            "channel_names": list(channel_names),
        }

        # compute per-channel features
        for ch_idx, ch_name in enumerate(channel_names):
            x = e[ch_idx, :].astype(float)
            # spectrogram (time-resolved PSD) for band mean/std
            freqs, times, Sxx = _spectrogram(x, sfreq, nperseg=nperseg or min(256, x.size), noverlap=noverlap)
            # full-epoch (Welch) PSD for PAF and 1/f
            freqs_w, Pxx = _psd_welch(x, sfreq, nperseg=nperseg or min(256, x.size))

            # per-band time-series -> mean/std
            for band_name, band_range in BANDS.items():
                band_ts = band_power_time_series_from_spectrogram(freqs, Sxx, band_range)
                row[f"{band_name}_{ch_name}_mean"] = float(np.mean(band_ts))
                row[f"{band_name}_{ch_name}_std"] = float(np.std(band_ts))

            # total power across full PSD and as time-resolved mean
            total_psd = np.trapz(Pxx, freqs_w)
            row[f"total_{ch_name}_mean"] = float(total_psd)
            # time-resolved total power (sum across all freqs per time bin)
            total_ts = np.trapz(Sxx, freqs, axis=0)
            row[f"total_{ch_name}_std"] = float(np.std(total_ts))

            # ratios computed on full-epoch band integrals (from spectrogram mean)
            th = float(row[f"theta_{ch_name}_mean"])
            al = float(row[f"alpha_{ch_name}_mean"])
            be = float(row[f"beta_{ch_name}_mean"])
            row[f"theta_alpha_ratio_{ch_name}"] = float(th / (al + 1e-12))
            row[f"theta_beta_ratio_{ch_name}"] = float(th / (be + 1e-12))
            row[f"(theta+alpha)/beta_{ch_name}"] = float((th + al) / (be + 1e-12))

            # PAF and spectral entropy + 1/f slope using Welch PSD
            row[f"paf_{ch_name}"] = float(peak_alpha_freq_from_psd(freqs_w, Pxx))
            row[f"spec_entropy_{ch_name}"] = float(spectral_entropy_from_psd(Pxx))
            row[f"one_over_f_slope_{ch_name}"] = float(one_over_f_slope(freqs_w, Pxx))

            # time-frequency band means (tf_delta_mean etc.)
            for band_name, band_range in BANDS.items():
                band_ts = band_power_time_series_from_spectrogram(freqs, Sxx, band_range)
                row[f"tf_{band_name}_mean_{ch_name}"] = float(np.mean(band_ts))

            # nonlinear metrics
            row[f"perm_entropy_{ch_name}"] = float(permutation_entropy(x))
            row[f"higuchi_fd_{ch_name}"] = float(higuchi_fd(x))
            row[f"sampen_{ch_name}"] = float(sample_entropy(x))

        # aggregate across channels for compatibility/backwards compatibility
        for band in BANDS.keys():
            vals = [row[f"{band}_{ch}_mean"] for ch in channel_names]
            row[f"{band}_power_mean"] = float(np.mean(vals))
            row[f"{band}_power_std"] = float(np.std(vals))

        totals = [row[f"total_{ch}_mean"] for ch in channel_names]
        row["total_power_mean"] = float(np.mean(totals))
        row["total_power_std"] = float(np.std(totals))

        # ratios
        row["theta_alpha_ratio"] = float(row["theta_power_mean"] / (row["alpha_power_mean"] + 1e-12))
        row["theta_beta_ratio"] = float(row["theta_power_mean"] / (row["beta_power_mean"] + 1e-12))
        row["(theta+alpha)/beta"] = float((row["theta_power_mean"] + row["alpha_power_mean"]) / (row["beta_power_mean"] + 1e-12))

        # pairwise coherence per-band (global average across pairs)
        try:
            coh_map = _bandwise_pairwise_coherence(e, sfreq, bands=BANDS, nperseg=nperseg or 256)
            # store as e.g. coh_delta, coh_theta ...
            for band_name, val in coh_map.items():
                row[f"coh_{band_name}"] = float(val)
        except Exception:
            for band_name in BANDS.keys():
                row[f"coh_{band_name}"] = float("nan")

        # tf aggregated means across channels (existing fields tf_delta_mean etc. - mean across channels)
        for band_name in BANDS.keys():
            vals = [row[f"tf_{band_name}_mean_{ch}"] for ch in channel_names]
            row[f"tf_{band_name}_mean"] = float(np.mean(vals))

        # aggregate nonlinear metrics
        row["perm_entropy_mean"] = float(np.nanmean([row[f"perm_entropy_{ch}"] for ch in channel_names]))
        row["higuchi_fd_mean"] = float(np.nanmean([row[f"higuchi_fd_{ch}"] for ch in channel_names]))
        row["sampen_mean"] = float(np.nanmean([row[f"sampen_{ch}"] for ch in channel_names]))

        # NEW: aggregate spectral entropy and 1/f slope across channels (tests expect these names)
        row["spec_entropy_mean"] = float(np.nanmean([row[f"spec_entropy_{ch}"] for ch in channel_names]))
        row["one_over_f_slope"] = float(np.nanmean([row[f"one_over_f_slope_{ch}"] for ch in channel_names]))

        # channel zero totals metadata
        num_zero = int(np.sum([1 if row[f"total_{ch}_mean"] == 0 else 0 for ch in channel_names]))
        row["num_channels_with_zero_total"] = int(num_zero)
        row["has_any_zero_total"] = bool(num_zero > 0)

        # append row
        rows.append(row)

    df = pd.DataFrame(rows)

    # canonical column ordering: epoch/timestamps/session info first
    front = ["epoch_index", "start_ts", "end_ts", "center_ts", "session_id", "sfreq", "n_channels", "channel_names"]
    rest = [c for c in df.columns if c not in front]
    df = df[front + sorted(rest)]
    return df