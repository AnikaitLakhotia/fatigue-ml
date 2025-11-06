"""
_fast_spectral.py - focused, high-performance spectral utilities.

Public:
- welch_psd_batched(epochs, sfreq, nperseg, noverlap, backend, device)
- spectrogram_batched(epochs, sfreq, nperseg, noverlap, backend, device)
- pairwise_coherence_batched(epochs, sfreq, nperseg, noverlap, backend, device, pairs)

Notes:
- epochs: np.ndarray shaped (n_epochs, n_channels, n_samples)
- backend: 'numpy' or 'torch'
- If backend == 'torch' then torch must be installed and device may be 'cuda' or 'cpu'
"""
from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import numpy as np

_BACKENDS = ("numpy", "torch")


def _get_torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception:
        return None


def _frame_numpy(x: np.ndarray, nperseg: int, noverlap: int) -> np.ndarray:
    """
    Create framed windows view for vectorized FFT.
    Input x: (n_epochs, n_ch, n_samples)
    Output: (n_epochs, n_ch, n_windows, nperseg)
    """
    n_epochs, n_ch, n_samples = x.shape
    step = nperseg - noverlap
    if step <= 0:
        raise ValueError("nperseg must be greater than noverlap")
    n_windows = 1 + (n_samples - nperseg) // step if n_samples >= nperseg else 1
    if n_windows <= 0:
        pad = nperseg - n_samples
        x = np.pad(x, ((0, 0), (0, 0), (0, pad)), mode="constant")
        n_windows = 1 + (x.shape[2] - nperseg) // step

    s0, s1, s2 = x.strides
    shape = (n_epochs, n_ch, n_windows, nperseg)
    strides = (s0, s1, s2 * step, s2)
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def _welch_numpy(epochs: np.ndarray, sfreq: float, nperseg: int, noverlap: int) -> Tuple[np.ndarray, np.ndarray]:
    # epochs: (n_epochs, n_ch, n_samples)
    win = np.hanning(nperseg).astype(epochs.dtype)
    windows = _frame_numpy(epochs, nperseg, noverlap)  # (n_e, n_ch, n_w, nperseg)
    windows = windows * win[None, None, None, :]
    fft = np.fft.rfft(windows, axis=-1)  # (n_e, n_ch, n_w, n_freqs)
    psd = (np.abs(fft) ** 2).mean(axis=2)  # average over windows -> (n_e, n_ch, n_freqs)
    U = (win ** 2).sum()
    psd = psd / (sfreq * U)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / sfreq)
    return psd, freqs


def _spectrogram_numpy(epochs: np.ndarray, sfreq: float, nperseg: int, noverlap: int):
    win = np.hanning(nperseg).astype(epochs.dtype)
    windows = _frame_numpy(epochs, nperseg, noverlap)
    windows = windows * win[None, None, None, :]
    fft = np.fft.rfft(windows, axis=-1)  # (n_e, n_ch, n_w, n_freqs)
    S = (np.abs(fft) ** 2) / ((win ** 2).sum() * sfreq)
    S = np.transpose(S, (0, 1, 3, 2))  # (n_e, n_ch, n_freqs, n_times)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / sfreq)
    step = nperseg - noverlap
    times = (np.arange(S.shape[-1]) * step + (nperseg / 2.0)) / sfreq
    return S, freqs, times


def _pairwise_coherence_numpy(epochs: np.ndarray, sfreq: float, nperseg: int, noverlap: int, pairs: Optional[List[tuple]] = None):
    """
    Compute average coherence per epoch across frequencies.
    Returns (n_epochs, n_ch, n_ch) or (n_epochs, n_pairs) if 'pairs' provided.
    """
    n_epochs, n_ch, n_samples = epochs.shape
    win = np.hanning(nperseg).astype(epochs.dtype)
    windows = _frame_numpy(epochs, nperseg, noverlap)  # (n_e, n_ch, n_w, nperseg)
    windows = windows * win[None, None, None, :]
    fft = np.fft.rfft(windows, axis=-1)  # (n_e, n_ch, n_w, n_freqs)
    # average across windows -> (n_e, n_ch, n_freqs)
    F = fft.mean(axis=2)
    # cross-spectrum: (n_e, n_ch, n_ch, n_freqs)
    CS = np.einsum("eif,ejf->eijf", F, np.conjugate(F))
    Sxx = (np.abs(F) ** 2)
    Sxx_pair = Sxx[:, :, None, :]  # (n_e, n_ch, 1, n_freqs)
    Syy_pair = Sxx[:, None, :, :]  # (n_e, 1, n_ch, n_freqs)
    denom = np.sqrt(Sxx_pair * Syy_pair)
    denom = np.where(denom <= 0, 1e-12, denom)
    coh_freq = np.abs(CS) / denom
    C = coh_freq.mean(axis=-1)  # average across freqs -> (n_e, n_ch, n_ch)
    if pairs:
        out = np.stack([C[:, i, j] for (i, j) in pairs], axis=1)
        return out, np.fft.rfftfreq(nperseg, d=1.0 / sfreq)
    return C, np.fft.rfftfreq(nperseg, d=1.0 / sfreq)


def _welch_torch(epochs, sfreq: float, nperseg: int, noverlap: int):
    torch = _get_torch()
    if torch is None:
        raise RuntimeError("torch not installed")
    n_epochs, n_ch, n_samples = epochs.shape
    device = epochs.device
    batch = epochs.reshape(-1, n_samples)  # (B, T)
    hop = nperseg - noverlap
    if nperseg > n_samples:
        batch = torch.nn.functional.pad(batch, (0, nperseg - n_samples))
    window = torch.hann_window(nperseg, device=device)
    stft = torch.stft(batch, n_fft=nperseg, hop_length=hop, win_length=nperseg, window=window, return_complex=True)
    # stft: (B, freq_bins, n_frames)
    P = stft.abs() ** 2
    P = P.mean(dim=-1)  # average across frames -> (B, freq_bins)
    freqs = torch.fft.rfftfreq(nperseg, d=1.0 / sfreq) if hasattr(torch.fft, "rfftfreq") else torch.linspace(0, sfreq / 2.0, P.shape[1], device=device)
    P = P.reshape(n_epochs, n_ch, -1).cpu().numpy() / ((window ** 2).sum() * sfreq)
    return P, freqs.cpu().numpy()


def _pairwise_coherence_torch(epochs, sfreq: float, nperseg: int, noverlap: int, pairs: Optional[List[tuple]] = None):
    torch = _get_torch()
    if torch is None:
        raise RuntimeError("torch not installed")
    n_epochs, n_ch, n_samples = epochs.shape
    device = epochs.device
    batch = epochs.reshape(-1, n_samples)
    hop = nperseg - noverlap
    if nperseg > n_samples:
        batch = torch.nn.functional.pad(batch, (0, nperseg - n_samples))
    window = torch.hann_window(nperseg, device=device)
    stft = torch.stft(batch, n_fft=nperseg, hop_length=hop, win_length=nperseg, window=window, return_complex=True)
    B, n_freqs, n_frames = stft.shape
    stft = stft.reshape(n_epochs, n_ch, n_freqs, n_frames)
    F = stft.mean(dim=-1)  # (n_e, n_ch, n_freqs)
    Fi = F.unsqueeze(2)  # (n_e, n_ch, 1, n_freqs)
    Fj = F.unsqueeze(1)  # (n_e, 1, n_ch, n_freqs)
    CS = Fi * torch.conj(Fj)
    Sxx = (F.abs() ** 2)
    denom = torch.sqrt(Sxx[:, :, None, :] * Sxx[:, None, :, :])
    denom = torch.where(denom <= 0, torch.tensor(1e-12, device=device), denom)
    coh = (CS.abs() / denom).mean(dim=-1)  # (n_e, n_ch, n_ch)
    if pairs:
        out = torch.stack([coh[:, i, j] for (i, j) in pairs], dim=1).cpu().numpy()
        freqs = torch.fft.rfftfreq(nperseg, d=1.0 / sfreq).cpu().numpy()
        return out, freqs
    freqs = torch.fft.rfftfreq(nperseg, d=1.0 / sfreq).cpu().numpy()
    return coh.cpu().numpy(), freqs


# Public API


def welch_psd_batched(epochs: np.ndarray, sfreq: float, nperseg: int = 256, noverlap: int = 128, backend: str = "numpy") -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute batched Welch PSD:
      epochs -> Pxx (n_epochs, n_ch, n_freqs), freqs
    """
    backend = backend or "numpy"
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}")
    if backend == "torch":
        torch = _get_torch()
        if torch is None:
            raise RuntimeError("torch requested but not installed")
        ep = torch.as_tensor(epochs, device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        return _welch_torch(ep, sfreq, nperseg, noverlap)
    # numpy path
    if epochs.dtype not in (np.float32, np.float64):
        epochs = epochs.astype(np.float32, copy=False)
    return _welch_numpy(epochs, sfreq, nperseg, noverlap)


def spectrogram_batched(epochs: np.ndarray, sfreq: float, nperseg: int = 256, noverlap: int = 128, backend: str = "numpy"):
    """
    Compute spectrograms batched:
      returns S (n_epochs, n_ch, n_freqs, n_times), freqs, times
    """
    if backend == "torch":
        torch = _get_torch()
        if torch is None:
            raise RuntimeError("torch requested but not installed")
        ep = torch.as_tensor(epochs, device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        # re-use torch stft flow from _welch_torch but preserve frames dimension
        n_epochs, n_ch, n_samples = ep.shape
        batch = ep.reshape(-1, n_samples)
        hop = nperseg - noverlap
        window = torch.hann_window(nperseg, device=batch.device)
        stft = torch.stft(batch, n_fft=nperseg, hop_length=hop, win_length=nperseg, window=window, return_complex=True)
        S = (stft.abs() ** 2) / ((window ** 2).sum() * sfreq)  # (B, n_freqs, n_frames)
        B, n_freqs, n_frames = S.shape
        S = S.reshape(n_epochs, n_ch, n_freqs, n_frames).cpu().numpy()
        freqs = torch.fft.rfftfreq(nperseg, d=1.0 / sfreq).cpu().numpy()
        times = (np.arange(S.shape[-1]) * (nperseg - noverlap) + (nperseg / 2.0)) / sfreq
        return S, freqs, times
    # numpy
    return _spectrogram_numpy(epochs, sfreq, nperseg, noverlap)


def pairwise_coherence_batched(epochs: np.ndarray, sfreq: float, nperseg: int = 256, noverlap: int = 128, backend: str = "numpy", pairs: Optional[List[tuple]] = None):
    """
    Compute pairwise coherence per epoch. If pairs provided, returns (n_epochs, n_pairs).
    """
    if backend == "torch":
        torch = _get_torch()
        if torch is None:
            raise RuntimeError("torch requested but not installed")
        ep = torch.as_tensor(epochs, device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        return _pairwise_coherence_torch(ep, sfreq, nperseg, noverlap, pairs=pairs)
    return _pairwise_coherence_numpy(epochs, sfreq, nperseg, noverlap, pairs=pairs)
