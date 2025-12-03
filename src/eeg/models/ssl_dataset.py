# src/eeg/models/ssl_dataset.py
"""Dataset and augmentation utilities for self-supervised EEG training.

Provides:
- SSLDataset: loads EEG arrays from supported files and returns two augmented views.
- SSLAugmentations: light augmentations (scaling + additive noise).
- pad_collate_fn: collate function that pads variable-length sequences to the batch max length.
- _load_array_lazy: loader that supports .fif, .npy, .npz, .pt and sanitizes NaN/Inf values.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import random

# Lazy import for MNE only when needed
try:
    import mne  # type: ignore
except Exception:
    mne = None  # runtime will raise if a .fif file is loaded without mne


class SSLAugmentations:
    """Simple EEG augmentations applied per-sample to produce two views.

    - Per-channel random scaling
    - Additive Gaussian noise
    """
    def __init__(self, noise_level: float = 0.02, scale_range: Tuple[float, float] = (0.95, 1.05)):
        self.noise_level = float(noise_level)
        self.scale_range = tuple(scale_range)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Expect x shape: (channels, samples)
        if x.ndim == 1:
            x = x[None, :]
        ch = x.shape[0]
        scales = np.random.uniform(self.scale_range[0], self.scale_range[1], size=(ch, 1))
        x = x * scales
        noise = np.random.normal(0.0, self.noise_level, size=x.shape)
        return x + noise


def _safe_zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-channel z-score with small epsilon to avoid division by zero."""
    if x.ndim == 1:
        x = x[None, :]
    mean = np.nanmean(x, axis=1, keepdims=True)
    std = np.nanstd(x, axis=1, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (x - mean) / std


def _load_array_lazy(p: str) -> np.ndarray:
    """Load an EEG array from disk, sanitize NaN/Inf, and return float32 array.

    Supported formats:
      - .npy : saved numpy array
      - .npz : compressed archive (first array taken)
      - .pt  : torch tensor saved with torch.save
      - .fif : mne Raw file (requires mne installed)

    Returned shape: (channels, samples)
    """
    pth = str(p)
    if pth.endswith(".npy"):
        arr = np.load(pth, allow_pickle=False)
    elif pth.endswith(".npz"):
        data = np.load(pth, allow_pickle=False)
        # pick first array in archive
        first_key = list(data.files)[0]
        arr = data[first_key]
    elif pth.endswith(".pt"):
        t = torch.load(pth, map_location="cpu")
        if isinstance(t, np.ndarray):
            arr = t
        else:
            try:
                arr = t.numpy()
            except Exception:
                raise ValueError(f"Unsupported .pt content for file {pth}")
    elif pth.endswith(".fif"):
        if mne is None:
            raise ImportError("mne is required to read .fif files (install mne).")
        raw = mne.io.read_raw_fif(pth, preload=True, verbose=False)
        arr = raw.get_data()  # shape (n_channels, n_samples)
    else:
        raise ValueError(f"Unsupported array source: {p}")

    # Ensure 2D array (channels, samples)
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[None, :]

    # Promote to float64 for stable sanitization
    arr = arr.astype(np.float64, copy=False)

    # Replace infinite values with NaN so they get handled below
    arr[np.isinf(arr)] = np.nan

    # If entire channel is NaN, replace with zeros
    channel_all_nan = np.isnan(arr).all(axis=1)
    if np.any(channel_all_nan):
        arr[channel_all_nan, :] = 0.0

    # Fill remaining NaNs with channel mean
    ch_mean = np.nanmean(arr, axis=1, keepdims=True)
    inds = np.isnan(arr)
    if inds.any():
        arr[inds] = np.take(ch_mean, np.where(inds)[0])

    # Per-channel zscore to have stable numeric ranges
    arr = _safe_zscore(arr, eps=1e-8).astype(np.float32)

    # Clip to a reasonable range to protect against outliers
    np.clip(arr, -1e3, 1e3, out=arr)

    return arr


class SSLDataset(Dataset):
    """Dataset that loads EEG arrays and returns two augmented views for SSL.

    Each __getitem__ returns a tuple (view1, view2) where each view is a torch.FloatTensor
    shaped (channels, samples).
    """
    def __init__(self, sources: List[str], transforms: Optional[SSLAugmentations] = None):
        self.sources = [str(s) for s in sources]
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, idx: int):
        p = self.sources[idx]
        x = _load_array_lazy(p)  # numpy array (C, T)

        if self.transforms is not None:
            v1 = self.transforms(x)
            v2 = self.transforms(x)
        else:
            v1 = x.copy()
            v2 = x.copy()

        # convert to torch tensors
        return torch.tensor(v1, dtype=torch.float32), torch.tensor(v2, dtype=torch.float32)


def pad_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """Collate function that pads variable-length sequences in a batch.

    Returns:
        x1_padded, x2_padded: tensors shaped (B, C, L_max)
    """
    x1_list, x2_list = zip(*batch)

    def pad_list(x_list):
        # x: torch.Tensor [C, L]
        max_len = max(x.shape[1] for x in x_list)
        channels = x_list[0].shape[0]
        batch_size = len(x_list)
        out = torch.zeros((batch_size, channels, max_len), dtype=torch.float32)
        for i, x in enumerate(x_list):
            out[i, :, : x.shape[1]] = x
        return out

    return pad_list(x1_list), pad_list(x2_list)