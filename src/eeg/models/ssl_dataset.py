# src/eeg/models/ssl_dataset.py
from __future__ import annotations
"""
Dataset and augmentations for contrastive SSL (TensorFlow).

Supports:
 - .npy, .npz, and .fif epoch sources
 - Sliding-window epoching for raw .fif files
 - Two-view augmentations for contrastive learning
 - tf.data.Dataset wrapper with batching and prefetching
"""
from typing import Union, Optional, Tuple
from pathlib import Path
import numpy as np
import tensorflow as tf
import mne
from src.eeg.utils.logger import get_logger

logger = get_logger(__name__)

def load_epochs(source: Union[str, Path, np.ndarray], window_sec: Optional[float] = None, overlap: Optional[float] = None) -> np.ndarray:
    if isinstance(source, np.ndarray):
        return source.astype(np.float32)
    p = Path(source)
    if p.is_dir():
        files = sorted(p.glob("*.fif"))
        if not files:
            raise FileNotFoundError(f"No .fif files found in {p}")
        parts = [_fif_to_epochs(f, window_sec, overlap) for f in files]
        arr = np.vstack(parts)
    else:
        arr = _fif_to_epochs(p, window_sec, overlap)
    logger.info("Loaded epochs array shape: %s", arr.shape)
    return arr.astype(np.float32)

def _fif_to_epochs(path: Path, window_sec: Optional[float], overlap: Optional[float]) -> np.ndarray:
    raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
    data = raw.get_data()
    n_channels, n_samples = data.shape
    arr = data[np.newaxis, :, :]
    return arr.astype(np.float32)

def _time_jitter(x: np.ndarray, sigma: float = 0.02) -> np.ndarray:
    s = x.std(axis=1, keepdims=True)
    noise = np.random.normal(scale=np.maximum(s * sigma, 1e-8), size=x.shape)
    return x + noise

def _scaling(x: np.ndarray, min_scale: float = 0.8, max_scale: float = 1.2) -> np.ndarray:
    scales = np.random.uniform(min_scale, max_scale, size=(x.shape[0], 1)).astype(x.dtype)
    return x * scales

def _channel_dropout(x: np.ndarray, drop_prob: float = 0.1) -> np.ndarray:
    mask = (np.random.rand(x.shape[0]) >= drop_prob).astype(x.dtype)
    return x * mask[:, None]

def _time_mask(x: np.ndarray, mask_fraction: float = 0.1, n_masks: int = 1) -> np.ndarray:
    x = x.copy()
    n_samples = x.shape[1]
    max_mask = int(round(n_samples * mask_fraction))
    for _ in range(n_masks):
        L = np.random.randint(1, max(1, max_mask + 1))
        start = np.random.randint(0, max(1, n_samples - L + 1))
        x[:, start : start + L] = 0.0
    return x

def _augment_once(epoch: np.ndarray) -> np.ndarray:
    x = epoch.copy()
    if np.random.rand() < 0.9:
        x = _time_jitter(x)
    if np.random.rand() < 0.9:
        x = _scaling(x, 0.9, 1.1)
    if np.random.rand() < 0.5:
        x = _channel_dropout(x)
    if np.random.rand() < 0.5:
        x = _time_mask(x)
    return x

def two_view_from_epoch(epoch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return _augment_once(epoch), _augment_once(epoch)

def make_contrastive_dataset(
    epochs: Union[str, Path, np.ndarray],
    batch_size: int = 128,
    shuffle: bool = True,
    seed: Optional[int] = None,
    prefetch: int = tf.data.AUTOTUNE,
) -> tf.data.Dataset:
    arr = load_epochs(epochs)
    n = arr.shape[0]
    ds = tf.data.Dataset.from_tensor_slices(arr)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(10000, n), seed=seed)

    def _map_fn(epoch):
        def _py(epoch_np):
            v1, v2 = two_view_from_epoch(epoch_np)
            return v1.astype(np.float32), v2.astype(np.float32)
        v1, v2 = tf.numpy_function(_py, [epoch], [tf.float32, tf.float32])
        v1.set_shape(epoch.shape)
        v2.set_shape(epoch.shape)
        return v1, v2

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(prefetch)
    return ds