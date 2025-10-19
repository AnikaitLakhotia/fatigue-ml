# src/eeg/models/autoencoder.py
from __future__ import annotations

"""
Simple Autoencoder baseline using PyTorch.

Classes and functions:
  - Autoencoder(nn.Module): small MLP autoencoder
  - TrainAutoencoderConfig: dataclass for config
  - train_autoencoder(features_df, cfg, out_dir): training loop saving model and stats

This autoencoder is intentionally small and designed for tabular feature inputs (per-epoch vectors).
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
import json

from src.eeg.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except Exception:  # pragma: no cover - import guard
    torch = None
    nn = None
    DataLoader = None


class _NumpyDataset(Dataset):
    def __init__(self, arr: np.ndarray):
        self.arr = arr.astype(np.float32)

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, idx):
        return self.arr[idx]


if torch is not None:

    class Autoencoder(nn.Module):
        """Small MLP autoencoder for tabular features."""

        def __init__(self, input_dim: int, latent_dim: int = 16):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, max(64, input_dim)),
                nn.ReLU(),
                nn.Linear(max(64, input_dim), latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, max(64, input_dim)),
                nn.ReLU(),
                nn.Linear(max(64, input_dim), input_dim),
            )

        def forward(self, x):
            z = self.encoder(x)
            x_rec = self.decoder(z)
            return x_rec, z

else:
    Autoencoder = None  # type: ignore


@dataclass
class TrainAutoencoderConfig:
    """Configuration for training autoencoder."""

    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    latent_dim: int = 16
    device: str = "cpu"


def train_autoencoder(
    features_df: pd.DataFrame, cfg: TrainAutoencoderConfig, out_dir: str | Path
) -> Dict[str, Any]:
    """
    Train autoencoder on numeric columns of features_df.

    Args:
        features_df: DataFrame with numeric columns (rows=epochs)
        cfg: TrainAutoencoderConfig
        out_dir: directory to write model & metrics

    Returns:
        dict with training metrics and saved paths
    """
    if torch is None:
        raise RuntimeError(
            "PyTorch not available. Install torch to use autoencoder training."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = features_df.select_dtypes(include="number").values.astype(np.float32)
    if X.shape[0] < 2:
        raise ValueError("Need at least 2 rows to train autoencoder.")

    # standardize
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    Xs = (X - mean) / std

    input_dim = Xs.shape[1]
    model = Autoencoder(input_dim=input_dim, latent_dim=cfg.latent_dim)
    device = torch.device(
        cfg.device if torch.cuda.is_available() and cfg.device != "cpu" else "cpu"
    )
    model.to(device)

    ds = _NumpyDataset(Xs)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    history = {"loss": []}
    model.train()
    for ep in range(cfg.epochs):
        epoch_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            rec, _ = model(batch)
            loss = loss_fn(rec, batch)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item()) * batch.size(0)
        epoch_loss /= len(ds)
        history["loss"].append(epoch_loss)
        logger.info("AE epoch %d/%d: loss=%.6f", ep + 1, cfg.epochs, epoch_loss)

    model_path = out_dir / "autoencoder.pt"
    torch.save(
        {"model_state": model.state_dict(), "mean": mean, "std": std}, model_path
    )
    metrics_path = out_dir / "ae_training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(history, f)
    logger.info("Saved AE model -> %s and metrics -> %s", model_path, metrics_path)

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "history": history,
    }
