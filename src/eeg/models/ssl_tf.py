# src/eeg/models/ssl_tf.py
"""Self-Supervised Learning (SSL) model for EEG signals.

This module defines:
- SSLEncoder: CNN encoder for EEG sequences.
- SSLProjectionHead: Projector network for contrastive learning.
- SSLModelPL: PyTorch Lightning wrapper with robust logging and NaN detection.

The model handles variable-length sequences via global average pooling.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class SSLEncoder(nn.Module):
    """CNN encoder for EEG signals.

    Input: [B, C, T]
    Output: [B, hidden, T]
    """
    def __init__(self, in_channels: int = 9, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        return self.net(x)


class SSLProjectionHead(nn.Module):
    """Projection head.

    Input: [B, hidden]
    Output: [B, proj_dim]
    """
    def __init__(self, hidden: int = 256, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SSLModelPL(pl.LightningModule):
    """LightningModule for SSL training with NT-Xent loss."""

    def __init__(
        self,
        encoder_in_channels: int = 9,
        encoder_hidden: int = 256,
        proj_dim: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = SSLEncoder(in_channels=encoder_in_channels, hidden=encoder_hidden)
        self.projector = SSLProjectionHead(hidden=encoder_hidden, proj_dim=proj_dim)

        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.temperature = float(temperature)

        # will be set when configure_optimizers is called
        self._optimizer_ref = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        h = self.encoder(x)           # [B, hidden, T]
        h = h.mean(dim=-1)            # global average pooling -> [B, hidden]
        z = self.projector(h)         # [B, proj_dim]
        return z

    def _log_input_stats(self, x: torch.Tensor, prefix: str) -> None:
        with torch.no_grad():
            xi = x.detach().cpu()
            self.log(f"{prefix}/min", float(xi.min()), on_step=True, on_epoch=False, logger=True)
            self.log(f"{prefix}/max", float(xi.max()), on_step=True, on_epoch=False, logger=True)
            self.log(f"{prefix}/mean", float(xi.mean()), on_step=True, on_epoch=False, logger=True)
            self.log(f"{prefix}/std", float(xi.std()), on_step=True, on_epoch=False, logger=True)

    def _current_lr(self) -> float:
        if self.trainer is None:
            return 0.0
        optimizers = getattr(self.trainer, "optimizers", None)
        if not optimizers:
            return 0.0
        opt = optimizers[0]
        return float(opt.param_groups[0].get("lr", 0.0))

    @torch.cuda.amp.autocast(enabled=False)
    def _nt_xent_loss(self, z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
        """Normalized Temperature-Scaled Cross Entropy Loss (SimCLR)."""
        z1 = z1.float()
        z2 = z2.float()
        batch_size = z1.size(0)

        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        z = torch.cat([z1, z2], dim=0)  # [2*B, D]
        sim = torch.matmul(z, z.T) / temperature  # [2B, 2B]

        # Mask self-similarity
        diag_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=sim.device)
        sim = sim.masked_fill(diag_mask, float('-inf'))

        # Labels: positive pairs are diagonal offset by B
        targets = torch.arange(batch_size, device=sim.device)
        targets = torch.cat([targets + batch_size, targets], dim=0)

        loss = F.cross_entropy(sim, targets)
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> Optional[torch.Tensor]:
        x1, x2 = batch
        if batch_idx == 0:
            self._log_input_stats(x1, "input/x1")
            self._log_input_stats(x2, "input/x2")

        z1 = self.forward(x1)
        z2 = self.forward(x2)

        if not torch.isfinite(z1).all() or not torch.isfinite(z2).all():
            self.log("debug/z_has_inf_or_nan", 1.0, on_step=True, logger=True)
            print(f"NaN/Inf detected in model outputs at train batch {batch_idx}")
            return None

        loss = self._nt_xent_loss(z1, z2, temperature=self.temperature)

        if not torch.isfinite(loss):
            self.log("debug/loss_is_nan", 1.0, on_step=True, logger=True)
            print(f"NaN loss at train batch {batch_idx}; lr={self._current_lr():.3e}")
            return None

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("lr", self._current_lr(), on_step=True, on_epoch=False, logger=True)
        return loss

    def on_after_backward(self) -> None:
        total_norm = 0.0
        found = False
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                found = True
        if found:
            total_norm = total_norm ** 0.5
            self.log("grad/grad_norm", float(total_norm), on_step=True, on_epoch=False, logger=True)

    def validation_step(self, batch: Any, batch_idx: int) -> Optional[torch.Tensor]:
        x1, x2 = batch
        if batch_idx == 0:
            self._log_input_stats(x1, "val_input/x1")
            self._log_input_stats(x2, "val_input/x2")

        z1 = self.forward(x1)
        z2 = self.forward(x2)

        if not torch.isfinite(z1).all() or not torch.isfinite(z2).all():
            self.log("debug/val_z_has_inf_or_nan", 1.0, on_step=False, on_epoch=True, logger=True)
            print(f"NaN/Inf detected in val model outputs at batch {batch_idx}")
            return None

        val_loss = self._nt_xent_loss(z1, z2, temperature=self.temperature)
        if not torch.isfinite(val_loss):
            self.log("debug/val_loss_is_nan", 1.0, on_step=False, on_epoch=True, logger=True)
            print(f"NaN val loss at batch {batch_idx}")
            return None

        self.log("val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self._optimizer_ref = optimizer
        return optimizer