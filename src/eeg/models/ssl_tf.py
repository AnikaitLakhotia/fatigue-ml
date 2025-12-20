# src/eeg/models/ssl_tf.py
"""Self-Supervised Learning (SSL) model for EEG signals.

This module defines:
- SSLEncoder: CNN encoder for EEG sequences.
- SSLProjectionHead: Projector network for contrastive learning.
- SSLModelPL: PyTorch Lightning wrapper with robust logging and NaN detection.

The LightningModule is defensive: it's safe to call `training_step` / `validation_step`
without an attached Trainer (useful for unit tests). When called *without* a Trainer
and a non-finite value is detected, the module raises RuntimeError so tests that
expect failure can observe it. When attached to a Trainer the module logs metrics
and returns None on problematic batches to allow Trainer-driven handling.
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
    """LightningModule for SSL training with NT-Xent loss.

    Defensive behavior:
      - Safe to call `training_step` / `validation_step` without an attached Trainer.
      - If called without a Trainer and non-finite outputs (or loss) are detected,
        raises RuntimeError so unit tests can assert failure.
      - If attached to a Trainer, logs metrics and returns None for problematic batches.
    """

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

        # optimizer ref (for later inspection)
        self._optimizer_ref = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        h = self.encoder(x)  # [B, hidden, T]
        h = h.mean(dim=-1)  # global average pooling -> [B, hidden]
        z = self.projector(h)  # [B, proj_dim]
        return z

    # Trainer-safe helpers
    def _has_trainer(self) -> bool:
        """
        Return True if the module is attached to a Trainer.

        Accessing `self.trainer` directly can raise RuntimeError (Lightning property),
        so wrap in try/except.
        """
        try:
            _ = self.trainer
            return True
        except RuntimeError:
            return False

    def _safe_log(self, *args, **kwargs) -> None:
        """
        Call Lightning `self.log` only when attached to a Trainer.

        If no Trainer is attached this is a no-op.
        """
        if not self._has_trainer():
            return
        try:
            self.log(*args, **kwargs)
        except Exception:
            # swallow logging errors when used under unusual test harnesses
            return

    def _current_lr(self) -> float:
        """
        Return the current learning rate when a Trainer/optimizer exists, else 0.0.
        """
        if not self._has_trainer():
            return 0.0
        try:
            optimizers = getattr(self.trainer, "optimizers", None)
            if not optimizers:
                return 0.0
            return float(optimizers[0].param_groups[0].get("lr", 0.0))
        except Exception:
            return 0.0

    # ----------------------
    # Loss (NT-Xent)
    # ----------------------
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
        sim = sim.masked_fill(diag_mask, float("-inf"))

        # Labels: positive pairs are diagonal offset by B
        targets = torch.arange(batch_size, device=sim.device)
        targets = torch.cat([targets + batch_size, targets], dim=0)

        loss = F.cross_entropy(sim, targets)
        return loss

    # Steps
    def training_step(self, batch: Any, batch_idx: int) -> Optional[torch.Tensor]:
        """
        Training step that returns a loss tensor.

        Behavior:
          - If non-finite embeddings or loss are encountered and NO Trainer is attached,
            raise RuntimeError (so unit tests can assert failures).
          - If a Trainer is attached, log and return None for problematic batches.
        """
        x1, x2 = batch
        if batch_idx == 0:
            self._log_input_stats(x1, "input/x1")
            self._log_input_stats(x2, "input/x2")

        z1 = self.forward(x1)
        z2 = self.forward(x2)

        # detect NaN/Inf in embeddings
        if not torch.isfinite(z1).all() or not torch.isfinite(z2).all():
            if not self._has_trainer():
                # in direct-call (unit test) mode, raise so tests can detect the failure
                raise RuntimeError("NaN/Inf detected in model outputs")
            # under Trainer, log and skip this batch
            self._safe_log("debug/z_has_inf_or_nan", 1.0, on_step=True, logger=True)
            return None

        loss = self._nt_xent_loss(z1, z2, temperature=self.temperature)

        # detect NaN/Inf in loss
        if not torch.isfinite(loss):
            if not self._has_trainer():
                raise RuntimeError("Loss is NaN or Inf")
            self._safe_log("debug/loss_is_nan", 1.0, on_step=True, logger=True)
            # report learning rate if available (no-op if not)
            self._safe_log("lr", self._current_lr(), on_step=True, on_epoch=False, logger=True)
            return None

        # normal logging when attached to Trainer
        self._safe_log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self._safe_log("lr", self._current_lr(), on_step=True, on_epoch=False, logger=True)
        return loss

    def on_after_backward(self) -> None:
        """
        Compute and log gradient norm if gradients available.
        """
        total_norm = 0.0
        found = False
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                found = True
        if found:
            total_norm = total_norm ** 0.5
            self._safe_log("grad/grad_norm", float(total_norm), on_step=True, on_epoch=False, logger=True)

    def validation_step(self, batch: Any, batch_idx: int) -> Optional[torch.Tensor]:
        """
        Validation step that returns validation loss.

        Same NaN semantics as training_step (raise when no Trainer attached).
        """
        x1, x2 = batch
        if batch_idx == 0:
            self._log_input_stats(x1, "val_input/x1")
            self._log_input_stats(x2, "val_input/x2")

        z1 = self.forward(x1)
        z2 = self.forward(x2)

        if not torch.isfinite(z1).all() or not torch.isfinite(z2).all():
            if not self._has_trainer():
                raise RuntimeError("NaN/Inf detected in val model outputs")
            self._safe_log("debug/val_z_has_inf_or_nan", 1.0, on_step=False, on_epoch=True, logger=True)
            return None

        val_loss = self._nt_xent_loss(z1, z2, temperature=self.temperature)
        if not torch.isfinite(val_loss):
            if not self._has_trainer():
                raise RuntimeError("NaN val loss detected")
            self._safe_log("debug/val_loss_is_nan", 1.0, on_step=False, on_epoch=True, logger=True)
            return None

        self._safe_log("val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    # Utils
    def _log_input_stats(self, x: torch.Tensor, prefix: str) -> None:
        """Log simple input statistics (safe without Trainer)."""
        if x is None:
            return
        with torch.no_grad():
            xi = x.detach().cpu()
            self._safe_log(f"{prefix}/min", float(xi.min()), on_step=True, on_epoch=False, logger=True)
            self._safe_log(f"{prefix}/max", float(xi.max()), on_step=True, on_epoch=False, logger=True)
            self._safe_log(f"{prefix}/mean", float(xi.mean()), on_step=True, on_epoch=False, logger=True)
            self._safe_log(f"{prefix}/std", float(xi.std()), on_step=True, on_epoch=False, logger=True)

    def configure_optimizers(self):
        """Configure optimizer (Adam)."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self._optimizer_ref = optimizer
        return optimizer