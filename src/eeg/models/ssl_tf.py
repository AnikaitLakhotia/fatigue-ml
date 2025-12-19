# src/eeg/models/ssl_tf.py
"""Self-Supervised Learning (SSL) model for EEG signals.

This module defines:
- SSLEncoder: CNN encoder for EEG sequences.
- SSLProjectionHead: Projector network for contrastive learning.
- SSLModelPL: PyTorch Lightning wrapper with robust logging and NaN detection.

Improvements in this commit:
- Robust checks for NaN/Inf in model outputs and loss (fail-fast behavior).
- Stable NT-Xent (SimCLR) loss implementation with explicit masking; numeric
  stability considerations (avoid NaN caused by -inf in softmax).
- No use of autocast decorator inside loss (explicit dtype handling).
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
        """Forward pass.

        Args:
            x: Tensor of shape [B, C, T]

        Returns:
            Tensor of shape [B, hidden, T]
        """
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
        """Forward projection.

        Args:
            x: Tensor of shape [B, hidden]

        Returns:
            Tensor of shape [B, proj_dim]
        """
        return self.net(x)


class SSLModelPL(pl.LightningModule):
    """LightningModule for SSL training with NT-Xent loss.

    This LightningModule is defensive: it validates embeddings and losses
    for finite values and fails fast (raises informative RuntimeErrors)
    so downstream training infra notices issues instead of silently skipping
    optimization steps.
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
        # save hyperparameters for reproducibility and logging
        self.save_hyperparameters()

        self.encoder = SSLEncoder(in_channels=encoder_in_channels, hidden=encoder_hidden)
        self.projector = SSLProjectionHead(hidden=encoder_hidden, proj_dim=proj_dim)

        # optimizer hyperparams (also available via self.hparams)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.temperature = float(temperature)

        # internal ref (set in configure_optimizers) - helpful for logging lr in training
        self._optimizer_ref = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute projected embeddings for input batch.

        Args:
            x: Tensor [B, C, T]

        Returns:
            z: Tensor [B, proj_dim]
        """
        h = self.encoder(x)  # [B, hidden, T]
        h = h.mean(dim=-1)  # global average pooling -> [B, hidden]
        z = self.projector(h)  # [B, proj_dim]
        return z

    def _log_input_stats(self, x: torch.Tensor, prefix: str) -> None:
        """Log simple stats about input tensors for debugging.

        Args:
            x: Tensor, typically on CPU or accessible device
            prefix: logging prefix (e.g., "input/x1")
        """
        with torch.no_grad():
            xi = x.detach().cpu()
            self.log(f"{prefix}/min", float(xi.min()), on_step=True, on_epoch=False, logger=True)
            self.log(f"{prefix}/max", float(xi.max()), on_step=True, on_epoch=False, logger=True)
            self.log(f"{prefix}/mean", float(xi.mean()), on_step=True, on_epoch=False, logger=True)
            self.log(f"{prefix}/std", float(xi.std()), on_step=True, on_epoch=False, logger=True)

    def _current_lr(self) -> float:
        """Return current LR (first parameter group) if available, else 0.0."""
        if self.trainer is None:
            return 0.0
        optimizers = getattr(self.trainer, "optimizers", None)
        if not optimizers:
            return 0.0
        opt = optimizers[0]
        return float(opt.param_groups[0].get("lr", 0.0))

    def _nt_xent_loss(self, z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        Normalized Temperature-Scaled Cross Entropy Loss (SimCLR style).

        This implementation:
          - normalizes embeddings
          - constructs 2B x 2B similarity logits
          - masks self-similarities (diagonal) by a large negative value to remove them from denominators
          - uses explicit positive index mapping as target for cross-entropy

        Args:
            z1: Tensor [B, D]
            z2: Tensor [B, D]
            temperature: float scaling parameter

        Returns:
            loss: scalar tensor
        """
        # Ensure float32 for numerical stability in mixed precision contexts
        z1 = z1.float()
        z2 = z2.float()
        batch_size = z1.size(0)

        # Normalize embeddings to unit length
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Concatenate to shape [2B, D]
        z = torch.cat([z1, z2], dim=0)

        # Cosine similarity matrix (dot product of normalized vectors)
        logits = torch.matmul(z, z.T) / float(temperature)  # [2B, 2B]

        # For numerical stability avoid using -inf directly with softmax; use large negative
        LARGE_NEG = -1e9
        diag_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(diag_mask, LARGE_NEG)

        # Build targets: for i in [0..B-1], positive is i+B; for i in [B..2B-1], positive is i-B
        targets = torch.arange(batch_size, device=logits.device)
        targets = torch.cat([targets + batch_size, targets], dim=0)  # shape [2B]

        # Cross entropy expects raw logits and integer targets
        loss = F.cross_entropy(logits, targets)
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Training step for one batch.

        Behavior:
          - Logs input stats for the first batch (useful for debugging).
          - Computes embeddings, checks for NaN/Inf in embeddings and loss.
          - Raises RuntimeError on NaN/Inf to fail fast (preferred to silent skipping).

        Args:
            batch: tuple (x1, x2) of tensors
            batch_idx: index of batch

        Returns:
            loss tensor suitable for optimization
        """
        x1, x2 = batch
        if batch_idx == 0:
            self._log_input_stats(x1, "input/x1")
            self._log_input_stats(x2, "input/x2")

        z1 = self.forward(x1)
        z2 = self.forward(x2)

        # Validate outputs (fail-fast)
        if not torch.isfinite(z1).all() or not torch.isfinite(z2).all():
            # log and raise so orchestrators see explicit error
            self.log("debug/z_has_inf_or_nan", 1.0, on_step=True, logger=True)
            raise RuntimeError(f"NaN/Inf detected in model outputs at train batch {batch_idx}")

        loss = self._nt_xent_loss(z1, z2, temperature=self.temperature)

        if not torch.isfinite(loss):
            self.log("debug/loss_is_nan", 1.0, on_step=True, logger=True)
            raise RuntimeError(f"NaN loss at train batch {batch_idx}; lr={self._current_lr():.3e}")

        # Standard logging
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("lr", self._current_lr(), on_step=True, on_epoch=False, logger=True)
        return loss

    def on_after_backward(self) -> None:
        """Log gradient norm after backward pass for debugging/training stability."""
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

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Validation step for one batch.

        Same safety checks as training_step; raises on NaN values.

        Args:
            batch: tuple (x1, x2)
            batch_idx: index

        Returns:
            val_loss tensor (logged)
        """
        x1, x2 = batch
        if batch_idx == 0:
            self._log_input_stats(x1, "val_input/x1")
            self._log_input_stats(x2, "val_input/x2")

        z1 = self.forward(x1)
        z2 = self.forward(x2)

        if not torch.isfinite(z1).all() or not torch.isfinite(z2).all():
            self.log("debug/val_z_has_inf_or_nan", 1.0, on_step=False, on_epoch=True, logger=True)
            raise RuntimeError(f"NaN/Inf detected in val model outputs at batch {batch_idx}")

        val_loss = self._nt_xent_loss(z1, z2, temperature=self.temperature)
        if not torch.isfinite(val_loss):
            self.log("debug/val_loss_is_nan", 1.0, on_step=False, on_epoch=True, logger=True)
            raise RuntimeError(f"NaN val loss at batch {batch_idx}")

        self.log("val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def configure_optimizers(self):
        """Configure optimizer (Adam)."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self._optimizer_ref = optimizer
        return optimizer