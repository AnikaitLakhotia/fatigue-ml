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

    def __init__(self, in_channels: int = 5, hidden: int = 8):
        """
        Args:
            in_channels: number of EEG channels (default 5)
            hidden: final channel dimension of encoder outputs (should be <= 8)
        """
        super().__init__()
        # Deep temporal feature extractor: progressively extract features, keep time resolution
        self.feature_extractor = nn.Sequential(
            # wide receptive field first layer
            nn.Conv1d(in_channels, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.GELU(),
            # downmix & refine
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            # residual-like refinement
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            # light temporal pooling via stride to increase abstraction (keeps variable length support)
            nn.Conv1d(128, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        # final 1x1 conv reduces channel dimension to `hidden`
        self.bottleneck = nn.Sequential(
            nn.Conv1d(64, int(hidden), kernel_size=1),
            nn.BatchNorm1d(int(hidden)),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, C, T] (padded with zeros for shorter samples)

        Returns:
            Tensor of shape [B, hidden, T] (same time resolution as input)
        """
        # feature extraction (operates on padded sequences correctly)
        h = self.feature_extractor(x)
        h = self.bottleneck(h)
        return h


class SSLProjectionHead(nn.Module):
    """Projection head.

    Input: [B, hidden]
    Output: [B, proj_dim]

    The projection head maps the compact encoder representation into a space
    suited for contrastive learning. We use a small MLP with optional BN and
    final linear projection. The head includes a final L2-normalization step
    to stabilize NT-Xent training.
    """

    def __init__(self, hidden: int = 8, proj_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(hidden), max(32, int(hidden) * 4)),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(max(32, int(hidden) * 4)),
            nn.Linear(max(32, int(hidden) * 4), proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        # encourage stable embeddings for NT-Xent
        z = F.normalize(z, dim=1)
        return z


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
        encoder_in_channels: int = 5,
        encoder_hidden: int = 8,
        proj_dim: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        temperature: float = 0.5,
    ):
        """
        Args:
            encoder_in_channels: number of input EEG channels (default 5)
            encoder_hidden: output channel dimension of encoder (defaults ≤ 8)
            proj_dim: dimensionality of the contrastive projection space
            lr: learning rate
            weight_decay: weight decay for optimizer
            temperature: NT-Xent temperature
        """
        super().__init__()
        # Save hyperparameters for checkpointing/inspection
        self.save_hyperparameters()

        # encoder and projector
        self.encoder = SSLEncoder(in_channels=encoder_in_channels, hidden=encoder_hidden)
        self.projector = SSLProjectionHead(hidden=encoder_hidden, proj_dim=proj_dim)

        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.temperature = float(temperature)

        # optimizer ref (for later inspection)
        self._optimizer_ref = None

    def _masked_global_mean(self, feat: torch.Tensor, ref_x: torch.Tensor) -> torch.Tensor:
        """
        Compute masked global mean over time dimension.

        Args:
            feat: [B, H, T] features from encoder
            ref_x: [B, C, T] original input used to detect padding (assumed padded with zeros)

        Returns:
            pooled: [B, H] mean across valid (non-zero) time points
        """
        # mask timepoints that are all-zero across channels in the original input
        with torch.no_grad():
            # sum absolute across channels -> [B, T]
            valid = (ref_x.abs().sum(dim=1) > 0.0).float()  # 1.0 for valid timesteps
        # expand mask to channels
        mask = valid.unsqueeze(1)  # [B, 1, T]
        denom = mask.sum(dim=-1)  # [B, 1]
        denom = denom.clamp(min=1.0)  # avoid divide-by-zero (samples that are all-zero -> denom=1)
        summed = (feat * mask).sum(dim=-1)  # [B, H]
        pooled = summed / denom  # [B, H]
        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning projection vectors.

        Args:
            x: [B, C, T] (padded with zeros)

        Returns:
            z: [B, proj_dim] normalized projection vectors
        """
        # x -> encoder -> [B, H, T]
        h = self.encoder(x)
        # masked global average pooling to handle variable-length sequences padded with zeros
        h_pool = self._masked_global_mean(h, x)  # [B, H]
        # projector -> [B, proj_dim] (already normalized in projection head)
        z = self.projector(h_pool)
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

    # Loss (NT-Xent)
    @torch.cuda.amp.autocast(enabled=False)
    def _nt_xent_loss(self, z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
        """Normalized Temperature-Scaled Cross Entropy Loss (SimCLR).

        Notes:
            - z1, z2 are expected to be L2-normalized already (projection head normalizes).
            - We build a (2B x 2B) similarity matrix and apply masking for self-similarity.
        """
        z1 = z1.float()
        z2 = z2.float()
        batch_size = z1.size(0)

        # If batch_size is 0 or 1, handle gracefully
        if batch_size < 1:
            return torch.tensor(0.0, device=z1.device)

        # Concatenate and compute similarity
        z = torch.cat([z1, z2], dim=0)  # [2B, D]
        sim = torch.matmul(z, z.T) / temperature  # [2B, 2B]

        # Mask self-similarity to -inf so they do not get selected
        diag_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=sim.device)
        sim = sim.masked_fill(diag_mask, float("-inf"))

        # Labels: for entries i in [0..B-1], positive is i+B; for entries i in [B..2B-1], positive is i-B
        targets = torch.arange(batch_size, device=sim.device)
        targets = torch.cat([targets + batch_size, targets], dim=0)  # [2B]

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
        x1, x2 = batch  # each is [B, C, T]
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
            # safe min/max/mean/std (handle empty tensors gracefully)
            try:
                self._safe_log(f"{prefix}/min", float(xi.min()), on_step=True, on_epoch=False, logger=True)
                self._safe_log(f"{prefix}/max", float(xi.max()), on_step=True, on_epoch=False, logger=True)
                self._safe_log(f"{prefix}/mean", float(xi.mean()), on_step=True, on_epoch=False, logger=True)
                self._safe_log(f"{prefix}/std", float(xi.std()), on_step=True, on_epoch=False, logger=True)
            except Exception:
                # swallow any unexpected logging errors during numeric summarization
                return

    def configure_optimizers(self):
        """Configure optimizer (Adam)."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self._optimizer_ref = optimizer
        return optimizer
