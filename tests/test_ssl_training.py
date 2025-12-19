# tests/test_ssl_training.py
"""Unit tests for SSLModelPL training behaviour and NT-Xent stability."""

from __future__ import annotations

import torch
import numpy as np
import pytest

from src.eeg.models.ssl_tf import SSLModelPL


def test_nt_xent_stability_small_random():
    """NT-Xent should return a finite loss for random embeddings."""
    torch.manual_seed(0)
    batch = 4
    proj_dim = 16
    z1 = torch.randn(batch, proj_dim)
    z2 = torch.randn(batch, proj_dim)
    model = SSLModelPL(encoder_in_channels=1, encoder_hidden=32, proj_dim=proj_dim)
    loss = model._nt_xent_loss(z1, z2, temperature=0.5)
    assert torch.isfinite(loss).all()
    assert float(loss) >= 0.0


def test_training_step_returns_loss_and_raises_on_nan():
    """training_step should return finite loss for valid inputs and raise on NaN embeddings."""
    torch.manual_seed(1)
    batch_size = 2
    channels = 3
    length = 64

    # create small synthetic batch: x1 and x2 of shape [B, C, T]
    x1 = torch.randn(batch_size, channels, length)
    x2 = torch.randn(batch_size, channels, length)

    model = SSLModelPL(encoder_in_channels=channels, encoder_hidden=64, proj_dim=32)

    # valid case: should return a finite tensor loss
    loss = model.training_step((x1, x2), batch_idx=0)
    assert torch.is_tensor(loss)
    assert torch.isfinite(loss).all()

    # inject NaN by monkeypatching forward to return NaN tensors
    def nan_forward(_):
        return torch.full((batch_size, 32), float("nan"))

    model.forward = nan_forward  # type: ignore

    with pytest.raises(RuntimeError):
        _ = model.training_step((x1, x2), batch_idx=1)