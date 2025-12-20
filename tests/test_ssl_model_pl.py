# tests/test_ssl_model_pl.py
"""Unit tests for SSLModelPL behavior (trainer vs no-trainer, NaN handling)."""

from __future__ import annotations

import pytest

# Skip the tests if torch isn't available in the environment
torch = pytest.importorskip("torch")

import numpy as np
from types import SimpleNamespace

from src.eeg.models.ssl_tf import SSLModelPL


def _make_batch(batch_size=2, n_ch=4, n_t=32, dtype=torch.float32):
    """Create a valid random batch (x1, x2) with shape (B, C, T)."""
    x1 = torch.randn((batch_size, n_ch, n_t), dtype=dtype)
    x2 = torch.randn((batch_size, n_ch, n_t), dtype=dtype)
    return x1, x2


def test_training_step_returns_loss_when_called_directly():
    """Direct call (no Trainer attached) with valid inputs should return a finite loss tensor."""
    model = SSLModelPL(encoder_in_channels=4, encoder_hidden=64, proj_dim=32)
    x1, x2 = _make_batch()
    loss = model.training_step((x1, x2), batch_idx=0)
    assert loss is not None
    assert torch.is_tensor(loss)
    assert torch.isfinite(loss).all()


def test_training_step_raises_on_nan_when_called_directly():
    """Direct call (no Trainer attached) with NaN in inputs should raise RuntimeError per module contract."""
    model = SSLModelPL(encoder_in_channels=4, encoder_hidden=64, proj_dim=32)
    # create NaN batch
    x1 = torch.full((2, 4, 32), float("nan"), dtype=torch.float32)
    x2 = torch.full((2, 4, 32), float("nan"), dtype=torch.float32)
    with pytest.raises(RuntimeError):
        model.training_step((x1, x2), batch_idx=0)


def test_training_step_under_trainer_handles_nan_gracefully(fake_trainer):
    """
    When a Trainer-like object is attached, NaNs should not raise but should return None
    (the module will log / skip the batch under Trainer behavior).
    """
    model = SSLModelPL(encoder_in_channels=4, encoder_hidden=64, proj_dim=32)
    # attach simple stub as _trainer so model._has_trainer() returns True
    fake_opt = SimpleNamespace(param_groups=[{"lr": 1e-4}])
    stub_trainer = SimpleNamespace(optimizers=[fake_opt])
    model._trainer = stub_trainer  # attach stub

    x1 = torch.full((2, 4, 32), float("nan"), dtype=torch.float32)
    x2 = torch.full((2, 4, 32), float("nan"), dtype=torch.float32)

    out = model.training_step((x1, x2), batch_idx=0)
    assert out is None  