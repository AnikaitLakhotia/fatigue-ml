# tests/conftest.py
"""Pytest fixtures for deterministic tests and lightweight Trainer stubs."""

from __future__ import annotations

import os
import random
from types import SimpleNamespace

import numpy as np
import pytest

# Try to import torch but do not fail if missing (tests that depend on torch will use importorskip)
try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore


@pytest.fixture(autouse=True, scope="session")
def deterministic_test_env():
    """
    Make tests deterministic and reasonably fast:
      - set PYTHONHASHSEED
      - seed python, numpy (and torch if available)
      - cap CPU threads for torch if present
    """
    seed = int(os.environ.get("PYTEST_SEED", "42"))
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        try:
            # limit intra-op threads for deterministic speed
            torch.set_num_threads(1)
        except Exception:
            pass
    yield


@pytest.fixture
def fake_trainer():
    """
    Return a very small trainer-like stub useful to attach to LightningModules for unit tests.

    The object exposes:
      - optimizers: list containing a simple param-group-like object with 'param_groups'
    """
    # param_groups mimic optimizer.param_groups list structure for _current_lr lookup
    fake_opt = SimpleNamespace(param_groups=[{"lr": 1e-3}])
    # trainer-like stub with optimizers attribute
    trainer = SimpleNamespace(optimizers=[fake_opt])
    return trainer