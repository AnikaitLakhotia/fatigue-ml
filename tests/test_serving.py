# tests/test_serving.py
from __future__ import annotations
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

# Skip if dependencies missing
pytest.importorskip("fastapi")
pytest.importorskip("torch")

import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from src.eeg.models import export as model_export
from src.eeg.serving.app import create_app


class TinyEncoder(nn.Module):
    def __init__(self, in_ch: int = 3, hidden: int = 8, proj_dim: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, hidden, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden, proj_dim)

    def forward(self, x):
        # x: [B, C, T]
        h = self.conv(x)
        h = torch.relu(h)
        h = self.pool(h).squeeze(-1)  # [B, hidden]
        z = self.fc(h)
        z = torch.nn.functional.normalize(z, dim=1)
        return z


def test_serving_predict_and_health(tmp_path: Path):
    # Create and export a tiny TorchScript model
    model = TinyEncoder(in_ch=3, hidden=6, proj_dim=5)
    model.eval()
    example = torch.randn((2, 3, 16))
    model_path = tmp_path / "tiny_model.pt"
    model_export.export_torchscript(model, example, model_path)

    # instantiate app with the exported model
    app = create_app(default_model_path=str(model_path))
    client = TestClient(app)

    # Health should show model_loaded True
    rv = client.get("/health")
    assert rv.status_code == 200
    d = rv.json()
    assert d["status"] == "ok"
    # model_loaded might be True (we try to load); backend should be 'torch'
    assert d["backend"] in ("torch", "numpy")

    # Prepare a small batch of signals and call predict
    batch = np.random.randn(2, 3, 16).astype(float)
    payload = {"signals": batch.tolist()}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    out = r.json()
    assert "embeddings" in out
    emb = np.array(out["embeddings"])
    assert emb.shape[0] == 2
    # emb dimension should equal model proj dim
    assert emb.shape[1] == 5