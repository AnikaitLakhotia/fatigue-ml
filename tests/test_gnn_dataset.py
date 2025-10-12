"""Tests for GNN dataset numpy fallback writer.

Verifies that adjacency and node features can be persisted as compressed .npz files.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np

from src.eeg.data.gnn_dataset import save_numpy_graphs


def test_save_numpy_graphs(tmp_path: Path) -> None:
    """Save a trivial identity adjacency with node features and verify .npz file created."""
    adj = np.eye(4)
    node_feat = np.ones((4, 3))
    out_dir = tmp_path / "graphs"
    save_numpy_graphs([adj], [node_feat], out_dir)
    files = list(out_dir.glob("*.npz"))
    assert len(files) == 1
    # load back and check keys
    data = np.load(files[0])
    assert "adj" in data
    assert "node_features" in data
