"""Storage helpers for large arrays and dataset artifacts.

Provides:
  - save_array_zarr_or_npz(path, arr, metadata=None)
  - load_array_zarr_or_npz(path)
Favors zarr when installed; falls back to npz.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
from typing import Any

def save_array_zarr_or_npz(path: Path, arr: np.ndarray, metadata: dict | None = None) -> None:
    try:
        import zarr  # type: ignore
    except Exception:
        # npz fallback
        path = path.with_suffix(".npz")
        np.savez_compressed(str(path), arr=arr, metadata=metadata or {})
        return
    store = zarr.open_group(str(path), mode="w")
    store.create_dataset("arr", data=arr, compressor=zarr.Blosc(cname="zstd", clevel=3))
    store.attrs["metadata"] = metadata or {}

def load_array_zarr_or_npz(path: Path):
    try:
        import zarr  # type: ignore
        g = zarr.open_group(str(path), mode="r")
        return g["arr"][:], dict(g.attrs)
    except Exception:
        data = np.load(str(path))
        return data["arr"], data.get("metadata", {})
