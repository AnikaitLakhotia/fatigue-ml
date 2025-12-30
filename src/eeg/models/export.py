# src/eeg/models/export.py
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional
import numpy as np

try:
    import torch  # type: ignore
    from torch import nn  # type: ignore
except Exception:  # pragma: no cover - runtime guard
    torch = None
    nn = None  # type: ignore


def export_torchscript(model: "nn.Module", example_input: "torch.Tensor", out_path: str | Path) -> Path:
    """
    Export a PyTorch module to a TorchScript file (traced).
    Returns the path written.

    Requires torch to be installed.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required to export TorchScript models.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Try script first (preferred), fallback to trace
    try:
        scripted = torch.jit.script(model)
    except Exception:
        scripted = torch.jit.trace(model, example_input)
    torch.jit.save(scripted, str(out_path))
    return out_path


def export_onnx(
    model: "nn.Module",
    example_input: "torch.Tensor",
    out_path: str | Path,
    opset_version: int = 14,
) -> Path:
    """
    Export a PyTorch model to ONNX format. May fail depending on ops used.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required to export ONNX models.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ensure model in eval
    model.eval()
    with torch.no_grad():
        try:
            torch.onnx.export(
                model,
                example_input,
                str(out_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            )
        except Exception as exc:
            raise RuntimeError(f"ONNX export failed: {exc}") from exc
    return out_path


def load_torchscript_model(path: str | Path) -> Callable[[np.ndarray], np.ndarray]:
    """
    Load a TorchScript model saved via export_torchscript and return a callable
    that accepts numpy arrays (B, C, T) and returns numpy arrays (B, D).

    Raises RuntimeError if torch is not available or loading fails.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required to load TorchScript models.")

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    # try torch.jit.load (works for saved scripted/traced modules)
    try:
        jmod = torch.jit.load(str(p), map_location="cpu")
    except RuntimeError as exc:
        # try plain torch.load and wrap if it is a state_dict (not supported)
        raise RuntimeError(f"Failed to load TorchScript model: {exc}") from exc

    jmod.eval()

    def forward_numpy(x: np.ndarray) -> np.ndarray:
        """
        x : np.ndarray shaped (B, C, T) or (C, T) -> returns np.ndarray shaped (B, D)
        """
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 2:
            # single sample (C, T) -> (1, C, T)
            arr = arr[None, ...]
        # to torch
        with torch.no_grad():
            t = torch.from_numpy(arr)
            if t.ndim != 3:
                raise ValueError("Input must be ndarray with shape (B, C, T) or (C, T)")
            out = jmod(t)  # assume model returns tensor [B, D]
            if isinstance(out, (tuple, list)):
                out = out[0]
            if not isinstance(out, torch.Tensor):
                # try convert
                out = torch.as_tensor(out)
            res = out.detach().cpu().numpy()
        return res

    return forward_numpy


def load_optional_torchscript(path: Optional[str | Path]) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """
    Convenience: if path is None, return None. If torch missing or load fails, raise.
    """
    if path is None:
        return None
    return load_torchscript_model(path)