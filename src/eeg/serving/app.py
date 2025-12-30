# src/eeg/serving/app.py
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
from pathlib import Path
import numpy as np
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.eeg.models import export as model_export

# Types
PredictCallable = Callable[[np.ndarray], np.ndarray]


class PredictRequest(BaseModel):
    # Batch of signals: [batch, channels, time]
    signals: List[List[List[float]]] = Field(..., description="Batch of signals (B x C x T)")
    model_path: Optional[str] = Field(None, description="Optional TorchScript model path to use for this request (server-wide default used otherwise).")


class PredictResponse(BaseModel):
    embeddings: List[List[float]]
    model_loaded: bool
    backend: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    backend: str


def _build_dummy_model_func(proj_dim: int = 16, seed: int = 0) -> PredictCallable:
    """
    Build a deterministic numpy-based dummy model for inference fallback.
    It computes channel-wise mean and applies a fixed random linear projection.
    """
    rng = np.random.default_rng(seed)
    # create a fixed projection matrix: project (channels) -> proj_dim
    # but actual input vector will be (channels,) after mean; we map that via matrix
    def model_fn(x: np.ndarray) -> np.ndarray:
        # x: (B, C, T)
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 3:
            raise ValueError("Input must be shape (B, C, T)")
        # pool over time
        pooled = x.mean(axis=2)  # (B, C)
        # create projection lazily based on dimension
        B, C = pooled.shape
        # deterministically generate projection matrix based on C and proj_dim
        rng2 = np.random.default_rng(seed + C)
        proj = rng2.standard_normal((C, proj_dim)).astype(np.float32) * 0.1
        out = pooled @ proj  # (B, proj_dim)
        # L2 normalize rows
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms = np.where(norms <= 0, 1.0, norms)
        out = out / norms
        return out.astype(np.float32)

    return model_fn


def create_app(default_model_path: Optional[str] = None, default_backend: Optional[str] = None) -> FastAPI:
    """
    Create and return a FastAPI app instance wired to a model.

    - default_model_path: optional path to a TorchScript model to load.
    - default_backend: hint string for health response (e.g. "torch" / "numpy")
    """
    app = FastAPI(title="EEG SSL Serving", version="0.1")

    # Try to load a TorchScript model if provided
    model_callable: Optional[PredictCallable] = None
    backend = "none"
    model_loaded = False
    if default_model_path:
        try:
            model_callable = model_export.load_torchscript_model(default_model_path)
            backend = "torch"
            model_loaded = True
        except Exception:
            # fall back to dummy; still start app but log
            import logging

            logging.getLogger(__name__).exception("Failed to load TorchScript model at %s; falling back to dummy model.", default_model_path)
            model_callable = _build_dummy_model_func()
            backend = default_backend or "numpy"
            model_loaded = False
    else:
        model_callable = _build_dummy_model_func()
        backend = default_backend or "numpy"
        model_loaded = False

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", model_loaded=model_loaded, backend=backend)

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        nonlocal model_callable, backend, model_loaded
        # Allow per-request model override
        if req.model_path:
            try:
                model_callable = model_export.load_torchscript_model(req.model_path)
                backend = "torch"
                model_loaded = True
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Failed to load model: {exc}")

        # convert to numpy
        try:
            arr = np.array(req.signals, dtype=np.float32)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid signals payload; expected nested numeric lists (B x C x T).")

        if arr.ndim == 2:
            # allow single sample shape (C, T)
            arr = arr[None, ...]

        if arr.ndim != 3:
            raise HTTPException(status_code=400, detail=f"signals must be shape (B, C, T); got array with ndim={arr.ndim}")

        try:
            out = model_callable(arr)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}")

        # ensure return is serializable lists
        return PredictResponse(embeddings=out.tolist(), model_loaded=model_loaded, backend=backend)

    # Expose the model callable for tests / introspection (not part of the API)
    app.state._model_callable = model_callable
    app.state._backend = backend
    app.state._model_loaded = model_loaded
    app.state._default_model_path = default_model_path

    return app


# default app for uvicorn: loads model if MODEL_PATH env var is set
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH")
app = create_app(default_model_path=DEFAULT_MODEL_PATH, default_backend="torch" if DEFAULT_MODEL_PATH else "numpy")