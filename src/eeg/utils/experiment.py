# src/eeg/utils/experiment.py
"""MLflow helper utilities with graceful no-op when mlflow is unavailable.

Provides a small context manager wrapper around `mlflow.start_run` so training
scripts can opt-in to MLflow tracking in CI/local runs without hard depending
on mlflow at import time.

API:
    start_run(experiment_name=None, tracking_uri=None, run_name=None, params=None)
        Context manager yielding the mlflow module (or None if mlflow not available).

    safe_log_params(mlflow_module, params)
    safe_log_artifacts(mlflow_module, path, artifact_path=None)
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

__all__ = ["start_run", "safe_log_params", "safe_log_artifacts", "mlflow_available"]


def _import_mlflow():
    try:
        import mlflow  # type: ignore

        return mlflow
    except Exception:
        return None


mlflow_available = _import_mlflow() is not None


@contextmanager
def start_run(
    experiment_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    run_name: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Iterator[Optional[object]]:
    """
    Context manager that starts an MLflow run if `mlflow` is installed.

    Yields:
        mlflow module object if available, else None.
    """
    mlflow = _import_mlflow()
    if mlflow is None:
        # no-op context when mlflow isn't installed
        yield None
        return

    # set tracking uri if provided (e.g. file://tmp/mlruns)
    if tracking_uri:
        try:
            mlflow.set_tracking_uri(str(tracking_uri))
        except Exception:
            # be tolerant of tracking URI issues in CI
            pass

    if experiment_name:
        try:
            mlflow.set_experiment(experiment_name)
        except Exception:
            # ignore errors setting experiment
            pass

    # start run and optionally log params
    with mlflow.start_run(run_name=run_name):
        if params:
            try:
                mlflow.log_params({k: str(v) for k, v in params.items()})
            except Exception:
                # swallow logging errors
                pass
        yield mlflow


def safe_log_params(mlflow_module: Optional[object], params: Dict[str, Any]) -> None:
    """Log params if mlflow_module present."""
    if mlflow_module is None:
        return
    try:
        mlflow_module.log_params({k: str(v) for k, v in params.items()})
    except Exception:
        return


def safe_log_artifacts(mlflow_module: Optional[object], path: str | Path, artifact_path: Optional[str] = None) -> None:
    """Log artifact directory if mlflow_module present."""
    if mlflow_module is None:
        return
    try:
        mlflow_module.log_artifacts(str(path), artifact_path=artifact_path)
    except Exception:
        # some older mlflow versions don't accept artifact_path kwarg; try without
        try:
            mlflow_module.log_artifacts(str(path))
        except Exception:
            return