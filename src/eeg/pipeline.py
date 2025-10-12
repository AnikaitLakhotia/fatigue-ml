"""Top-level pipeline orchestration: CSV -> per-session FIF -> features / exports.

This module glues together config-driven ingestion, preprocessing, epoching,
feature extraction and Tier-3 exports (TF patches, PLV/PLI, graph exports).
"""

from __future__ import annotations
from typing import Any, Dict, Optional
from pathlib import Path

from .utils.config_loader import load_config
from .utils.logger import get_logger
from .scripts.process_sessions import run_preprocess_stage
from .scripts.extract_features import extract_features_from_session

logger = get_logger(__name__)


def run_from_config(cfg_path: str | Path) -> None:
    """
    Load a config file and run the pipeline as specified.

    Args:
        cfg_path: Path to YAML/JSON config describing stages and parameters.

    Returns:
        None
    """
    cfg = load_config(cfg_path)
    run_pipeline(cfg)


def run_pipeline(cfg: Dict[str, Any]) -> None:
    """
    Execute the pipeline stages according to the provided config mapping.

    Args:
        cfg: Configuration dictionary with 'data', 'preprocessing', 'features' keys.

    Returns:
        None
    """
    data_cfg = cfg.get("data", {})
    # Preprocess (CSV -> .fif)
    if data_cfg.get("run_preprocess", True):
        run_preprocess_stage(
            input_path=data_cfg.get("input"),
            output_dir=data_cfg.get("interim_dir", "data/interim"),
            cfg=cfg,
        )
    # Feature extraction (.fif -> features)
    if data_cfg.get("run_features", True):
        run_features = cfg.get("features", {})
        in_dir = data_cfg.get("interim_dir", "data/interim")
        out_dir = data_cfg.get("features_out", "data/processed/features")
        # Here we call existing extract_features script per-file
        from pathlib import Path
        for f in sorted(Path(in_dir).glob("*.fif")):
            extract_features_from_session(f, Path(out_dir), window=run_features.get("window", 10.0), overlap=run_features.get("overlap", 0.5))
    logger.info("Pipeline complete.")
