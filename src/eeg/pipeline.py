"""Pipeline orchestration: CSV → per-session `.fif` → features parquet.

This module glues together the ingestion (split CSV), per-session processing
(existing `process_single_session`), epoching, and feature extraction stages.
It is intentionally explicit and config-driven.
"""

from __future__ import annotations
from typing import Any, Dict
from src.eeg.utils.config_loader import load_config
from src.eeg.utils.logger import get_logger
from src.eeg.scripts.process_sessions import run_preprocess_stage
from src.eeg.scripts.extract_features import run_features_stage

logger = get_logger(__name__)


def run_from_config(cfg_path: str) -> None:
    """Run the full pipeline from a config file path."""
    cfg = load_config(cfg_path)
    run_pipeline(cfg)


def run_pipeline(cfg: Dict[str, Any]) -> None:
    """Run preprocessing and feature extraction stages from a config dict."""
    data_cfg = cfg.get("data", {})
    if data_cfg.get("run_preprocess", True):
        run_preprocess_stage(
            input_path=data_cfg.get("input"),
            output_dir=data_cfg.get("interim_dir", "data/interim"),
            cfg=cfg,
        )
    if data_cfg.get("run_features", True):
        run_features_stage(
            input_dir=data_cfg.get("interim_dir", "data/interim"),
            output_dir=data_cfg.get("features_out", "data/processed/features"),
            cfg=cfg,
        )
    logger.info("Pipeline complete.")

