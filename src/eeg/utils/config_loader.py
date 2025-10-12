"""Utility to load YAML config files (OmegaConf/Hydra compatible)."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
from omegaconf import OmegaConf

from src.eeg.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load and resolve a YAML configuration file.

    Args:
        path: path to YAML file.

    Returns:
        dict with resolved config.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    cfg = OmegaConf.load(str(p))
    logger.info("Loaded config %s", p)
    return OmegaConf.to_container(cfg, resolve=True)
