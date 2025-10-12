"""Configuration loader utilities.

Provides a thin wrapper around OmegaConf to load and resolve YAML/JSON configuration files.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
from omegaconf import OmegaConf
from .logger import get_logger

logger = get_logger(__name__)


def load_config(path: str | Path) -> Dict[str, Any]:
    """
    Load and resolve a configuration file (YAML/JSON).

    Args:
        path: Path to the config file.

    Returns:
        A plain Python dictionary with resolved config values.

    Raises:
        FileNotFoundError: If the path does not exist.
        Exception: If OmegaConf fails to parse.
    """
    p = Path(path)
    if not p.exists():
        logger.error("Config file not found: %s", p)
        raise FileNotFoundError(f"Config not found: {p}")
    cfg = OmegaConf.load(str(p))
    logger.info("Loaded config: %s", p)
    return OmegaConf.to_container(cfg, resolve=True)
