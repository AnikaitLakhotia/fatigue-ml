"""Centralized logger factory used across the pipeline.

Provides a single function `get_logger` that returns a configured `logging.Logger`.
All modules should import and use this logger factory to keep messages consistent.
"""

from __future__ import annotations
import logging
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Create or retrieve a module-scoped logger.

    Args:
        name: Optional logger name (defaults to module-level 'eeg').

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name or "eeg")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
