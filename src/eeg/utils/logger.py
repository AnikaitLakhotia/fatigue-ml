"""Structured JSON logger used across the pipeline.

Provides a compact JSON formatter for stdout which is easy to ingest into
ELK/Cloud logs or CI logs.
"""

from __future__ import annotations
import json
import logging
import sys
from typing import Any, Dict

LOG_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


class _JsonFormatter(logging.Formatter):
    """Format log records as compact JSON strings."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "name": record.name,
            "level": record.levelname,
            "time": self.formatTime(record, LOG_TIME_FORMAT),
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name: str = "eeg", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that writes compact JSON lines to stdout.

    Args:
        name: Logger name.
        level: Logging level (e.g., logging.INFO).

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
