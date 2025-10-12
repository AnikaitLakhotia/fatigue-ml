"""CLI entrypoint for the full pipeline.

Usage:
    python -m src.eeg.cli --config configs/preprocessing.yaml
"""

from __future__ import annotations
import argparse
from pathlib import Path

from src.eeg.pipeline import run_from_config
from src.eeg.utils.logger import get_logger

logger = get_logger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fatigue-pipeline")
    parser.add_argument("--config", type=str, default="configs/preprocessing.yaml", help="Path to pipeline config YAML")
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    logger.info("Starting pipeline with config %s", config_path)
    run_from_config(config_path)


if __name__ == "__main__":
    main()
