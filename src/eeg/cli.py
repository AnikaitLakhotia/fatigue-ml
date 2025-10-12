"""
CLI entrypoint for the EEG pipeline.

Provides two thin commands:
  - pipeline: run the full pipeline from a YAML config (uses run_from_config)
  - extract: run the feature extraction CLI (wrapped around scripts.extract_features_full.main)

Usage:
  python -m src.eeg.cli pipeline --config configs/preprocessing.yaml
  python -m src.eeg.cli extract --input data/interim --out data/features --window 10 --overlap 0.5
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Optional
from src.eeg.pipeline import run_from_config
from src.eeg.scripts.extract_features_full import main as extract_main
from src.eeg.utils.logger import get_logger

logger = get_logger(__name__)


def cli(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="fatigue-pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pipeline = sub.add_parser("pipeline", help="Run pipeline from config YAML")
    p_pipeline.add_argument("--config", type=str, default="configs/preprocessing.yaml", help="Path to pipeline config YAML")

    p_extract = sub.add_parser("extract", help="Run full feature extractor (wraps extract_features_full)")
    p_extract.add_argument("--input", required=True, help="Input folder with preprocessed .fif files")
    p_extract.add_argument("--out", required=True, help="Output folder for feature files")
    p_extract.add_argument("--window", type=float, default=10.0, help="Window length (s)")
    p_extract.add_argument("--overlap", type=float, default=0.5, help="Window overlap fraction")
    p_extract.add_argument("--per-channel", action="store_true", help="Include per-channel features")
    p_extract.add_argument("--features", type=str, default=None, help="Comma-separated list of feature names to enable (defaults to all)")
    p_extract.add_argument("--save-spectrograms", action="store_true", help="Save spectrogram tensors if available")
    p_extract.add_argument("--save-connectivity", action="store_true", help="Save connectivity matrices if available")

    args = parser.parse_args(argv)
    if args.cmd == "pipeline":
        cfg_path = Path(args.config)
        logger.info("Starting pipeline with config %s", cfg_path)
        run_from_config(str(cfg_path))
    elif args.cmd == "extract":
        # build argv for extract_features_full.main
        extract_argv = [
            "--input", str(args.input),
            "--out", str(args.out),
            "--window", str(args.window),
            "--overlap", str(args.overlap),
        ]
        if args.per_channel:
            extract_argv.append("--per-channel")
        if args.features:
            extract_argv.extend(["--features", args.features])
        if args.save_spectrograms:
            extract_argv.append("--save-spectrograms")
        if args.save_connectivity:
            extract_argv.append("--save-connectivity")
        extract_main(extract_argv)
    else:
        parser.print_help()


if __name__ == "__main__":
    cli()
