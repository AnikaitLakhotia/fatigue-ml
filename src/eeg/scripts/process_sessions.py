"""CLI entrypoint: process combined CSV into per-session preprocessed FIF files.

Usage:
    python -m eeg.scripts.process_sessions --input data/raw/combined_dataset.csv --out data/interim
"""

from __future__ import annotations
from pathlib import Path
import argparse
from src.eeg.data.io import process_all_sessions_entrypoint
from src.eeg.utils.logger import get_logger

logger = get_logger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Process combined dataset into per-session preprocessed files.")
    parser.add_argument("--input", type=str, default="data/raw/combined_dataset.csv", help="Path to combined CSV")
    parser.add_argument("--out", type=str, default="data/interim", help="Directory for per-session .fif files")
    parser.add_argument("--tmp", type=str, default=None, help="Optional temporary directory for shards")
    args = parser.parse_args(argv)

    input_p = Path(args.input)
    out_p = Path(args.out)
    tmp_p = Path(args.tmp) if args.tmp else None

    logger.info("Starting processing: input=%s out=%s", input_p, out_p)
    generated = process_all_sessions_entrypoint(input_p, out_p, tmp_p)
    logger.info("Processing complete: generated %d files", len(generated))


if __name__ == "__main__":
    main()
