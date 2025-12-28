# src/eeg/scripts/train_ssl.py
"""
Orchestrator: prepare data (optional CSV -> per-session FIF) and run SSL training.

This script aims to be a thin, safe wrapper around:
  - src.eeg.scripts.process_sessions.process_sessions
  - src.eeg.scripts.train_ssl_tf.main

Usage examples:
  # Use already prepared .fif files (glob or list)
  python -m src.eeg.scripts.train_ssl --data data/interim/*.fif --out_dir runs/ssl/run1 --epochs 5

  # If you have a combined CSV (containing 'session_id' column), convert to .fif first:
  python -m src.eeg.scripts.train_ssl --csv data/raw/combined.csv --interim_dir data/interim --out_dir runs/ssl/run1 --epochs 5

Notes:
  - This wrapper does not attempt to replicate feature extraction. The SSL trainer can work
    directly with .fif / .npy / .npz files as supported by the SSLDataset loader.
  - To keep the wrapper lightweight, the actual training logic is delegated to train_ssl_tf.main.
"""

from __future__ import annotations

import argparse
import glob
import logging
import shutil
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Local imports: reuse existing scripts
from src.eeg.scripts import train_ssl_tf  # type: ignore
from src.eeg.scripts.process_sessions import process_sessions  # type: ignore


def expand_inputs(patterns: List[str]) -> List[str]:
    """Expand globs and return a deduplicated list of paths (strings)."""
    out: List[str] = []
    for p in patterns:
        if any(ch in p for ch in ["*", "?", "["]):
            out.extend(sorted(glob.glob(p)))
        else:
            out.append(p)
    # dedupe while preserving order
    seen = set()
    deduped = []
    for x in out:
        if x not in seen:
            deduped.append(x)
            seen.add(x)
    return deduped


def build_train_argv(
    files: List[str],
    out_dir: str,
    batch_size: int,
    epochs: int,
    num_workers: int,
    lr: float,
    weight_decay: float,
    proj_dim: int,
    encoder_hidden: int,
    gpus: int,
    precision: str,
    resume_from: Optional[str],
    seed: int,
) -> List[str]:
    """Construct argv list for train_ssl_tf.main(...)"""
    argv: List[str] = []
    argv.append("--data")
    argv.extend(files)
    argv.extend(["--out_dir", str(out_dir)])
    argv.extend(["--batch_size", str(batch_size)])
    argv.extend(["--epochs", str(epochs)])
    argv.extend(["--num_workers", str(num_workers)])
    argv.extend(["--lr", str(lr)])
    argv.extend(["--weight_decay", str(weight_decay)])
    argv.extend(["--proj_dim", str(proj_dim)])
    argv.extend(["--encoder_hidden", str(encoder_hidden)])
    argv.extend(["--gpus", str(gpus)])
    argv.extend(["--precision", precision])
    if resume_from:
        argv.extend(["--resume_from", resume_from])
    argv.extend(["--seed", str(seed)])
    return argv


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="train_ssl")
    p.add_argument("--csv", type=str, default=None, help="Optional combined CSV path to convert -> per-session .fif (requires session_id column)")
    p.add_argument("--interim_dir", type=str, default="data/interim", help="Directory to write/read .fif files")
    p.add_argument("--data", nargs="*", help="List / glob of EEG files (.fif/.npy/.npz/.pt) to train on")
    p.add_argument("--out_dir", required=True, help="Output run directory for checkpoints/logs")
    # training/pass-through args
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--proj_dim", type=int, default=128)
    p.add_argument("--encoder_hidden", type=int, default=256)
    p.add_argument("--gpus", type=int, default=0)
    p.add_argument("--precision", type=str, default="16-mixed")
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clean_interim", action="store_true", help="If set and --csv provided, remove interim_dir before writing new .fif files.")
    args = p.parse_args(argv)

    # Prepare interim dir if CSV given
    interim_dir = Path(args.interim_dir)
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        if args.clean_interim and interim_dir.exists():
            logger.info("Cleaning interim dir: %s", interim_dir)
            shutil.rmtree(interim_dir)
        interim_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Converting combined CSV -> per-session .fif files in %s", interim_dir)
        # rely on existing helper; it writes out per-session FIFs
        process_sessions(csv_path, interim_dir, overwrite=True)
        # set data files to interim .fif files
        data_patterns = [str(interim_dir / "*.fif")]
    else:
        data_patterns = args.data or []

    if not data_patterns:
        raise ValueError("No input data provided. Use --csv to convert or provide --data paths/globs.")

    files = expand_inputs(data_patterns)
    if not files:
        raise FileNotFoundError(f"No files found for patterns: {data_patterns}")

    logger.info("Found %d data files; first few: %s", len(files), files[:5])

    # Build argv for the existing training driver
    train_argv = build_train_argv(
        files=files,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        proj_dim=args.proj_dim,
        encoder_hidden=args.encoder_hidden,
        gpus=args.gpus,
        precision=args.precision,
        resume_from=args.resume_from,
        seed=args.seed,
    )

    logger.info("Launching SSL trainer with %d files -> out_dir=%s", len(files), args.out_dir)
    # Delegate to existing driver (it accepts an argv list)
    try:
        train_ssl_tf.main(train_argv)
    except SystemExit as exc:
        # argparse in nested main may call sys.exit; re-raise with context
        logger.exception("Training CLI exited: %s", exc)
        raise
    except Exception:
        logger.exception("SSL training failed")
        raise


if __name__ == "__main__":
    main()
