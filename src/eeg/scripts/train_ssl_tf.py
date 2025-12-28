# src/eeg/scripts/train_ssl_tf.py
"""Training entrypoint for SSL model.

This script builds dataloaders with pinned memory and persistent workers,
handles variable-length EEG sequences via padded collate, configures a
PyTorch Lightning Trainer for mixed-precision and checkpointing, and runs training.

A small "fast" mode (CLI flag --fast or env PYTEST_FAST=1) configures the Trainer
to run a very small quick dev run suitable for CI/tests.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from src.eeg.utils.logger import get_logger
from src.eeg.models.ssl_dataset import SSLDataset, SSLAugmentations, pad_collate_fn, _load_array_lazy
from src.eeg.models.ssl_tf import SSLModelPL

logger = get_logger(__name__)


def make_dataloader(
    sources: List[str],
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    persistent_workers: bool = True,
) -> DataLoader:
    """Create DataLoader for SSLDataset using pad_collate_fn for variable-length signals."""
    ds = SSLDataset(sources, transforms=SSLAugmentations())
    pin_memory = torch.cuda.is_available()
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        drop_last=True,
        collate_fn=pad_collate_fn,
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Build and parse CLI args.

    Accepts an optional argv list (so the function can be called programmatically
    from tests without reading sys.argv).
    """
    p = argparse.ArgumentParser(prog="train_ssl_tf")
    p.add_argument("--data", nargs="+", required=True, help="Glob or list of EEG files (.npy/.npz/.pt/.fif)")
    p.add_argument("--out_dir", type=str, required=True, help="Output run directory for checkpoints/logs")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--proj_dim", type=int, default=128)
    p.add_argument("--encoder_hidden", type=int, default=256)
    p.add_argument("--gpus", type=int, default=0, help="Number of GPUs to use (0 => CPU / MPS)")
    p.add_argument("--precision", type=str, default="16-mixed", help="Precision for Trainer (use '32' for CPU/MPS)")
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--fast",
        action="store_true",
        help="Enable fast-mode for quick CI/test runs (equivalent to PYTEST_FAST=1).",
    )
    return p.parse_args(argv)


def expand_sources(inputs: List[str]) -> List[str]:
    """Expand glob patterns into a flat list of paths."""
    out: List[str] = []
    for item in inputs:
        if any(ch in item for ch in ["*", "?"]):
            out.extend(sorted(glob.glob(item)))
        else:
            out.append(item)
    return out


def main(argv: Optional[List[str]] = None) -> None:
    """Entrypoint for training."""
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    # detect fast mode: explicit flag or env var
    fast_env = os.getenv("PYTEST_FAST", "0") == "1"
    fast_mode = args.fast or fast_env

    # If fast mode requested, ensure a minimum batch size of 2 to avoid BatchNorm errors
    if fast_mode and args.batch_size < 2:
        logger.info("Fast mode requested: bumping batch_size from %d to 2 to avoid BatchNorm issues in training.", args.batch_size)
        args.batch_size = 2

    sources = expand_sources(args.data)
    if len(sources) == 0:
        raise RuntimeError("No data files found for training")

    # Inspect first sample to determine channel count
    first = _load_array_lazy(sources[0])
    if first.ndim >= 2:
        in_ch = first.shape[0]
    else:
        in_ch = 1

    # Build dataloaders
    train_loader = make_dataloader(sources, args.batch_size, args.num_workers, shuffle=True)
    val_loader = make_dataloader(sources, max(1, args.batch_size // 2), args.num_workers, shuffle=False)

    # Initialize model
    model = SSLModelPL(
        encoder_in_channels=in_ch,
        encoder_hidden=args.encoder_hidden,
        proj_dim=args.proj_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Prepare logging and checkpointing
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_logger = TensorBoardLogger(str(out_dir), name="tb")
    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir / "checkpoints"),
        save_top_k=3,
        monitor="val/loss",
        mode="min",
    )
    lrmon = LearningRateMonitor(logging_interval="step")

    # Device configuration
    if torch.cuda.is_available() and args.gpus > 0:
        accelerator = "cuda"
        devices = int(args.gpus)
    elif getattr(torch.backends, "mps", None) is not None and getattr(torch.backends.mps, "is_available", lambda: False)() and args.gpus > 0:
        accelerator = "mps"
        devices = int(args.gpus)
    else:
        accelerator = "cpu"
        devices = 1

    # Trainer precision: use the requested precision only for CUDA; for CPU/MPS use 32-bit
    trainer_precision = args.precision if accelerator == "cuda" else 32

    logger.info("Training config: accelerator=%s devices=%s precision=%s fast_mode=%s", accelerator, devices, trainer_precision, fast_mode)

    # Build trainer kwargs
    trainer_kwargs = dict(
        max_epochs=args.epochs,
        logger=tb_logger,
        callbacks=[ckpt_cb, lrmon],
        precision=trainer_precision,
        accelerator=accelerator,
        devices=devices,
        deterministic=False,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        enable_checkpointing=True,
    )

    # If fast_mode is enabled (tests/CI), throttle training to minimal work
    if fast_mode:
        # Small/lightweight dev-run settings
        trainer_kwargs.update(
            {
                "fast_dev_run": True,
                # limit batches defensively in case Lightning version doesn't support fast_dev_run in some older versions
                "limit_train_batches": 1,
                "limit_val_batches": 1,
                "log_every_n_steps": 1,
            }
        )

    # Instantiate trainer
    trainer = pl.Trainer(**trainer_kwargs)

    logger.info("Starting training (samples=%d, batch_size=%d)", len(sources), args.batch_size)
    # pass resume checkpoint via ckpt_path to fit
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_from)
    logger.info("Training complete. Best checkpoints in %s", str(out_dir / "checkpoints"))

    # In fast/CI mode, Lightning suppresses all logging and checkpoints.
    # Create a tiny sentinel artifact so integration tests can assert output existence.
    if fast_mode:
        sentinel = out_dir / ".fast_run_complete"
        sentinel.write_text("ok\n")
        logger.info("Fast mode: wrote sentinel artifact %s", sentinel)

if __name__ == "__main__":
    main()
