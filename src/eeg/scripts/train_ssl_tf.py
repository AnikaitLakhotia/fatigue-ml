# src/eeg/scripts/train_ssl_tf.py
from __future__ import annotations
"""
CLI script to train contrastive SSL (TensorFlow) on preprocessed EEG epochs.
"""
import argparse
from pathlib import Path
import tensorflow as tf
from src.eeg.models.ssl_dataset import make_contrastive_dataset
from src.eeg.models.ssl_tf import TemporalConvEncoderTF, ProjectionHeadTF, ContrastiveTrainerTF
from src.eeg.utils.logger import get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="train-ssl-tf")
    parser.add_argument("--epochs", required=True, help="Path to epochs .npy/.npz or directory")
    parser.add_argument("--out", required=True, help="Directory to save weights")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--epochs_num", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-6)
    parser.add_argument("--enc-dim", type=int, default=128)
    parser.add_argument("--proj-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("TensorFlow version: %s", tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    logger.info("GPUs found: %s", gpus)

    ds = make_contrastive_dataset(epochs=args.epochs, batch_size=args.batch, shuffle=True, seed=args.seed)

    for v1, v2 in ds.take(1):
        ch, samp = int(v1.shape[1]), int(v1.shape[2])
        break

    encoder = TemporalConvEncoderTF(n_channels=ch, n_samples=samp, enc_dim=args.enc_dim)
    projector = ProjectionHeadTF(in_dim=args.enc_dim, proj_dim=args.proj_dim)

    trainer = ContrastiveTrainerTF(
        encoder=encoder,
        projector=projector,
        learning_rate=args.lr,
        weight_decay=args.wd,
        save_dir=out_dir,
    )

    logger.info("Starting training: epochs=%d batch=%d lr=%.1e", args.epochs_num, args.batch, args.lr)
    res = trainer.train(dataset=ds, epochs=args.epochs_num, save_every_n_epochs=1)
    logger.info("Training complete. Results: %s", res)


if __name__ == "__main__":
    main()