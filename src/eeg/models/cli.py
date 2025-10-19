# src/eeg/models/cli.py
from __future__ import annotations

"""
CLI to run modeling workflows (session embeddings, unsupervised pipeline, optional AE training).
"""

import argparse
from pathlib import Path
from src.eeg.models.embeddings import make_session_embeddings
from src.eeg.models.unsupervised import run_unsupervised_pipeline
from src.eeg.models.autoencoder import TrainAutoencoderConfig, train_autoencoder
from src.eeg.utils.logger import get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="eeg-modeling")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_emb = sub.add_parser(
        "embeddings", help="Create session embeddings from per-epoch features"
    )
    p_emb.add_argument(
        "--in", dest="input_dir", required=True, help="Features folder (parquet files)"
    )
    p_emb.add_argument(
        "--out", dest="out_path", required=True, help="Output parquet path"
    )
    p_emb.add_argument(
        "--aggs",
        dest="aggs",
        default="mean,std,median,iqr",
        help="Comma-separated aggregation methods",
    )

    p_uns = sub.add_parser(
        "unsupervised", help="Run unsupervised pipeline on a features parquet"
    )
    p_uns.add_argument(
        "--features", required=True, help="Input features parquet (session or epoch)"
    )
    p_uns.add_argument("--out", required=True, help="Output dir")
    p_uns.add_argument("--pca", type=int, default=10)
    p_uns.add_argument("--umap", type=int, default=2)
    p_uns.add_argument(
        "--cluster", choices=["hdbscan", "gmm", "kmeans"], default="hdbscan"
    )

    p_ae = sub.add_parser(
        "train-ae", help="Train autoencoder (optional, requires torch)"
    )
    p_ae.add_argument(
        "--features", required=True, help="Input features parquet for AE training"
    )
    p_ae.add_argument("--out", required=True, help="Output dir for AE")
    p_ae.add_argument("--epochs", type=int, default=20)
    p_ae.add_argument("--bs", type=int, default=64)
    p_ae.add_argument("--lr", type=float, default=1e-3)
    p_ae.add_argument("--latent", type=int, default=16)

    args = parser.parse_args(argv)

    if args.cmd == "embeddings":
        aggs = [a.strip() for a in args.aggs.split(",") if a.strip()]
        make_session_embeddings(args.input_dir, args.out_path, agg_methods=aggs)
    elif args.cmd == "unsupervised":
        run_unsupervised_pipeline(
            args.features,
            args.out,
            pca_components=args.pca,
            umap_components=args.umap,
            cluster_method=args.cluster,
        )
    elif args.cmd == "train-ae":
        cfg = TrainAutoencoderConfig(
            batch_size=args.bs, epochs=args.epochs, lr=args.lr, latent_dim=args.latent
        )
        df = __import__("pandas").read_parquet(args.features)
        train_autoencoder(df, cfg, args.out)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
