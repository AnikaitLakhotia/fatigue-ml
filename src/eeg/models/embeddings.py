# src/eeg/models/embeddings.py
from __future__ import annotations

"""
Session embedding generator.

Reads per-epoch feature parquet files produced by the extractor and aggregates
them into session-level embeddings using descriptive statistics.

Produces one row per session and writes a combined parquet file, as well as
per-session `.npz` embedding files for SSL-style downstream models.

Functions:
    make_session_embeddings(input_dir, out_path, agg_methods, include_meta)
        Aggregate per-session epoch features into session-level vectors and save both
        parquet and npz versions.
"""

from pathlib import Path
from typing import List, Dict, Iterable, Optional
import pandas as pd
import numpy as np

from src.eeg.utils.logger import get_logger

logger = get_logger(__name__)


# Helpers
def _safe_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Return numeric columns in df excluding metadata-like columns."""
    meta_cols = {"session_id", "sfreq", "n_channels", "channel_names"}
    cols = df.select_dtypes(include="number").columns
    return [c for c in cols if c not in meta_cols]


def _aggregate_df_to_row(df: pd.DataFrame, agg_methods: Iterable[str]) -> Dict[str, float]:
    """
    Aggregate numeric columns of a per-epoch DataFrame to a single row dict.

    Args:
        df: Per-epoch DataFrame.
        agg_methods: Iterable of aggregation method names (e.g., 'mean', 'std', 'median', 'iqr').

    Returns:
        Dict[str, float]: Mapping of "<col>__<agg>" → value.
    """
    out = {}
    num_cols = _safe_numeric_columns(df)
    if not num_cols:
        return out

    for col in num_cols:
        vals = df[col].values
        if "mean" in agg_methods:
            out[f"{col}__mean"] = float(np.nanmean(vals))
        if "std" in agg_methods:
            out[f"{col}__std"] = float(np.nanstd(vals))
        if "median" in agg_methods:
            out[f"{col}__median"] = float(np.nanmedian(vals))
        if "min" in agg_methods:
            out[f"{col}__min"] = float(np.nanmin(vals))
        if "max" in agg_methods:
            out[f"{col}__max"] = float(np.nanmax(vals))
        if "iqr" in agg_methods:
            q75, q25 = np.nanpercentile(vals, [75, 25]) if len(vals) else (0.0, 0.0)
            out[f"{col}__iqr"] = float(q75 - q25)
    return out


# Main function
def make_session_embeddings(
    input_dir: str | Path,
    out_path: str | Path,
    agg_methods: Optional[Iterable[str]] = None,
    include_meta: bool = True,
) -> pd.DataFrame:
    """
    Read per-session parquet feature files and create session-level embeddings.

    For each session parquet file:
        - Aggregate numeric features with descriptive statistics.
        - Save the aggregated session embedding as a row in a combined parquet file.
        - Save a separate `.npz` file (for SSL tasks) containing the numeric embedding vector.

    Args:
        input_dir (str | Path): Directory containing files like `<session>_features.parquet`.
        out_path (str | Path): Output parquet path for combined embeddings.
        agg_methods (Iterable[str], optional): Aggregation methods (default: mean, std, median, iqr).
        include_meta (bool): If True, attach metadata if present in per-epoch DataFrame.

    Returns:
        pd.DataFrame: Combined session embeddings (one row per session).
    """
    input_dir = Path(input_dir)
    out_path = Path(out_path)
    ssl_dir = Path("data/ssl")
    ssl_dir.mkdir(parents=True, exist_ok=True)

    agg_methods = list(agg_methods or ("mean", "std", "median", "iqr"))
    files = sorted(input_dir.glob("*_features.parquet"))
    if not files:
        raise FileNotFoundError(f"No feature parquet files found in {input_dir}")

    rows = []
    for p in files:
        logger.info("Processing features for session: %s", p.name)
        df = pd.read_parquet(p)
        emb = _aggregate_df_to_row(df, agg_methods)
        if not emb:
            logger.warning("No numeric features in %s — skipping", p.name)
            continue

        # session metadata
        session_id = None
        if "session_id" in df.columns:
            session_id = str(df["session_id"].iat[0])
            emb["session_id"] = session_id
        else:
            session_id = p.stem.replace("_features", "")

        if include_meta:
            if "sfreq" in df.columns:
                emb["sfreq"] = float(df["sfreq"].iat[0])
            if "n_channels" in df.columns:
                emb["n_channels"] = int(df["n_channels"].iat[0])
            if "channel_names" in df.columns:
                emb["channel_names"] = df["channel_names"].iat[0]

        emb["source_file"] = p.name

        # --- SSL .npz embedding ---
        numeric_values = np.array(
            [v for k, v in emb.items() if isinstance(v, (int, float))],
            dtype=np.float32,
        )
        npz_path = ssl_dir / f"{session_id}_embedding.npz"
        np.savez_compressed(npz_path, embedding=numeric_values)
        logger.debug("Saved SSL embedding: %s (shape=%s)", npz_path.name, numeric_values.shape)

        rows.append(emb)

    # Write combined parquet
    out_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    logger.info(
        "Wrote session embeddings to %s (%d sessions × %d dims)",
        out_path,
        out_df.shape[0],
        out_df.shape[1],
    )
    logger.info("Wrote per-session SSL embeddings to %s", ssl_dir)

    return out_df
