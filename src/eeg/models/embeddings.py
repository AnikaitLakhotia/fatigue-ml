# src/eeg/models/embeddings.py
from __future__ import annotations

"""
Session embedding generator.

Reads per-epoch feature parquet files produced by the extractor and aggregates
them into session-level embeddings using descriptive statistics.

Produces one row per session and writes a combined parquet file.

Functions
---------
make_session_embeddings(input_dir, out_path, agg_methods, include_meta)
    Aggregate per-session epoch features into session-level vectors.
"""

from pathlib import Path
from typing import List, Dict, Iterable, Optional
import pandas as pd
import numpy as np

from src.eeg.utils.logger import get_logger

logger = get_logger(__name__)


def _safe_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Return numeric columns in df excluding metadata-like columns."""
    return df.select_dtypes(include="number").columns.tolist()


def _aggregate_df_to_row(
    df: pd.DataFrame, agg_methods: Iterable[str]
) -> Dict[str, float]:
    """
    Aggregate numeric columns of a per-epoch DataFrame to a single row dict.

    Args:
        df: per-epoch DataFrame
        agg_methods: iterable of aggregation method names (e.g. 'mean', 'std', 'median', 'iqr')

    Returns:
        Dictionary mapping "<col>__<agg>" -> value
    """
    out = {}
    num_cols = _safe_numeric_columns(df)
    if not num_cols:
        return out
    # compute aggregations
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


def make_session_embeddings(
    input_dir: str | Path,
    out_path: str | Path,
    agg_methods: Optional[Iterable[str]] = None,
    include_meta: bool = True,
) -> pd.DataFrame:
    """
    Read per-session parquet feature files and create session-level embeddings.

    Args:
        input_dir: directory containing files like `<session>_features.parquet`
        out_path: output parquet path for combined embeddings
        agg_methods: aggregation methods to apply (defaults: mean,std,median,iqr)
        include_meta: if True, attach metadata if present in per-epoch DF
    Returns:
        DataFrame of session embeddings (one row per session) and writes parquet to out_path.
    """
    input_dir = Path(input_dir)
    out_path = Path(out_path)
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
        # attach session id if available
        session_id = None
        if "session_id" in df.columns:
            session_id = df["session_id"].iat[0]
            emb["session_id"] = session_id
        # attach metadata optionally
        if include_meta:
            if "sfreq" in df.columns:
                emb["sfreq"] = float(df["sfreq"].iat[0])
            if "n_channels" in df.columns:
                emb["n_channels"] = int(df["n_channels"].iat[0])
            if "channel_names" in df.columns:
                emb["channel_names"] = df["channel_names"].iat[0]
        emb["source_file"] = p.name
        rows.append(emb)

    out_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path)
    logger.info(
        "Wrote session embeddings to %s (%d sessions × %d dims)",
        out_path,
        out_df.shape[0],
        out_df.shape[1],
    )
    return out_df
