"""Normalization helpers for epochs and tabular features."""

from __future__ import annotations
from typing import Optional, Sequence
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from ..utils.logger import get_logger

logger = get_logger(__name__)


def zscore_normalize_epochs(epochs: np.ndarray) -> np.ndarray:
    """
    Z-score each epoch per channel along the time axis.

    Args:
        epochs: ndarray (n_epochs, n_channels, n_samples)

    Returns:
        Normalized epochs with the same shape.

    Raises:
        ValueError: If epochs does not have 3 dimensions.
    """
    if epochs.ndim != 3:
        raise ValueError("epochs must be shape (n_epochs, n_channels, n_samples)")
    mean = epochs.mean(axis=2, keepdims=True)
    std = epochs.std(axis=2, keepdims=True)
    return (epochs - mean) / (std + 1e-8)


def robust_scale_dataframe(df: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Return a copy of df with selected columns scaled with RobustScaler (median/IQR).

    Args:
        df: pandas DataFrame
        columns: Sequence of columns to scale (defaults to all numeric columns)

    Returns:
        A new DataFrame with scaled selected columns.
    """
    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()
    scaler = RobustScaler()
    df_copy = df.copy()
    df_copy.loc[:, columns] = scaler.fit_transform(df_copy.loc[:, columns].values)
    logger.info("Applied robust scaling to %d columns", len(columns))
    return df_copy
