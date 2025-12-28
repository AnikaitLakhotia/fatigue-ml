# src/eeg/data/validation.py
"""Lightweight parquet/schema validator used by the pipeline and tests.

Provides:
    - validate_parquet(parquet_path, schema_path, use_great_expectations=True)

The validator accepts a JSON schema that follows a common, simple convention:
{
  "required": ["session_id", "theta_power_mean", ...],
  "properties": {
    "session_id": {"type": "string", "nullable": False},
    "theta_power_mean": {"type": "number", "nullable": False},
    ...
  }
}

This implementation uses a deterministic pandas-based fallback validator and reports
"pandas_fallback" as the engine. Optional Great Expectations integration can be
added later; tests expect either "pandas_fallback" or "great_expectations".
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

# Public API
__all__ = ["validate_parquet"]


def _load_schema(schema_path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(schema_path)
    if not p.exists():
        raise FileNotFoundError(f"Schema file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        schema = json.load(fh)
    return schema


def _is_numeric_dtype(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _is_integer_dtype(series: pd.Series) -> bool:
    return pd.api.types.is_integer_dtype(series)


def _is_string_dtype(series: pd.Series) -> bool:
    return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)


def _is_boolean_dtype(series: pd.Series) -> bool:
    return pd.api.types.is_bool_dtype(series)


def _dtype_matches(series: pd.Series, expected_type: str) -> bool:
    if expected_type in ("number", "float"):
        return _is_numeric_dtype(series)
    if expected_type == "integer":
        # integers are numeric too; accept numeric if all values are integer-valued
        if _is_integer_dtype(series):
            return True
        if _is_numeric_dtype(series):
            vals = series.dropna()
            if vals.empty:
                return True
            arr = pd.to_numeric(vals, errors="coerce").to_numpy()
            if not np.isfinite(arr).all():
                return False
            return np.all(np.equal(np.mod(arr, 1), 0))
        return False
    if expected_type in ("string", "str"):
        return _is_string_dtype(series)
    if expected_type in ("boolean", "bool"):
        return _is_boolean_dtype(series)
    # fallback: accept (object / unknown)
    return True


def validate_parquet(
    parquet_path: Union[str, Path],
    schema_path: Union[str, Path],
    *,
    use_great_expectations: bool = False,
) -> Dict[str, Any]:
    """
    Validate a parquet file against a JSON schema.

    Args:
        parquet_path: path to parquet file to validate (str or Path)
        schema_path: path to JSON schema describing required columns and types
        use_great_expectations: reserved flag for future GE integration (ignored currently)

    Returns:
        dict: {"success": bool, "errors": List[str], "engine": "pandas_fallback"}

    Raises:
        RuntimeError on hard validation failures (missing required columns).
    """
    p = Path(parquet_path)
    if not p.exists():
        raise FileNotFoundError(parquet_path)

    # Load parquet
    try:
        df = pd.read_parquet(p)
    except Exception as exc:
        raise RuntimeError(f"Failed to read parquet {p}: {exc}") from exc

    # Load schema
    schema = _load_schema(schema_path)

    errors: List[str] = []

    # Required columns
    required = schema.get("required", []) or []
    missing = [c for c in required if c not in df.columns]
    if missing:
        # For pipeline gating we raise on missing required columns
        msg = f"Missing required columns: {missing}"
        raise RuntimeError(msg)

    # Properties/type checks
    props = schema.get("properties", {}) or {}
    for col, colschema in props.items():
        if col not in df.columns:
            # if not required and not present, skip type check
            continue
        series = df[col]
        # nullability
        nullable = bool(colschema.get("nullable", True))
        if not nullable:
            # if all values are NaN/None, that's an error
            if series.isna().all():
                errors.append(f"Column '{col}' is non-nullable but contains only nulls")
        # type expectation
        exp_type = colschema.get("type")
        if exp_type:
            try:
                ok = _dtype_matches(series, exp_type)
            except Exception:
                ok = False
            if not ok:
                errors.append(f"Column '{col}' dtype mismatch: expected '{exp_type}', got pandas dtype '{series.dtype}'")

        # optional range checks
        if "minimum" in colschema or "maximum" in colschema:
            if _is_numeric_dtype(series):
                vals = pd.to_numeric(series.dropna(), errors="coerce")
                if not vals.empty:
                    mn = colschema.get("minimum", None)
                    mx = colschema.get("maximum", None)
                    if mn is not None and float(vals.min()) < float(mn):
                        errors.append(f"Column '{col}' min {float(vals.min())} < allowed minimum {mn}")
                    if mx is not None and float(vals.max()) > float(mx):
                        errors.append(f"Column '{col}' max {float(vals.max())} > allowed maximum {mx}")
            else:
                errors.append(f"Column '{col}' has range constraints but is not numeric")

    success = len(errors) == 0
    # report engine as pandas_fallback to satisfy tests that check for either
    # 'pandas_fallback' or 'great_expectations'
    result = {"success": success, "errors": errors, "engine": "pandas_fallback"}
    return result
