# src/eeg/scripts/extract_features.py
"""
Modern CLI to extract epoch-level features from preprocessed .fif files.

This script supports:
  - mode "single": process one .fif file.
  - mode "many" : process many .fif files in a directory (parallel).
It recovers session_id from raw.info when present (subject_info.his_id or description),
or falls back to the filename stem.

For each input .fif the script:
  - prefers a TIMESTAMP channel to be present; if TIMESTAMP is missing we now synthesize one
    (using sample indices and sfreq) to improve robustness for synthetic test data and
    some legacy FIF files.
  - creates sliding-window epochs (using sliding_window_epochs_from_raw or make_epochs),
  - extracts features via extract_features_from_epochs,
  - writes a parquet named <session_id>_features.parquet into the output directory.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import mne
import numpy as np
import pandas as pd

# epoching helpers: prefer sliding_window_epochs_from_raw but fall back to make_epochs
try:
    from src.eeg.preprocessing.epoching import sliding_window_epochs_from_raw as _sliding_window_epochs  # type: ignore
except Exception:
    _sliding_window_epochs = None  # type: ignore

try:
    from src.eeg.preprocessing.epoching import make_epochs as _make_epochs  # type: ignore
except Exception:
    _make_epochs = None  # type: ignore

from src.eeg.features.extract_features import extract_features_from_epochs  # user-provided extractor

# Optional validation helper (will use GE if available, otherwise pandas fallback)
try:
    from src.eeg.data.validation import validate_parquet  # type: ignore
except Exception:
    validate_parquet = None  # type: ignore

logger = logging.getLogger(__name__)


def _session_id_from_raw(raw: mne.io.BaseRaw) -> Optional[str]:
    """
    Recover a session id from raw.info if present.
    Looks for subject_info.his_id first, then falls back to description containing 'session_id='.
    """
    try:
        si = raw.info.get("subject_info")
        if isinstance(si, dict) and si.get("his_id"):
            return str(si.get("his_id"))
        # some MNE versions use a Bunch-like object
        if hasattr(si, "get") and si.get("his_id"):
            return str(si.get("his_id"))
    except Exception:
        pass
    desc = (raw.info.get("description") or "") if raw is not None else ""
    if "session_id=" in desc:
        try:
            after = desc.split("session_id=", 1)[1]
            token = after.split()[0].strip().strip("',\"")
            return token
        except Exception:
            return None
    return None


def _clean_session_stem(filename: str) -> str:
    """
    Return a cleaned session id from a filename stem.
    Strips common suffixes like '_preprocessed_raw'.
    """
    stem = Path(filename).stem
    for suf in ("_preprocessed_raw", "_preprocessed", "_raw"):
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def _find_fif_files(input_path: Path) -> List[Path]:
    """
    Find .fif files in a given path. If input_path is a file, return it.
    Otherwise search non-recursively for *.fif.
    """
    p = Path(input_path)
    if p.is_file():
        return [p]
    return sorted([x for x in p.glob("*.fif") if x.is_file()])


def _extract_accepted_kwargs(func, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter kwargs to only those accepted by func(signature).
    """
    sig = inspect.signature(func)
    accepted = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            accepted[k] = v
    return accepted


def process_single_fif(
    fif_path: str | Path,
    out_dir: str | Path,
    *,
    window: float = 10.0,
    overlap: float = 0.5,
    per_channel: bool = True,
    save_spectrograms: bool = False,
    save_connectivity: bool = False,
    save_ssl: bool = False,
    backend: str = "numpy",
    device: str = "cpu",
    connectivity_mode: str = "full",
    max_pairs: Optional[int] = None,
    nperseg: int = 256,
    noverlap: int = 128,
    parquet_engine: str = "pyarrow",
    overwrite: bool = True,
    validate: bool = False,
) -> Optional[Path]:
    """
    Process one .fif file: epoch, extract features, write parquet.

    Important: this function will now synthesize a TIMESTAMP channel if one is missing.
    Returns path to written parquet, or None on failure/skip.

    If `validate=True` and a schema validator is available, run validation on the
    written parquet and raise RuntimeError (and remove the parquet) on failure.
    """
    fif_path = Path(fif_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Processing %s", fif_path)
    try:
        raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
    except Exception as exc:
        logger.exception("Failed to read %s: %s", fif_path, exc)
        return None

    # If TIMESTAMP missing, synthesize it (useful for synthetic test files)
    if "TIMESTAMP" not in raw.ch_names:
        try:
            sfreq_candidate = float(raw.info.get("sfreq", 256.0))
            n_samples = raw.n_times
            ts = np.arange(n_samples, dtype=float) / float(sfreq_candidate)
            ts = ts.reshape(1, -1)
            ts_info = mne.create_info(["TIMESTAMP"], sfreq_candidate, ch_types=["misc"])
            ts_raw = mne.io.RawArray(ts, ts_info)
            # add to existing Raw
            raw.add_channels([ts_raw], force_update_info=True)
            logger.warning(
                "TIMESTAMP channel missing in %s; synthesized TIMESTAMP channel with sfreq=%.3f",
                fif_path,
                sfreq_candidate,
            )
        except Exception:
            logger.exception("Failed to synthesize TIMESTAMP for %s; skipping.", fif_path)
            return None

    # Recover session id from raw.info or fallback to filename stem
    recovered_session = _session_id_from_raw(raw) or _clean_session_stem(fif_path.name)

    # produce epochs and meta
    epochs = None
    meta = None
    try:
        if _sliding_window_epochs is not None:
            res = _sliding_window_epochs(raw, window=window, overlap=overlap)
        elif _make_epochs is not None:
            res = _make_epochs(raw, window=window, overlap=overlap)
        else:
            raise RuntimeError(
                "No epoching helper available (sliding_window_epochs_from_raw or make_epochs required)"
            )
        # res may be epochs or (epochs, meta)
        if isinstance(res, tuple) and len(res) >= 1:
            epochs = res[0]
            meta = res[1] if len(res) > 1 else None
        else:
            epochs = res
    except Exception as exc:
        logger.exception("Epoching failed for %s: %s", fif_path, exc)
        return None

    if epochs is None or getattr(epochs, "ndim", 0) != 3 or epochs.shape[0] == 0:
        logger.warning("No epochs for %s; skipping", fif_path)
        return None

    # If epoch meta missing or shorter than number of epochs, synthesize lightweight meta
    if not meta or len(meta) < epochs.shape[0]:
        sfreq = float(raw.info.get("sfreq", 256.0))
        chnames = [c for c in raw.ch_names if c != "TIMESTAMP"]
        base_meta = []
        n_epochs = epochs.shape[0]
        for i in range(n_epochs):
            base_meta.append(
                {
                    "epoch_index": i,
                    "start_ts": float(i * (epochs.shape[2] / sfreq)),
                    "end_ts": float((i + 1) * (epochs.shape[2] / sfreq)),
                    "center_ts": float(
                        ((i * (epochs.shape[2] / sfreq)) + ((i + 1) * (epochs.shape[2] / sfreq))) / 2.0
                    ),
                    "session_id": recovered_session,
                    "sfreq": sfreq,
                    "n_channels": epochs.shape[1],
                    "channel_names": chnames,
                }
            )
        meta = base_meta

    # ensure session_id present in each meta entry
    for m in meta:
        if not m.get("session_id"):
            m["session_id"] = recovered_session

    # Build kwargs to forward to extractor, using introspection
    extractor_kwargs = dict(
        sfreq=float(raw.info.get("sfreq", 256.0)),
        per_channel=per_channel,
        enabled=None,
        save_spectrograms=save_spectrograms,
        save_connectivity=save_connectivity,
        save_ssl=save_ssl,
        backend=backend,
        device=device,
        connectivity_mode=connectivity_mode,
        max_pairs=max_pairs,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    accepted = _extract_accepted_kwargs(extract_features_from_epochs, extractor_kwargs)
    # call extractor
    try:
        df = extract_features_from_epochs(epochs, epoch_meta=meta, **accepted)
    except Exception:
        logger.exception("Feature extraction failed for %s", fif_path)
        return None

    # ensure DataFrame and session_id column filled
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            logger.error("Extractor returned non-dataframe and could not convert for %s", fif_path)
            return None

    sid = recovered_session
    if "session_id" not in df.columns:
        df["session_id"] = sid
    else:
        df["session_id"] = df["session_id"].fillna(sid)

    # write parquet
    out_name = f"{sid}_features.parquet"
    out_path = out_dir / out_name
    if out_path.exists() and not overwrite:
        logger.info("Skipping existing %s", out_path)
        return out_path
    try:
        df.to_parquet(out_path, engine=parquet_engine, index=False)
        logger.info("Wrote features %s (n_epochs=%d)", out_path, len(df))
    except Exception:
        logger.exception("Failed writing parquet for %s", fif_path)
        return None

    # optional validation/gating
    if validate:
        if validate_parquet is None:
            logger.warning("Validation requested but validator not available (install great_expectations or ensure src.eeg.data.validation is importable). Skipping validation.")
        else:
            # pick epoch schema (repo-relative schemas/epoch_features.schema.json)
            schema_file = Path(__file__).resolve().parents[2] / "schemas" / "epoch_features.schema.json"
            schema_path = str(schema_file)
            try:
                logger.info("Running schema validation on %s using %s", out_path, schema_path)
                res = validate_parquet(out_path, schema_path=schema_path, use_great_expectations=True)
                logger.info("Validation result for %s: success=%s engine=%s", out_path, res.get("success", False), res.get("engine"))
            except Exception:
                # On validation failure: remove invalid artifact (best-effort) and raise to gate pipeline
                logger.exception("Validation failed for %s; removing output parquet and raising", out_path)
                try:
                    # Python 3.8+ supports missing_ok, but use safe fallback for older versions
                    out_path.unlink(missing_ok=True)  # type: ignore
                except TypeError:
                    try:
                        if out_path.exists():
                            out_path.unlink()
                    except Exception:
                        logger.exception("Failed to remove invalid output %s", out_path)
                raise

    return out_path


def _process_many(
    input_path: Path,
    out_dir: Path,
    *,
    n_jobs: Optional[int] = None,
    backend: str = "numpy",
    device: str = "cpu",
    **kwargs,
) -> List[Path]:
    files = _find_fif_files(input_path)
    if not files:
        logger.warning("No .fif files found in %s", input_path)
        return []

    n_jobs = n_jobs or (os.cpu_count() or 1)
    n_jobs = max(1, int(n_jobs))
    logger.info(
        "Processing %d files with %d workers (backend=%s device=%s)", len(files), n_jobs, backend, device
    )

    worker = partial(process_single_fif, out_dir=out_dir, backend=backend, device=device, **kwargs)

    results: List[Path] = []
    if n_jobs == 1:
        for f in files:
            try:
                res = worker(f)
                if res:
                    results.append(res)
            except Exception:
                logger.exception("Failed %s", f)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            future_to_file = {ex.submit(worker, f): f for f in files}
            for fut in as_completed(future_to_file):
                f = future_to_file[fut]
                try:
                    res = fut.result()
                    if res:
                        results.append(res)
                        logger.info("Completed %s", f)
                except Exception:
                    logger.exception("Worker failed for %s", f)
    return results


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract features from .fif files to parquet (modern CLI).")
    p.add_argument("mode", choices=("single", "many"), help="single file or many")
    p.add_argument("--input", "-i", required=True, help="Input .fif file or directory")
    p.add_argument("--out", "-o", required=True, help="Output directory for parquet files")
    p.add_argument("--window", type=float, default=10.0)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--per-channel", action="store_true")
    p.add_argument("--save-spectrograms", action="store_true")
    p.add_argument("--save-connectivity", action="store_true")
    p.add_argument("--save-ssl", action="store_true")
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument("--parquet-engine", default="pyarrow")
    p.add_argument("--no-overwrite", action="store_true")
    p.add_argument("--backend", choices=("numpy", "torch"), default="numpy")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--connectivity-mode", choices=("full", "minimal", "none"), default="full")
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--nperseg", type=int, default=256)
    p.add_argument("--noverlap", type=int, default=128)
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument(
        "--validate",
        action="store_true",
        help="Run schema validation (using schemas/epoch_features.schema.json) on each written parquet and fail if invalid",
    )
    return p


def main():
    args = _build_parser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    backend = args.backend
    if backend == "torch":
        try:
            import torch  # noqa: F401
            if args.device == "cuda" and not torch.cuda.is_available():
                logger.warning("Requested cuda device but torch.cuda not available. Falling back to CPU.")
        except Exception:
            raise RuntimeError("torch backend requested but torch is not installed")

    common_kwargs = dict(
        window=args.window,
        overlap=args.overlap,
        per_channel=args.per_channel,
        save_spectrograms=args.save_spectrograms,
        save_connectivity=args.save_connectivity,
        save_ssl=args.save_ssl,
        nperseg=args.nperseg,
        noverlap=args.noverlap,
        connectivity_mode=args.connectivity_mode,
        max_pairs=args.max_pairs,
        parquet_engine=args.parquet_engine,
        overwrite=not args.no_overwrite,
        validate=args.validate,
    )

    if args.mode == "single":
        process_single_fif(
            Path(args.input),
            Path(args.out),
            backend=backend,
            device=args.device,
            **common_kwargs,
        )
    else:
        _process_many(
            Path(args.input),
            Path(args.out),
            n_jobs=args.n_jobs,
            backend=backend,
            device=args.device,
            **common_kwargs,
        )


if __name__ == "__main__":
    main()
