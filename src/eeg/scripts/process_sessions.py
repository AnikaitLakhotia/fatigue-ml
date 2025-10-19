"""CLI helper to split combined CSV and produce per-session .fif preprocessed files."""

from __future__ import annotations
from pathlib import Path
import argparse
from typing import List, Optional

from ..utils.logger import get_logger
from ..data.io import split_combined_csv_by_session, process_single_session

logger = get_logger(__name__)


def run_preprocess_stage(
    input_path: str | Path, output_dir: str | Path, cfg: Optional[dict] = None
) -> List[Path]:
    """
    Split combined CSV and process each session CSV to a preprocessed .fif.

    Args:
        input_path: path to combined CSV
        output_dir: destination for .fif files
        cfg: optional config for filter parameters (ignored in CLI default)

    Returns:
        List of Path objects pointing to produced .fif files.
    """
    input_path = Path(input_path)
    per_session = split_combined_csv_by_session(input_path, Path("/tmp"))
    out_list: List[Path] = []
    for p in per_session:
        sid = p.stem
        out_fif = Path(output_dir) / f"{sid}_preprocessed.fif"
        logger.info("Processing %s -> %s", p.name, out_fif.name)
        if cfg:
            # For backwards compatibility: cfg-driven options could be propagated here.
            process_single_session(p, out_fif)
        else:
            process_single_session(p, out_fif)
        out_list.append(out_fif)
    logger.info("Preprocessing complete: %d sessions", len(out_list))
    return out_list


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="process_sessions")
    parser.add_argument(
        "--input", type=str, required=True, help="Combined CSV input path"
    )
    parser.add_argument(
        "--out", type=str, required=True, help="Output interim directory for FIFs"
    )
    args = parser.parse_args(argv)
    run_preprocess_stage(args.input, args.out, cfg=None)


if __name__ == "__main__":
    main()
