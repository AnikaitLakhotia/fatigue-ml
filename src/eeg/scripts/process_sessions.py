"""CLI and small API to split the combined CSV and process each session to .fif.

Exports:
  - run_preprocess_stage(input_path, output_dir, cfg)
  - main() CLI entrypoint
"""

from __future__ import annotations
from pathlib import Path
import argparse
from typing import List, Optional

from src.eeg.utils.logger import get_logger
from src.eeg.data.io import split_combined_csv_by_session, process_single_session

logger = get_logger(__name__)


def run_preprocess_stage(input_path: str | Path, output_dir: str | Path, cfg: Optional[dict] = None) -> List[Path]:
    """Split combined CSV and process each session to a preprocessed .fif.

    Args:
        input_path: path to combined CSV
        output_dir: destination folder for produced .fif files
        cfg: optional config dict (can contain filtering/ICA parameters)

    Returns:
        list of produced .fif Path objects
    """
    input_path = Path(input_path)
    tmp_dir = Path("/tmp")
    per_session = split_combined_csv_by_session(input_path, tmp_dir)
    out_list: List[Path] = []
    for p in per_session:
        sid = p.stem.replace("session_raw_", "")
        out_fif = Path(output_dir) / f"{sid}_preprocessed.fif"
        logger.info("Processing %s -> %s", p.name, out_fif.name)
        # propagate config if present
        if cfg:
            pf = cfg.get("preprocessing", {})
            l_freq = pf.get("filter", {}).get("low_cut", 0.5)
            h_freq = pf.get("filter", {}).get("high_cut", 45.0)
            notch = pf.get("filter", {}).get("notch", [50.0])
            reference = pf.get("reference", {}).get("type", "average")
            ica_method = pf.get("ica", {}).get("method", "fastica")
            process_single_session(p, out_fif, l_freq=l_freq, h_freq=h_freq, notch_freqs=notch, reference=reference, ica_method=ica_method)
        else:
            process_single_session(p, out_fif)
        out_list.append(out_fif)
    logger.info("Preprocessing complete: %d sessions", len(out_list))
    return out_list


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="process_sessions")
    parser.add_argument("--input", type=str, default="data/raw/combined_dataset.csv", help="Combined CSV input path")
    parser.add_argument("--out", type=str, default="data/interim", help="Output interim directory for FIFs")
    parser.add_argument("--config", type=str, default=None, help="Optional config YAML (not used in CLI mode)")
    args = parser.parse_args(argv)
    run_preprocess_stage(args.input, args.out, cfg=None)


if __name__ == "__main__":
    main()
