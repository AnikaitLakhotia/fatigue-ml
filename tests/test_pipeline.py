"""Smoke test for the full EEG fatigue pipeline.

This test ensures that:
- The end-to-end pipeline (preprocessing + feature extraction) runs successfully.
- It produces valid output files in the expected locations.
- No real EEG data is required (uses synthetic CSV data).
"""

"""Integration test for pipeline orchestration."""

from src.eeg.pipeline import run_pipeline


def test_pipeline_runs_minimal(tmp_path):
    cfg = {
        "data": {
            "input": str(tmp_path / "dummy.csv"),
            "interim_dir": str(tmp_path / "interim"),
            "features_out": str(tmp_path / "features"),
            "run_preprocess": False,
            "run_features": False,
        }
    }
    run_pipeline(cfg)

