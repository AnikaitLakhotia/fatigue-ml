"""Canonical filesystem paths used by the EEG pipeline.

Keep this file small and deterministic — it exposes useful project-level
constants that the pipeline and tests can import if desired.
"""

from __future__ import annotations
from pathlib import Path

ROOT = Path.cwd()
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
FEATURES = DATA / "features"
CONFIGS = ROOT / "configs"
