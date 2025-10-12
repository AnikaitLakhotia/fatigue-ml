"""Canonical paths used by the pipeline."""

from __future__ import annotations
from pathlib import Path

ROOT = Path.cwd()
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
FEATURES = DATA / "features"
CONFIGS = ROOT / "configs"
