# Fatigue-ML

## Overview
This repository contains a reproducible pipeline for EEG-based **fatigue** analysis.  
It covers ingestion, preprocessing, epoching, feature extraction and export of feature tables and sidecars for downstream modeling and visualization.

> **⚠️ Data policy:** Raw EEG data and derived artifacts (MNE `.fif`, `.parquet`, model weights, etc.) **must not** be committed to Git. See *Data & Git Policy* below.

### Highlights / what this repo does
- Robust session-aware ingestion and preprocessing (bandpass + notch filtering, re-referencing, ICA artifact removal).
- Per-session intermediate files (MNE `.fif`) for reproducibility and manual review.
- Comprehensive feature extraction:
  - Welch PSD (robust wrapper with fallbacks)
  - Absolute & relative bandpowers (δ/θ/α/β/γ) per-channel and aggregated
  - Peak Alpha Frequency (PAF)
  - Spectral entropy (normalized Shannon)
  - Robust 1/f slope estimate (`one_over_f_slope`)
  - Nonlinear features: permutation entropy, Higuchi FD, sample entropy (SampEn)
  - Pairwise coherence (mean per canonical band)
  - Optional per-channel flattened outputs and aggregate statistics (mean/std)
- Optional sidecars:
  - Per-epoch spectrograms
  - Sliding-window connectivity matrices
- Built-in QC and defensive logic:
  - Flat/broken channel detection + interpolation (if montage available) or drop.
  - PSD floor (EPS) and NaN/Inf sanitization to avoid divide-by-zero issues.
  - QC columns in outputs: `num_channels_with_zero_total`, `has_any_zero_total`.
- CLI scripts and programmatic API for batch processing.
- Tests and small synthetic fixtures for CI.

---

## Repo structure
```
fatigue-ml/
├── data/                 # local data (ignored by git)
├── src/                  # source code
│   └── eeg/
│       ├── data/         # ingestion / dataset helpers
│       ├── features/     # PSD, TF, entropy, coherence, extraction registry
│       ├── preprocessing/# filters, ICA, epoching, normalization
│       ├── scripts/      # CLI drivers (process_sessions, extract_features)
│       └── utils/        # logger, config loader, montage helpers
├── tests/                # unit & integration tests (use synthetic data)
├── pyproject.toml
├── poetry.lock
├── .pre-commit-config.yaml
├── .gitignore
└── README.md
```

> Note: `data/` and its subfolders are local-only and **should not** be tracked.

---

## Data placement (required before running)
Place the combined EEG CSV at:

```
data/raw/combined_dataset.csv
```

### Expected CSV format (minimal)
The ingestion step expects at least the following columns (order not strict):

```
CP3, C3, F5, PO3, PO4, F6, C4, CP4, timestamp, session_id
```

- `timestamp` can be epoch milliseconds or another monotonic time column per row.  
- `session_id` groups rows into continuous recording sessions (the ingestion script will split the combined CSV into per-session CSVs).

If your CSV contains more columns (subject metadata etc.), the ingestion code ignores unneeded columns but preserves `timestamp` and `session_id` for correct session splitting.

---

## Quickstart (development)

### 1. Install dependencies
This project uses Poetry. From repo root:

```bash
poetry install
```

### 2. Enable pre-commit hooks
```bash
poetry run pre-commit install
```

### 3. Run the test suite (fast)
```bash
poetry run pytest -q
```

---

## Usage examples

### 1) Preprocess combined CSV → per-session `.fif`
This step:
- Splits `combined_dataset.csv` into per-session CSVs,
- Loads each session into an MNE Raw,
- Filters, notches, references, runs ICA-based artifact removal,
- Saves preprocessed per-session `.fif` files to the `--out` directory.

```bash
python -m src.eeg.scripts.process_sessions --input data/raw/combined_dataset.csv --out data/interim
```

### 2) Extract features from `.fif` files (per-session `.fif` → features `.parquet`)
This step:
- Loads preprocessed `.fif`, generates sliding-window epochs,
- Extracts features (PSD, bandpowers, one_over_f_slope, entropy, nonlinear, coherence, ...),
- Writes per-session parquet files and manifest JSON. Optional sidecars (spectrograms, connectivity) are available.

Basic usage:

```bash
python -m src.eeg.scripts.extract_features --input data/interim --out data/features --window 10 --overlap 0.5
```

Additional useful flags:
- `--per-channel` include flattened per-channel features (e.g., `ch0_alpha_power`)  
- `--save-spectrograms` save per-epoch spectrogram sidecars (`.npz`)  
- `--save-connectivity` save sliding-window connectivity sidecars (`.npz`)  
- `--conn-win` and `--conn-step` to control sliding connectivity window and step  
- `--conn-bands "1-4;4-8;8-12;13-30;30-45"` to specify connectivity band ranges

### 3) Minimal end-to-end (small/test dataset)

```bash
# 1) Preprocess
python -m src.eeg.scripts.process_sessions --input data/raw/combined_dataset.csv --out data/interim

# 2) Extract features
python -m src.eeg.scripts.extract_features --input data/interim --out data/features --window 10 --overlap 0.5 --per-channel

# 3) Inspect features (Python)
python - <<'PY'
import pandas as pd
p = "data/features/<session_id>_features.parquet"
df = pd.read_parquet(p)
print(df.shape)
print(df.columns.tolist()[:50])
print(df.head())
PY
```

---

## Feature inventory (what the extractor currently produces)
- PSD (Welch), robust wrapper with nperseg fallback and EPS floor.
- Absolute band powers: delta/theta/alpha/beta/gamma (per-channel + aggregated means/stds).
- Relative bandpowers (band / total) are computed defensively (EPS).
- Peak Alpha Frequency (PAF) — `paf_mean`.
- Spectral entropy (`spec_entropy_mean`) — normalized Shannon.
- 1/f slope estimate (`one_over_f_slope`) — median slope across channels from robust log-log fit.
- Hjorth-like activity/mobility/complexity (if implemented in nonlinear module).
- Nonlinear features: permutation entropy, Higuchi fractal dimension, sample entropy.
- Pairwise coherence averaged per canonical band (mean across pairs).
- Aggregate statistics across channels (means, stds).
- Optional flattened per-channel features (via `--per-channel`) and sidecars (TF & connectivity).
- QC columns: `num_channels_with_zero_total`, `has_any_zero_total`.
- Metadata written per-parquet: `session_id`, `sfreq`, `n_channels`, `channel_names`.

---

## Tests & CI
- Unit tests use synthetic data fixtures (MNE `RawArray`) to validate feature outputs and protect against edge cases (flat channels, NaNs).
- Integration tests ensure end-to-end scripts run on synthetic inputs and produce expected artifacts.
- Add tests for new features or edge-case handling. Run them with:

```bash
poetry run pytest -q
```

---

## Development & engineering standards
- Formatting / linting: `black`, `isort`, `ruff`
- Type checks: `mypy` (optional but recommended)
- Pre-commit: use hooks to run linters and tests locally
- Commit message style: Conventional commit prefixes (feat:, fix:, chore:, test:, docs:)

---

## Makefile (suggested targets)
```makefile
install:
	poetry install

preprocess:
	python -m src.eeg.scripts.process_sessions --input data/raw/combined_dataset.csv --out data/interim

features:
	python -m src.eeg.scripts.extract_features --input data/interim --out data/features --window 10 --overlap 0.5

test:
	poetry run pytest -q
```

---

## Troubleshooting & tips
- If you see `total_<ch>` equals zero for a channel: check for flat/broken channels; the extractor will try to interpolate if a montage is present, otherwise it'll drop the channel and log a warning.
- If PSD functions fail on very short windows, reduce nperseg or use larger windows for extraction.
- For Parquet read/write, ensure `pyarrow` or `fastparquet` is installed: `poetry add pyarrow`.
- To inspect Raw files interactively, use MNE plotting:  
  ```py
  import mne
  r = mne.io.read_raw_fif("data/interim/session_xxx_preprocessed.fif", preload=True)
  r.plot()
  ```

---

## Data & Git policy (IMPORTANT)
- **Never** commit raw or derived data into the repository. Use `.gitignore` to exclude `data/` and large artifacts.
- If large/sensitive files are accidentally committed and pushed, coordinate with your team to rewrite history (e.g., `git-filter-repo` or BFG) and rotate any secrets potentially exposed.
- For artifact storage, use S3/GCS, DVC, or MLflow artifact stores — do not store big blobs in Git.

---

## Contributing
- Write tests for non-trivial changes.
- Keep changes small and well-documented.
- Use clear commit messages following the `feat/fix/docs/test/chore` prefixes.

---

## License
MIT
