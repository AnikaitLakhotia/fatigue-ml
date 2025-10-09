# Fatigue-ML

## Overview
This repository contains a machine learning pipeline for detecting **fatigue**. The codebase is organized for: reproducible preprocessing, feature extraction, modeling, and experiment tracking.

> **⚠️ Data policy:** Raw data and derived artifacts (e.g. `.fif`, `.parquet`, and model checkpoints) **must not** be committed to Git. See *Data & Git Policy* below.

### Pipeline highlights
- Session-aware ingestion and preprocessing (filtering, re-referencing, ICA)
- Per-session intermediate files (MNE `.fif`) for reproducibility
- Feature extraction (Welch PSD, band powers, spectral entropy, Hjorth, 1/f slope, coherence)
- Config-driven experiments and experiment tracking (Hydra, MLflow, optional W&B)
- Tests, linting, and pre-commit hooks for consistent engineering standards

---

## Repository layout
```
fatigue-ml/
├── data/                      # Local only: raw, interim, features, models (gitignored)
│   ├── raw/                   # Put combined_dataset.csv here (local only)
│   ├── interim/               # Preprocessed per-session .fif (auto-generated)
│   ├── features/              # Extracted features (.parquet/.csv)
│   └── models/                # Trained model checkpoints
├── src/
│   └── eeg/                   # Core package (data, features, models, scripts, utils)
├── tests/                     # Unit & integration tests (pytest)
├── .pre-commit-config.yaml    # Pre-commit hooks (black, isort, ruff, mypy)
├── .gitignore                 # Ensures data/ and artifacts aren't committed
├── poetry.lock                # Locked dependency versions (commit this)
├── pyproject.toml             # Project dependencies & tooling
├── Makefile                   # Convenience targets (preprocess, features, train, test)
└── README.md                  # This file
```

> Note: `.pre-commit-config.yaml`, `.gitignore`, and `poetry.lock` **should** be tracked in git. The `data/` folder and its contents should **not** be tracked (see policy below).

---

## Data placement (required before running)
Place the combined CSV at:

```
data/raw/combined_dataset.csv
```

### Expected CSV format (minimal)
The pipeline expects the CSV to contain these columns (order not strict):

```
CP3, C3, F5, PO3, PO4, F6, C4, CP4, timestamp, session_id
```

- `timestamp` should be epoch milliseconds or another monotonic time column per row.  
- `session_id` groups rows into continuous recording sessions; the ingestion script will split the combined CSV into per-session CSVs and then preprocess each session independently.

**Tip:** If your CSV has extra columns (subject metadata), the ingestion code will ignore unneeded columns but keep `timestamp` and `session_id` intact.

---

## Quickstart (development)

### 1. Install dependencies
This project uses Poetry for dependency management. Install and create the venv:

```bash
poetry install
poetry shell       # optional: spawn a shell with the environment
```

### 2. Enable pre-commit hooks
```bash
poetry run pre-commit install
```

### 3. Run tests (fast, uses small or mocked data)
```bash
poetry run pytest -q
```

---

## Usage examples

### Preprocess sessions (split CSV → per-session `.fif`)
The preprocessing step reads `data/raw/combined_dataset.csv`, splits by `session_id`, filters, runs ICA, and writes per-session MNE Raw files into `data/interim/`:
```bash
python -m src.eeg.scripts.process_sessions --input data/raw/combined_dataset.csv --out data/interim
```

### Extract features (per-session `.fif` → feature parquet)
This step reads `data/interim/*.fif`, epochs each session (sliding window), extracts features, and writes per-session parquet files to `data/features/`:
```bash
python -m src.eeg.scripts.extract_features --input data/interim --out data/features --window 10 --overlap 0.5
```

### Minimal end-to-end (for small/test dataset)
```bash
# 1) Preprocess (produces interim/*.fif)
python -m src.eeg.scripts.process_sessions --input data/raw/combined_dataset.csv --out data/interim

# 2) Feature extraction (produces features/*.parquet)
python -m src.eeg.scripts.extract_features --input data/interim --out data/features --window 10 --overlap 0.5

# 3) Quick inspect (Python)
python - <<'PY'
import pandas as pd
p = "data/features/<session_id>_features.parquet"
df = pd.read_parquet(p)
print(df.shape)
print(df.columns.tolist()[:20])
print(df.head())
PY
```

---

## Data folders overview (what each folder holds)
- `data/raw/` — **Input** combined CSV; **do not** commit.  
- `data/interim/` — **Auto-generated** per-session preprocessed files (.fif). Useful for reproducibility and manual inspection. **Do not** commit.  
- `data/features/` — Feature tables (Parquet/CSV) created by feature extraction. Not committed.  
- `data/models/` — Trained model artifacts and checkpoints. Not committed.

> All `data/` subfolders are in `.gitignore` to avoid accidental commits of large or sensitive files.

---

## Development workflow & engineering standards
- Code style & static checks: `black`, `isort`, `ruff`, `mypy`. Run locally or via pre-commit.  
- Tests: `pytest` for unit and integration tests. Strive for fast unit tests (use synthetic or trimmed data).  
- Experiment tracking: `mlflow` by default; `wandb` optional. Use artifact stores for large outputs (S3/GCS).  
- Configs: `hydra` for reproducible experiments and hyperparameters.

Suggested Makefile targets (recommended):
```makefile
install:        # install deps
	poetry install

preprocess:     # produce data/interim
	python -m src.eeg.scripts.process_sessions --input data/raw/combined_dataset.csv --out data/interim

features:       # produce data/features
	python -m src.eeg.scripts.extract_features --input data/interim --out data/features --window 10 --overlap 0.5

test:
	poetry run pytest -q
```

---

## Data & Git Policy (IMPORTANT)
- **Never** commit raw or derived data. `.gitignore` prevents accidental inclusion, but always check `git status` before committing.  
- If data is accidentally committed and pushed, coordinate with your team and use `git-filter-repo` or BFG to remove sensitive blobs from history. Rewriting public history requires team coordination.  
- Use dedicated artifact storage for large files: S3, GCS, DVC, or MLflow artifacts — **not** the Git repo.

---

## Troubleshooting & tips
- If preprocessing warns about filter lengths on very short recordings, use longer windows or skip filter in test runs.  
- For Parquet read/write, install `pyarrow` if necessary: `poetry add pyarrow`.  
- To inspect raw `.fif` files interactively, use MNE's plotting tools: `mne.io.read_raw_fif(...).plot()`.

---

## Contributing
- Add tests for non-trivial changes.  
- Keep commit messages descriptive and follow the project's convention (`feat:`, `fix:`, `chore:`).  
- Run `poetry run pre-commit run --all-files` before pushing.

---

## License
MIT
