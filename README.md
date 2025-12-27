# Fatigue-ML

EEG fatigue analysis pipeline for reproducible research and deployment.

This repository provides a full pipeline: CSV → per-session `.fif` → per-epoch features → session embeddings → unsupervised modeling and optional SSL training. It includes tools for preprocessing, feature extraction, modeling, serving, and observability.

---

## TL;DR

- Convert combined CSV into per-session `.fif` files (with `TIMESTAMP` channel).
- Extract per-epoch features (PSD, spectrogram, connectivity, nonlinear metrics).
- Create session-level embeddings and run unsupervised modeling (PCA/UMAP/clustering).
- Optional: SSL training, model export, FastAPI serving, MLflow tracking, Prometheus metrics.
- Tooling: Poetry, Docker images (cpu/gpu/serving), pre-commit hooks (ruff/black/mypy), CI workflows.

Key entrypoints:
- `python -m src.eeg.scripts.process_sessions` — CSV → per-session `.fif`
- `python -m src.eeg.scripts.extract_features` — `.fif` → features parquet
- `python -m src.eeg.models.cli embeddings` — features → session embeddings
- `python -m src.eeg.scripts.train_ssl_tf` — SSL training (PyTorch / Lightning)
- `uvicorn src.eeg.serving.app:app` — model serving (FastAPI)

---

## Quick start

### 1. Local dev setup
Recommended: use Python 3.11 and Poetry.

```bash
# install dependencies
poetry install

# validate formatting and types locally (pre-commit / CI style)
poetry run ruff check src tests
poetry run black --check .
poetry run mypy --strict src
```

### 2. Run preprocess (CSV → .fif)

```bash
python -m src.eeg.scripts.process_sessions \
  --input data/raw/combined_dataset.csv \
  --out data/interim
```

This produces files like `data/interim/{session_id}_preprocessed_raw.fif` and `{session_id}_preprocessed_meta.json`.

### 3. Extract features (parallel)

```bash
# CPU path (recommended in CI)
python -m src.eeg.scripts.extract_features many \
  --input data/interim \
  --out data/features \
  --backend numpy \
  --n-jobs 8 \
  --connectivity-mode minimal
```

GPU path (single-process per GPU recommended):

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.eeg.scripts.extract_features many \
  --input data/interim \
  --out data/features \
  --backend torch \
  --device cuda \
  --n-jobs 1
```

### 4. Create session embeddings

```bash
python -m src.eeg.models.cli embeddings \
  --in data/features \
  --out data/processed/session_embeddings.parquet
```

### 5. Unsupervised pipeline / clustering

```bash
python -m src.eeg.models.cli unsupervised \
  --features data/processed/session_embeddings.parquet \
  --out data/processed/models \
  --pca 10 --umap 2 --cluster kmeans
```

### 6. Train SSL model (optional; PyTorch required)

```bash
python -m src.eeg.scripts.train_ssl_tf \
  --data "data/ssl/*.npz" \
  --out_dir runs/ssl_run \
  --batch_size 8 \
  --epochs 10
```

---

## Docker

We provide Dockerfiles for reproducible builds. Example (CPU):

```bash
# build
make docker-build-cpu DOCKER_IMAGE=fatigue-ml DOCKER_TAG=local

# run quick sanity check
docker run --rm fatigue-ml:local
```

For serving:

```bash
# build serving image (Dockerfile.serving)
make docker-build-serving
docker run --rm -p 8080:8080 fatigue-ml-serving:local
```

---

## CI / Tests

CI runs:
- ruff (lint), black (format), mypy (type-check)
- pytest unit and integration tests
- Docker image build (cpu) artifact
- Optional GPU smoke tests on gated runners

Run tests locally:

```bash
poetry run pytest -q
```

Run a focused test:

```bash
poetry run pytest tests/test_ssl_training.py -q
```

---

## Modeling & Exports

- Session embeddings are created from per-epoch features (`src/eeg/models/embeddings.py`).
- Unsupervised pipeline supports PCA, UMAP and clustering (HDBSCAN/GMM/KMeans).
- A simple Autoencoder baseline is provided in `src/eeg/models/autoencoder.py`.
- Export helpers (TorchScript/ONNX) are in `src/eeg/models/export.py` (when present).

Model registry & tracking:
- MLflow integration helper available: `src/eeg/utils/experiment.py`.
- The training scripts optionally log metrics / artifacts to MLflow (local filesystem by default).

---

## Serving & Observability

- FastAPI app at `src/eeg/serving/app.py` (endpoints: `/health`, `/predict`).

- Prometheus metrics middleware and JSON-structured logs available; enable via env vars.
- OpenTelemetry stubs included for optional tracing.

---

## Project layout

```
src/eeg/
  features/        # PSD, spectrograms, connectivity, TF patches
  preprocessing/   # filters, epoching, normalization, artifact removal
  models/          # autoencoder, embeddings, unsupervised pipeline, export
  scripts/         # CLI: process_sessions, extract_features, train_ssl_tf
  serving/         # FastAPI + metrics + exporters
  utils/           # logger, config loader, experiment helpers
tests/             # unit & integration tests
Dockerfile.cpu
Dockerfile.gpu
Dockerfile.serving
Makefile
pyproject.toml
poetry.lock
```

---

## Contributing & governance

- Follow `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`.
- Branch strategy: small feature branches per commit/PR; run pre-commit hooks and all tests before opening PR.
- Protect `main` branch with required checks: `ruff`, `mypy`, `pytest`.

---

## Troubleshooting & runbook highlights

- If `poetry install` fails in Docker: ensure Poetry version pinned and `PATH` contains `/root/.local/bin`.
- If `mne` I/O fails reading `.fif`, verify MNE version compatibility.
- If GPU training crashes: set `CUDA_VISIBLE_DEVICES` and verify PyTorch + CUDA versions match installed drivers.

---

## License & Contact

Maintainer: Anikait Lakhotia <alokhoti@uwaterloo.ca>  
License: MIT
