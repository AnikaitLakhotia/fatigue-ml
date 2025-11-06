# Fatigue-ML - README

EEG fatigue analysis pipeline for reproducible research and deployment.

This is a system for EEG preprocessing, per-session feature extraction, SSL
embedding generation, and unsupervised modeling.

---

## TL;DR

- A focused pipeline that converts a combined CSV into per-session `.fif`
  files, extracts per-epoch features (band powers, spectrograms, connectivity),
  and creates session-level embeddings for downstream modeling.
- Built for scalability: batched FFTs, optional GPU (PyTorch), file-level
  parallelism, dtype/resampling optimizations, and clear CLI entrypoints.
- Key entrypoints:
  - `python -m src.eeg.scripts.process_sessions` - CSV → per-session `.fif`
  - `python -m src.eeg.scripts.extract_features` - `.fif` → features parquet
  - `python -m src.eeg.models.cli embeddings` - features → session embeddings

---

## Key Capabilities

- **Robust preprocessing:** bandpass + notch filtering, re-referencing, ICA artifact removal.
- **Session-level reproducibility:** `.fif` intermediate files saved per session.
- **Feature extraction:**
  - PSD (Welch), absolute & relative band powers (δ/θ/α/β/γ)
  - Peak Alpha Frequency (PAF)
  - Spectral entropy
  - 1/f slope
  - Nonlinear metrics: permutation entropy, Higuchi FD, sample entropy
  - Pairwise coherence (per canonical band)
  - Per-epoch **timestamps** (start, end, center)
  - **Per-channel features** for all 8 canonical locations: `['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4']`
- **Optional sidecars:** per-epoch spectrograms, sliding-window connectivity
- **SSL embeddings:** `.npz` files saved in `data/ssl/`
- **QC logic:** flat/broken channel detection, PSD floor, NaN/Inf sanitization

---

## Directory Layout

```
fatigue-ml/
├── data/
│   ├── raw/              # combined EEG CSV
│   ├── interim/          # per-session preprocessed .fif files
│   ├── features/         # per-session feature parquet files
│   ├── ssl/              # per-session .npz embeddings (SSL stage)
│   └── processed/        # combined embeddings parquet, clustering outputs
├── src/
│   └── eeg/
│       ├── data/         # dataset helpers, I/O
│       ├── preprocessing/# filters, epoching, normalization
│       ├── features/     # PSD, entropy, connectivity, patch extraction
│       ├── models/       # unsupervised.py, autoencoder.py, embeddings.py, eval.py
│       ├── scripts/      # CLI drivers (process_sessions, extract_features)
│       └── utils/        # logger, config, montage helpers
├── tests/                # unit & integration tests
├── Makefile
├── pyproject.toml
└── README.md
```

---

## Install

Use Poetry to manage dependencies. If you require GPU acceleration, install
PyTorch with the appropriate CUDA support for your environment.

```bash
poetry install
```

In CI, prefer a CPU-only matrix by default; add a separate GPU workflow if you
have GPU runners available.

---

## Usage

### 1) Preprocess CSV → session FIF

```bash
python3 -m src.eeg.scripts.process_sessions   --input data/raw/combined_dataset.csv   --out data/interim
```

This produces files like `data/interim/{session_id}_preprocessed_raw.fif` and
ensures a `TIMESTAMP` channel is present.

**Options:** `--no-resample`, `--resample-sfreq`, `--dtype float32|float64`, `--channels`.

### 2) Feature Extraction (parallel)

CPU (NumPy) path:

```bash
python3 -m src.eeg.scripts.extract_features   many   --input data/interim   --out data/features   --backend numpy   --n-jobs 8   --connectivity-mode minimal
```

GPU (Torch) path - ensure proper GPU assignment per worker:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.eeg.scripts.extract_features   many   --input data/interim   --out data/features   --backend torch   --device cuda   --n-jobs 1   --connectivity-mode minimal
```

**Options:** `--window`, `--overlap`, `--nperseg`, `--noverlap`, `--dtype`, `--save-spectrograms`, `--save-connectivity`, `--save-ssl`.

### 3) Create session embeddings

```bash
python -m src.eeg.models.cli embeddings   --in data/features   --out data/processed/session_embeddings.parquet
```

---

## File formats & outputs

- **`.fif`** - per-session preprocessed raw (MNE Raw) including `TIMESTAMP`.
- **Parquet** - per-session per-epoch feature tables (one parquet file per
  session by default) containing scalar features and optional spectrogram/
  connectivity sidecars (arrays may be stored or referenced).
- **`.npz`** - optional per-session SSL outputs (vectors or codebook tokens).

Each epoch row includes (examples):
- `timestamp_start`, `timestamp_end`, `timestamp_center`
- `delta_power_mean`, `theta_power_mean`, `alpha_power_mean`, ...
- `theta_alpha_ratio`, `one_over_f_slope`, `spec_entropy_mean`, ...
- Optionally per-channel `alpha_ch0`, `alpha_ch1`, ...

---

## Internals

- Batched Welch PSD and spectrogram computation (`src.eeg.features._fast_spectral`).
- Two backends: `numpy` (vectorized FFTs) and `torch` (stft on CPU/GPU).
- Connectivity computed via averaged STFT cross-spectra; `connectivity_mode`
  controls computation budget.
- Early dtype casting to `float32` by default to reduce memory and speed up
  FFTs; `float64` available if exact numerical parity is required.

---

## Performance tuning & best practices

1. Tune `--nperseg` / `--noverlap` to balance time-frequency resolution vs compute.
2. Resample early to lower the sample rate and reduce FFT cost.
3. Use `--connectivity-mode minimal` or `none` to avoid expensive all-pairs coherence.
4. For GPU: prefer one process per GPU or explicit GPU assignment per worker.
5. Use `pyarrow` parquet engine for fast IO.

---

## Testing & CI

- Add parity tests between `numpy` and `torch` backends on small synthetic inputs.
- Keep CI CPU-only by default; add optional GPU workflow for smoke tests.
- Use `ruff`, `black`, and `mypy` in pre-commit hooks and CI pipelines.

---

## Reproducibility & observability

- Pin dependency ranges and use lockfiles for CI.
- Log versions, git SHA, and environment (python, torch, cuda) per run.
- Emit per-file processing durations and counts to a metrics backend.

---

## Security & Privacy

- Do not commit data. `.gitignore` excludes `data/`, `.fif`, `.parquet`, and artifacts.
- Use encrypted object storage for raw EEG and lock down access via IAM.

---

## Contributing

Run linters & tests locally:

```bash
poetry run ruff check src tests
poetry run black --check .
poetry run pytest -q
```

Add tests for new helpers and ensure backend parity where applicable.

---

## Contact

Maintainer: Anikait Lakhotia <alokhoti@uwaterloo.ca>

---

## License

MIT
