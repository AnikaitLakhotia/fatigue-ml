# Fatigue-ML

## Overview

This repository provides a reproducible EEG-based fatigue analysis pipeline, covering ingestion, preprocessing, epoching, feature extraction, and unsupervised modeling (classical clustering or autoencoder embeddings).

⚠️ **Data policy:** Raw EEG data, `.fif` files, Parquet features, or model weights **must not** be committed to Git.

### Highlights
- **Robust preprocessing:** bandpass + notch filtering, re-referencing, ICA artifact removal  
- **Session-level reproducibility:** `.fif` intermediate files saved per session  
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

## Repository Structure

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

## Data Placement

Place your combined EEG CSV here:

```
data/raw/combined_dataset.csv
```

Expected minimal columns:

```
CP3, C3, F5, PO3, PO4, F6, C4, CP4, timestamp, session_id
```

---

## Quickstart

### 1. Install dependencies

```bash
poetry install
```

### 2. Run preprocessing

Convert combined CSV → per-session `.fif`:

```bash
python -m src.eeg.scripts.process_sessions --input data/raw/combined_dataset.csv --out data/interim
```

### 3. Extract features

Per-session `.fif` → per-epoch feature Parquet (+ optional `.npz` SSL outputs):

```bash
python -m src.eeg.scripts.extract_features --input data/interim --out data/features --window 10 --overlap 0.5 --save-spectrograms --save-connectivity --save-ssl
```

Each epoch row now includes:
- `timestamp_start`, `timestamp_end`, `timestamp_center`
- Per-channel features for each electrode

### 4. Generate session embeddings

```bash
python -m src.eeg.models.cli embeddings --in data/features --out data/processed/session_embeddings.parquet
```

---

## License

MIT
