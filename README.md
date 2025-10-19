Fatigue-ML

Overview

This repository provides a reproducible EEG-based fatigue analysis pipeline, covering ingestion, preprocessing, epoching, feature extraction, and unsupervised modeling (classical clustering or autoencoder embeddings).

⚠️ Data policy: Raw EEG data, .fif files, Parquet features, or model weights must not be committed to Git.

Highlights
	•	Robust preprocessing: bandpass + notch filtering, re-referencing, ICA artifact removal.
	•	Session-level reproducibility: .fif intermediate files saved per session.
	•	Feature extraction:
	•	PSD (Welch), absolute & relative band powers (δ/θ/α/β/γ)
	•	Peak Alpha Frequency (PAF)
	•	Spectral entropy
	•	1/f slope
	•	Nonlinear metrics: permutation entropy, Higuchi FD, sample entropy
	•	Pairwise coherence (per canonical band)
	•	Unsupervised modeling:
	•	Classical: PCA → UMAP → clustering (HDBSCAN, GMM, KMeans)
	•	Autoencoder: MLP-based latent embeddings
	•	Optional sidecars: per-epoch spectrograms, sliding-window connectivity
	•	QC logic: flat/broken channel detection, PSD floor, NaN/Inf sanitization

⸻

Repo structure

fatigue-ml/
├── data/                 # local EEG data (ignored by Git)
│   ├── raw/              # combined CSV
│   ├── interim/          # per-session preprocessed .fif files
│   └── features/         # per-session feature Parquet files & manifests
├── src/
│   └── eeg/
│       ├── data/         # dataset helpers, I/O
│       ├── preprocessing/# filters, ICA, epoching
│       ├── features/     # feature extraction modules
│       ├── models/       # unsupervised.py, autoencoder.py, embeddings, eval
│       ├── scripts/      # CLI drivers
│       └── utils/        # logger, config, montage helpers
├── tests/                # unit & integration tests
├── Makefile
├── pyproject.toml
└── README.md


⸻

Data placement

Place your combined EEG CSV here:

data/raw/combined_dataset.csv

Expected minimal columns:

CP3, C3, F5, PO3, PO4, F6, C4, CP4, timestamp, session_id

	•	timestamp = monotonic time per row
	•	session_id = session grouping

⸻

Quickstart

1) Install dependencies

poetry install

2) Enable pre-commit hooks

poetry run pre-commit install

3) Run tests

poetry run pytest -q


⸻

Preprocessing

Convert combined CSV → per-session .fif:

python -m src.eeg.scripts.process_sessions --input data/raw/combined_dataset.csv --out data/interim


⸻

Feature extraction

Per-session .fif → feature Parquet:

python -m src.eeg.scripts.extract_features --input data/interim --out data/features --window 10 --overlap 0.5

Optional flags:
	•	--per-channel → include flattened per-channel features
	•	--save-spectrograms → save .npz spectrograms
	•	--save-connectivity → save .npz connectivity matrices

⸻

Classical unsupervised modeling

Run PCA/UMAP + clustering (HDBSCAN/GMM/KMeans):

from src.eeg.models.unsupervised import run_unsupervised_pipeline

results = run_unsupervised_pipeline(
    features_path='data/features/04WdQ3hau8P7jP1LDvWy_preprocessed_features.parquet',
    out_dir='data/features/clustering',
    standardize=True,
    pca_components=10,
    umap_components=2,
    cluster_method='hdbscan',
)
print(results['embeddings'].head())

Outputs:
	•	embeddings.parquet → PCA/UMAP embeddings + cluster labels
	•	cluster_labels.npy → cluster assignments
	•	model_pipeline.joblib → saved sklearn pipeline + cluster model

⸻

Autoencoder-based modeling

Train a small MLP autoencoder on features:

from src.eeg.models.autoencoder import train_autoencoder, TrainAutoencoderConfig
import pandas as pd

df = pd.read_parquet('data/features/04WdQ3hau8P7jP1LDvWy_preprocessed_features.parquet')
cfg = TrainAutoencoderConfig(batch_size=64, epochs=20, latent_dim=16, device='cpu')

results = train_autoencoder(df, cfg, out_dir='data/features/autoencoder')
print(results['history'])

Outputs:
	•	autoencoder.pt → saved PyTorch model state + mean/std
	•	ae_training_metrics.json → training loss history

⸻

Makefile (suggested targets)

install:
	poetry install

preprocess:
	python -m src.eeg.scripts.process_sessions --input data/raw/combined_dataset.csv --out data/interim

features:
	python -m src.eeg.scripts.extract_features --input data/interim --out data/features --window 10 --overlap 0.5

unsupervised:
	python -m src.eeg.models.cli --method unsupervised --input data/features --out data/features/clustering

autoencoder:
	python -m src.eeg.models.cli --method autoencoder --input data/features --out data/features/autoencoder

test:
	poetry run pytest -q


⸻

Data & Git policy
	•	Never commit raw or derived EEG data. Use .gitignore.
	•	Store large artifacts in S3/GCS/DVC/MLflow, not Git.

⸻

Contributing
	•	Write tests for non-trivial changes.
	•	Use Conventional Commit prefixes: feat/fix/docs/test/chore.
	•	Keep commits small and descriptive.

License

MIT