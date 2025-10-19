# Makefile — convenience commands for dev

.PHONY: install test format lint docs preprocess features embeddings unsupervised clean

# Use poetry-managed environment
install:
	@echo "Installing dependencies with poetry..."
	poetry install

# Run full test suite
test:
	poetry run pytest -q

# Format and lint
format:
	poetry run black .
	poetry run isort .

lint:
	poetry run ruff check src tests

# Run preprocessing script (example)
preprocess:
	poetry run python -m src.eeg.scripts.process_sessions --input data/raw/combined_dataset.csv --out data/interim

# Extract features
features:
	poetry run python -m src.eeg.scripts.extract_features --input data/interim --out data/features --window 10 --overlap 0.5

# Make session embeddings (models)
embeddings:
	poetry run python -m src.eeg.models.cli embeddings --in data/features --out data/processed/session_embeddings.parquet

# Run unsupervised modeling (example)
unsupervised:
	poetry run python -m src.eeg.models.cli unsupervised --features data/processed/session_embeddings.parquet --out data/models/unsupervised --pca 10 --umap 2 --cluster hdbscan

# Clean caches and build artifacts
clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache
