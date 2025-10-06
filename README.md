# EEG Fatigue Detection (Machine Learning)

## Overview

This project implements a **ML pipeline** for analyzing EEG signals (256Hz, multi-volunteer dataset) to detect **cognitive fatigue** using both unsupervised and supervised approaches.

> **⚠️ Note:** Raw data and derived artifacts (`.fif`, `.parquet`, model checkpoints) must **not** be committed. See Data & Git Policy.

---

## Repo Structure

```
src/eeg/       - Core source code (data, models, utils, scripts)
tests/         - Unit & integration tests
data/          - Local-only storage (raw, interim, features, models, results)
pyproject.toml - Project configuration
Makefile       - Automation tasks
README.md      - Project documentation
```

---

## Setup

### 1. Install dependencies

```bash
poetry install
```

### 2. Enable pre-commit hooks

```bash
poetry run pre-commit install
```

---

## Development Workflow

- Code must pass **black**, **isort**, **ruff**, and **mypy** before commit.
- Experiments tracked via **MLflow** (default), **W&B** optional.
- Config-driven experiments using **Hydra** for reproducibility.

---

## Data & Git Policy

- Never commit raw or derived files. `.gitignore` already blocks `*.csv`, `.fif`, etc.

---

## License

MIT License
