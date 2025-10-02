# EEG Fatigue Detection (Machine Learning)

## Overview
This project implements a **ML pipeline** for analyzing EEG signals
(256Hz, multi-volunteer dataset) with the goal of detecting **fatigue**.

---

## Repo Structure
```
src/eeg/       - Core source code (data, models, utils, scripts)
tests/         - Unit & integration tests
```

---

## Setup
### 1. Install dependencies
```bash
poetry install
```

### 2. Enable pre-commit
```bash
poetry run pre-commit install
```

---

## Development Workflow
- All code must pass **black**, **isort**, **ruff**, and **mypy** before commit.
- Experiments tracked via **MLflow** by default; **W&B** optional.
- Config-driven experiments (Hydra) for reproducibility.

---

## License
MIT License

