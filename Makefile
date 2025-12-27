# Makefile - developer / CI convenience tasks
# Usage: make <target>
#
# Environment vars (optional):
#   POETRY       - path to poetry binary (default: poetry)
#   DOCKER_IMAGE - base docker image name (default: fatigue-ml)
#   DOCKER_TAG   - docker image tag (default: latest)
#   DOCKER_REGISTRY - registry for pushing images (optional)
#   PYTORCH_CUDA_INDEX - optional index URL to install CUDA-enabled PyTorch in GPU Dockerfile
#
# Example:
#   make install
#   make lint
#   make test
#   make docker-build-cpu DOCKER_TAG=dev

POETRY ?= poetry
PY ?= python3

DOCKER_IMAGE ?= fatigue-ml
DOCKER_TAG   ?= latest
DOCKER_REGISTRY ?=
PYTORCH_CUDA_INDEX ?=

# Default goal
.DEFAULT_GOAL := help

.PHONY: help install lock install-dev fmt lint fmt-check typecheck test coverage \
        build-wheel clean ci-check docker-build-cpu docker-build-gpu docker-push \
        docker-run-test docker-run bash

help:
	@echo "Makefile targets:"
	@echo
	@echo "  install            Install project dependencies via Poetry"
	@echo "  lock               Generate/refresh poetry.lock"
	@echo "  install-dev        Install dev dependencies (for linting/typechecking/tests)"
	@echo "  fmt                Run code formatters (black + isort)"
	@echo "  fmt-check          Check formatting (black --check, isort --check-only)"
	@echo "  lint               Run linters (ruff)"
	@echo "  typecheck          Run mypy --strict on src/"
	@echo "  test               Run pytest -q"
	@echo "  coverage           Run pytest with coverage report"
	@echo "  build-wheel        Build a wheel via Poetry"
	@echo "  ci-check           Run a CI-like sequence: install, lint, typecheck, test"
	@echo
	@echo "  docker-build-cpu   Build CPU Docker image (Dockerfile.cpu)"
	@echo "  docker-build-gpu   Build GPU Docker image (Dockerfile.gpu)"
	@echo "  docker-push        Tag & push image to DOCKER_REGISTRY (requires env var)"
	@echo "  docker-run-test    Run pytest inside the built docker image (mounts repo)"
	@echo "  docker-run         Run default command in image"
	@echo
	@echo "  clean              Remove build artifacts and caches"
	@echo

# Project / Python tasks
install:
	@echo "Installing dependencies with Poetry..."
	$(POETRY) config virtualenvs.create false || true
	$(POETRY) install --no-interaction --no-ansi

lock:
	@echo "Generating/updating poetry.lock..."
	$(POETRY) lock

install-dev:
	@echo "Installing dev dependencies (lint, test, typecheck)..."
	$(POETRY) config virtualenvs.create false || true
	$(POETRY) install --no-interaction --no-ansi

fmt:
	@echo "Formatting code (black + isort)..."
	$(POETRY) run black .
	$(POETRY) run isort .

fmt-check:
	@echo "Checking formatting (black --check + isort --check-only)..."
	$(POETRY) run black --check .
	$(POETRY) run isort --check-only .

lint:
	@echo "Running ruff..."
	$(POETRY) run ruff check src tests

typecheck:
	@echo "Running mypy --strict on src/ ..."
	$(POETRY) run mypy --strict src

test:
	@echo "Running pytest..."
	$(POETRY) run pytest -q

coverage:
	@echo "Running pytest with coverage..."
	$(POETRY) run pytest --cov=src --cov-report=term-missing

build-wheel:
	@echo "Building wheel (poetry build)..."
	$(POETRY) build -f wheel

ci-check: install lint typecheck test
	@echo "CI check completed."

# Docker targets
docker-build-cpu:
	@echo "Building CPU Docker image: $(DOCKER_IMAGE):$(DOCKER_TAG)"
	docker build -f Dockerfile.cpu -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-build-gpu:
	@echo "Building GPU Docker image: $(DOCKER_IMAGE):gpu-$(DOCKER_TAG)"
	# Pass PYTORCH_CUDA_INDEX if set to allow installing CUDA-specific PyTorch wheels
ifeq ($(strip $(PYTORCH_CUDA_INDEX)),)
	docker build -f Dockerfile.gpu -t $(DOCKER_IMAGE):gpu-$(DOCKER_TAG) .
else
	docker build --build-arg PYTORCH_CUDA_INDEX="$(PYTORCH_CUDA_INDEX)" -f Dockerfile.gpu -t $(DOCKER_IMAGE):gpu-$(DOCKER_TAG) .
endif

docker-push:
ifndef DOCKER_REGISTRY
	$(error DOCKER_REGISTRY is not set. Usage: make docker-push DOCKER_REGISTRY=<registry> DOCKER_TAG=<tag>)
endif
	@echo "Tagging and pushing $(DOCKER_IMAGE):$(DOCKER_TAG) -> $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)"
	docker tag $(DOCKER_IMAGE):$(DOCKER_TAG) $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	docker push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)

docker-run-test:
	@echo "Running pytest inside Docker image (binds current repo)"
	docker run --rm -v $(PWD):/app -w /app $(DOCKER_IMAGE):$(DOCKER_TAG) bash -c "pytest -q || (echo 'Tests failed in container' && exit 1)"

docker-run:
	@echo "Running container (interactive). To override default cmd, append CMD='...'"
	docker run --rm -it -v $(PWD):/app -w /app $(DOCKER_IMAGE):$(DOCKER_TAG) bash

# Utilities
bash:
	@echo "Opening shell inside CPU docker image (if built)."
	docker run --rm -it -v $(PWD):/app -w /app $(DOCKER_IMAGE):$(DOCKER_TAG) bash

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf build/ dist/ .pytest_cache .mypy_cache .ruff_cache __pycache__ *.egg-info
	@find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	@find . -name "*.pyc" -type f -delete
	@echo "Done."