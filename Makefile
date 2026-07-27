UV ?= uv

.PHONY: help install install-dev check lint fmt typecheck test cov clean build \
        docker docker-local image-lambda

help:
	@echo "install       install core deps"
	@echo "install-dev   install dev + test + lint stack"
	@echo "lint          ruff lint + format check"
	@echo "fmt           ruff auto-format + fix"
	@echo "typecheck     mypy over src/"
	@echo "test          run pytest"
	@echo "cov           pytest with coverage report"
	@echo "build         build wheel + sdist"
	@echo "docker        build main image"
	@echo "docker-local  docker-compose up with ollama"
	@echo "image-lambda  build lambda container"
	@echo "clean         drop caches + build artefacts"

install:
	$(UV) sync --locked --no-dev

install-dev:
	$(UV) sync --locked --extra bedrock --extra dask --extra linking --extra duckdb
	$(UV) run pre-commit install

check: lint typecheck test

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

fmt:
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

typecheck:
	$(UV) run mypy src/qualipilot

test:
	$(UV) run pytest

cov:
	$(UV) run pytest --cov-report=term-missing --cov-report=html

build:
	$(UV) build --clear

docker:
	docker build -f docker/Dockerfile -t qualipilot:dev .

docker-local:
	docker compose -f docker/docker-compose.yml up --build

image-lambda:
	docker build --platform linux/amd64 -f docker/Dockerfile.lambda -t qualipilot-lambda:dev .

clean:
	rm -rf build dist .coverage coverage.xml htmlcov .pytest_cache .mypy_cache .ruff_cache .hypothesis
	find src tests scripts examples -type d -name __pycache__ -exec rm -rf {} +
