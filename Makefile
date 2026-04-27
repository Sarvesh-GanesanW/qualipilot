PY ?= python
UV ?= uv

.PHONY: help install install-dev lint fmt typecheck test cov clean build \
        docker docker-local docker-bedrock image-lambda run-local release

help:
	@echo "install       install core deps"
	@echo "install-dev   install dev + test + lint stack"
	@echo "lint          ruff lint + format check"
	@echo "fmt           ruff auto-format + fix"
	@echo "typecheck     mypy over src/"
	@echo "test          run pytest"
	@echo "cov           pytest with coverage report"
	@echo "build         hatch wheel + sdist"
	@echo "docker        build main image"
	@echo "docker-local  docker-compose up with ollama"
	@echo "image-lambda  build lambda container"
	@echo "clean         drop caches + build artefacts"

install:
	$(UV) pip install -e .

install-dev:
	$(UV) pip install -e ".[dev,all]"
	pre-commit install

lint:
	ruff check src tests
	ruff format --check src tests

fmt:
	ruff check --fix src tests
	ruff format src tests

typecheck:
	mypy src/qualipilot

test:
	pytest

cov:
	pytest --cov=qualipilot --cov-report=term-missing --cov-report=html

build:
	$(UV) build

docker:
	docker build -f docker/Dockerfile -t qualipilot:latest .

docker-local:
	docker compose -f docker/docker-compose.yml up --build

image-lambda:
	docker build -f docker/Dockerfile.lambda -t qualipilot-lambda:latest .

clean:
	rm -rf build dist .coverage coverage.xml htmlcov .pytest_cache .mypy_cache .ruff_cache .hypothesis
	find . -type d -name __pycache__ -exec rm -rf {} +
