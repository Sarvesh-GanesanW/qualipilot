# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Lowered `Development Status` classifier from `5 - Production/Stable` to
  `4 - Beta`. Several engines (Spark, Dask, DuckDB) and the Lambda handler
  ship without test coverage; the previous classifier overpromised.

### Added
- `python -m qualipilot` entry point (`__main__.py`).
- This `CHANGELOG.md`.

### Fixed
- `LICENSE` now uses ASCII quotes; some packaging tools mis-detect the
  Unicode smart quotes the previous file shipped with.
- CI test job now installs `linking` and `duckdb` extras; previously
  `tests/test_linking.py` imported `rapidfuzz` unconditionally and the
  whole matrix red-failed on a `ModuleNotFoundError`.

## [2.0.0] — 2026-04-27

### Added
- First public PyPI release as `qualipilot`.
- Pluggable dataframe engines: Polars (default), Pandas, Dask, cuDF,
  DuckDB, Spark.
- Pluggable LLM providers: AWS Bedrock (Converse API), Ollama,
  OpenAI-compatible, plus a `none` provider.
- In-house Fellegi-Sunter record linker with Polars blocking, rapidfuzz
  string distance, numpy EM, and DuckDB-backed alternative path.
- Typer CLI with `check` and `link` subcommands and severity-gate exit
  codes (`--fail-on ok|warn|error`).
- Pydantic v2 typed config + results, JSON/HTML/Markdown reports.
- Docker, Terraform/Lambda, and GitHub Actions release workflow with
  PyPI Trusted Publishing (OIDC, no tokens).

### Notes
- Renamed from `data-pilot-checker` / module `datapilot` because
  `datapilot` was already taken on PyPI.
