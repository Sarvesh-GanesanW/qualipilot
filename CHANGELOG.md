# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Lowered `Development Status` classifier from `5 - Production/Stable` to
  `4 - Beta`. Several engines (Spark, Dask, DuckDB) and the Lambda handler
  ship without test coverage; the previous classifier overpromised.
- `qualipilot.linking.em.estimate_parameters` now returns a typed
  `EMParams` (TypedDict). Callers that unpacked `params["m"]` etc. as
  the old `dict[str, np.ndarray | float]` keep working — the keys are
  identical — but they now get proper static types at use sites.
- `Check._execute` abstract return type narrowed from `tuple[str, dict]`
  to `tuple[Severity, dict]`. Subclasses now annotate severity
  explicitly; mypy catches accidental returns of bogus severity strings.
- CLI options `--engine`, `--format`, `--llm`, and `--fail-on` are now
  proper `Choice` types rendered by Typer. Typos like `--engine polrs`
  exit cleanly with a parameter error instead of crashing inside
  `build_engine`. The accepted values stay identical.

### Added
- `LinkConfig.em_random_seed` (default `0`). The polars and DuckDB
  linkers both honour it, replacing the previously hardcoded seed.
  Useful when you want deterministic-but-different sampling across
  comparative trials.
- `LLMConfig` now carries a per-provider temperature validator. Setting
  `provider="bedrock"` with `temperature > 1.0` raises at config time;
  Bedrock would otherwise 400 mid-run.
- Test suite for `qualipilot.engines.duckdb_engine` (was 0% covered,
  now 87%). Includes parity tests against polars/pandas for null counts,
  duplicates, quantiles, `count_outside`; from-CSV round-trip; isolation
  test for two engines in the same process.

### Fixed
- `DuckDBEngine.from_any` no longer crashes on file paths. Previously
  it passed the path as a `?` parameter to `read_csv_auto` /
  `read_parquet` / `read_json_auto`; DuckDB rejects parameters in those
  table-function slots with "Unexpected prepared parameter". The fix
  inlines the path with single-quote escaping to keep the call
  injection-safe.

### Added
- `python -m qualipilot` entry point (`__main__.py`).
- This `CHANGELOG.md`.

### Fixed
- `LICENSE` now uses ASCII quotes; some packaging tools mis-detect the
  Unicode smart quotes the previous file shipped with.
- CI test job now installs `linking` and `duckdb` extras; previously
  `tests/test_linking.py` imported `rapidfuzz` unconditionally and the
  whole matrix red-failed on a `ModuleNotFoundError`.
- 35 `mypy --strict` errors across `engines/duckdb_engine`,
  `linking/em`, `linking/linker`, `linking/duckdb_linker`,
  `engines/{pandas,dask,cudf}_engine`, `checks/{base,missing}`,
  `llm/bedrock`, `logging_setup`, `lakehouse`, and `cli`. The package
  now type-checks cleanly under strict mode, matching its
  `Typing :: Typed` classifier and `py.typed` marker.

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
