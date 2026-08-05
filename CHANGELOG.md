# Changelog

This project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [3.1.1] - 2026-08-05

### Added

- Explicit caller-owned Spark session and DuckDB connection support for
  runtime adapters.

### Fixed

- GroundZero data quality checks now stay bound to their originating runtime
  session instead of resolving an implicit engine context.
- GroundZero examples now use the session-oriented data quality API rather
  than the placeholder `checkDataQuality` name.

## [3.1.0] - 2026-08-05

### Added

- GroundZero Spark and DuckDB runtime integration through managed LLM
  connection names.
- Borrowed DuckDB relations can be checked without closing their runtime
  connection.

### Changed

- The Spark extra now supports the GZ runtime's PySpark 3.5.6 as well as
  PySpark 4.2.

## [3.0.0] - 2026-07-27

### Breaking changes

- Removed the untested cuDF engine and lakehouse loaders.
- Removed the unused `Engine.describe()` method.
- Removed the per-check result helper classes; payloads remain available on
  `CheckResult.payload`.
- Every enabled LLM provider now requires an explicit model.
- Removed `--api-key`; use `QUALIPILOT_LLM__API_KEY` or a protected
  configuration file.
- Configuration files now require an explicit `--config` path.
- Removed ambiguous array-oriented `.json` input; use JSONL/NDJSON.
- Narrowed the Spark extra to the tested PySpark 4.2 series.
- Replaced the Terraform `qualipilot_config_json` string with a validated
  Lambda-safe `qualipilot_config` object.

### Added

- Dataset row, column, and dtype contracts.
- Non-destructive linkage normalization plus deterministic survivor
  selection, field merging, row consolidation, lineage, and conflict audits.
- Input source, source version, package version, and configuration hash in
  reports.
- Exact-version S3 processing and deterministic terminal report keys in
  Lambda.
- Locked dependencies, cross-platform CI, release verification, Terraform
  validation, dependency audits, and container smoke tests.

### Fixed

- Configuration precedence, secret redaction, and malformed-config errors.
- Atomic report writes and format selection.
- Dask duplicate and JSON handling.
- DuckDB identifier quoting, null semantics, duplicate queries, and resource
  cleanup.
- Record-linkage validation, candidate limits, deterministic clustering, and
  Polars/DuckDB comparison parity.
- Singleton-heavy consolidation no longer performs one dataframe slice per
  cluster.
- Lambda input validation, least-privilege IAM, retryable failure reports,
  report retention, and cached-outcome metrics.
- Reproducible Lambda image builds now include pinned Amazon Linux security
  updates.
- PyPI publishing now isolates OIDC credentials from build and test steps.
- HTML and Markdown escaping for untrusted report content.

## [2.0.1] - 2026-04-28

- Corrected package metadata, CLI errors, DuckDB path handling, strict typing,
  and report rendering.

## [2.0.0] - 2026-04-27

- First public `qualipilot` release.
