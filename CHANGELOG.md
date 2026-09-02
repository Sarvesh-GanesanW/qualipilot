# Changelog

This project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Standards/assurance mapping, a reproducible 100-million-row Spark benchmark,
  signed build-provenance attestations for release artifacts, and Lambda
  duration/concurrency alarms.

### Changed

- Spark and other engines evaluate configured range rules in one batched
  aggregate instead of one scan per column.
- LLM reports retain the resolved managed provider/model, treat report content
  as untrusted, and label generated narratives as advisory.
- Record consolidation now rejects warning linkage fits unless explicitly
  enabled after validation.
- NER audits include the spaCy runtime and model license metadata, and unknown
  label filters now fail instead of silently returning no entities.

### Fixed

- Warm Lambda invocations reuse the S3 SDK client and connection pool.

## [3.3.0] - 2026-09-02

### Added

- Managed GroundZero xAI connections and OpenAI Responses API endpoints.

### Changed

- GroundZero LLM connections now follow current provider defaults and preserve
  complete Chat Completions or Responses endpoint URLs.
- Hugging Face router endpoints use the OpenAI-compatible API while legacy
  inference endpoints remain supported.
- OpenAI project, Anthropic workspace, Anthropic sampling, and AWS Bedrock
  assume-role settings now follow the managed connection contract.
- Reports record the provider and model resolved from managed GroundZero
  connections, and range checks batch multi-column violation counts.

### Fixed

- Refreshed Lambda OS packages to include patched OpenSSL builds.

## [3.2.0] - 2026-08-26

### Added

- Optional spaCy named-entity extraction API and CLI with batched processing,
  character offsets, model/source provenance, label filtering, and atomic JSON
  audits.
- Per-check severity overrides, portable dtype-family contracts, and quantile
  provenance across dataframe engines.
- Ground-truth linkage evaluation with confusion-matrix counts, precision,
  recall, and F1, including labeled pairs omitted during blocking.
- Forward-compatible `QualityReport.from_json()` loading for newer optional
  report fields.

### Changed

- Linkage fitting now reports convergence and safety diagnostics, smooths
  learned probabilities, and rejects degenerate or inverted fits before
  clustering or consolidation.
- Freshness checks compare timezone-aware instants consistently across
  engines and fail closed when configured temporal data is missing or invalid.
- Managed GroundZero LLM connections now require explicit models; Hugging Face
  connections also require an explicit inference endpoint or base URL.
- LLM report payloads are bounded while retaining actionable findings, and
  Markdown reports retain JSON payload details for extension checks.

### Fixed

- Portable numeric, date, and timestamp handling for object/string columns,
  including Spark and Dask edge cases.
- NER audits now reject non-finite identifiers before serialization and record
  the normalized label filter used during extraction.
- Report deserialization remains compatible with the minimum supported
  Pydantic release.
- Refreshed locked development dependencies and the Lambda base image.

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
