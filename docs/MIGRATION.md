# Migrating from 2.x to 3.0

Version 3 removes APIs that were untested or unused and tightens configuration
at trust boundaries.

## Result payload classes

The following public models were removed:

- `ColumnNullStat`
- `DuplicateInfo`
- `OutlierInfo`
- `RangeViolationInfo`
- `CardinalityInfo`
- `FreshnessInfo`

Check data remains in `CheckResult.payload` with the same check-specific field
names. Read it from the matching result:

```python
missing = next(
    result for result in report.results if result.name == "missing_values"
)
for column in missing.payload["per_column"]:
    print(column["column"], column["null_count"])
```

Applications that need a stable typed projection should validate the payload
with an application-owned Pydantic model or `TypedDict`.

`CheckResult.payload` remains intentionally untyped and may gain new
check-specific fields in compatible releases.

## Engines and loaders

- `Engine.describe()` was removed because the checker never used it. Use the
  native dataframe API for exploratory statistics.
- The cuDF adapter was removed. Convert supported inputs to Polars or Pandas
  before running the checker.
- `qualipilot.lakehouse` and the `iceberg`/`delta` extras were removed. Load
  those tables with their maintained client or Spark, then pass the resulting
  dataframe to Qualipilot.
- DuckDB relations are no longer accepted as inputs because adapting them
  forced eager materialization. Pass a supported file path, Arrow table,
  Pandas dataframe, or Polars dataframe.
- Array-oriented `.json` inputs were removed because backend parsers disagreed
  on their row model and required eager loading. Use `.jsonl` or `.ndjson`,
  with one object per line.
- The optional Spark engine now targets PySpark 4.2.x, the version exercised
  by CI. Keep an older Qualipilot release if a Spark 3.5 runtime is mandatory.

## Configuration and CLI

- Configuration files are loaded only with `--config`; working-directory
  discovery was removed to prevent implicit network calls and file writes.
- Enabled LLM providers require an explicit model.
- `--api-key` was removed to keep secrets out of process listings. Use
  `QUALIPILOT_LLM__API_KEY` or a protected explicit config file.
- Config-file logging fields were removed. Use global CLI logging flags or
  `QUALIPILOT_LOG_LEVEL` and `QUALIPILOT_JSON_LOGS`.
- The Terraform deployment input `qualipilot_config_json` was replaced by
  the typed HCL object `qualipilot_config`. Remove `jsonencode` and JSON
  string quoting from Terraform callers. The object exposes only
  Lambda-supported settings and accepts only `UTC` for
  `checks.freshness_timezone`.

## Reports

Reports now include a schema version, package version, source provenance,
check execution status, and structured LLM failure status. Consumers should
ignore unknown fields and gate on `schema_version`.
