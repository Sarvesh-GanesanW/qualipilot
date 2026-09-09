# Quality contracts

Quality contracts are declarative rule packs stored in `CheckConfig.rule_packs`.
They accept only named columns and the operators `eq`, `ne`, `gt`, `ge`, `lt`,
and `le`; Qualipilot never evaluates Python expressions or user SQL.

```python
from qualipilot import DataQualityChecker
from qualipilot.models.config import (
    CheckConfig,
    ComparisonRule,
    ForeignKeyRule,
    IcebergTableConfig,
    QualityContract,
    QualipilotConfig,
    SchemaPolicy,
)

contract = QualityContract(
    name="orders-v1",
    comparisons=[
        ComparisonRule(
            name="end-after-start",
            left_column="end",
            operator="ge",
            right_column="start",
        )
    ],
    foreign_keys=[
        ForeignKeyRule(
            name="customer",
            columns=["customer_id"],
            reference="customers",
            reference_columns=["id"],
        )
    ],
)
report = DataQualityChecker(
    orders,
    QualipilotConfig(checks=CheckConfig(rule_packs=[contract])),
    references={"customers": customers},
).run()
```

References are caller-owned dataframes and resolve through the source engine.
For a borrowed DuckDB relation, pass the caller-owned `duckdb_connection` that
created it so source and references execute in the same connection; Qualipilot
otherwise rejects references before running a check. `nulls_pass=True` means
any null or floating NaN key or comparison operand is not a violation. The Iceberg check uses the caller's Spark Iceberg runtime, including the
native Java Table API and snapshot-scoped metadata tables. It records metadata
health and does not assert full physical file integrity.

Nested values are opt-in. Pandas and Polars sort map keys before JSON
serialization; Spark and DuckDB use their native JSON serializers, whose map
key order can differ across runtimes. Do not use normalized nested map columns
as cross-engine duplicate keys when byte-for-byte equality matters.

## Iceberg table contracts

Use `DataQualityChecker.from_iceberg()` to bind both row checks and Iceberg
metadata checks to the same caller-session snapshot. It uses the caller's
Spark Iceberg runtime and never closes the session or modifies the table.

```python
with DataQualityChecker.from_iceberg(
    "catalog.analytics.orders",
    spark_session=spark,
    config=QualipilotConfig(
        iceberg=IcebergTableConfig(
            identifier="catalog.analytics.orders",
            max_manifest_count=100,
            expected_schema_id=7,
            file_probe_max_files=100,
        )
    ),
)
    ) as checker:
    report = checker.run()
```

The optional file probe checks bounded file existence and recorded byte length
through the caller's Hadoop configuration. It does not validate Parquet
contents or checksums. Run the real local-catalog integration test with:

```sh
env -u SPARK_HOME -u PYSPARK_PYTHON uv run --isolated \
  --with 'pyspark==3.5.6' --with-editable . \
  pytest -q -m integration tests/test_iceberg_integration.py --no-cov
```

## Typed reports and metrics

`CheckResult.typed_payload()` returns a concrete model for built-in checks,
including linkage skipped/result variants, contract findings, and Iceberg
metadata. Use `result.payload_as(MyPydanticModel)` for application extensions.
`QualityReport.from_json()` is the forward-compatible report reader; direct
`model_validate()` is intentionally strict for same-version payloads.

`QualityReport.iter_metrics()` and `prometheus_text()` expose source-redacted,
finite metrics. `emit_metrics(callback)` sends the same typed metric records to
a stdlib logging or application adapter.

## Nested inputs

Nested columns are rejected by default. With `allow_nested=True`, Pandas,
Polars, and Dask serialize container values with canonical JSON; Spark and
DuckDB use native JSON serialization. Native map key order can differ across
Spark/DuckDB runtimes, so normalized maps are not portable byte-for-byte
duplicate keys. Rules operate on whole normalized columns; nested leaf paths
are not rule targets.

## Reproducible fixture evidence

```sh
uv run python scripts/contract_evidence.py --output contract-evidence.json
```

The command hashes semantic report content after removing run timestamps and
check durations. It is project evidence, not independent validation.

Rule packs can include `SchemaPolicy(columns={...})` to control additions,
removals, and type changes. Register application checks by explicit name with
`register_check("my_check", callable)` and list that name in
`CheckConfig(registered_checks=["my_check"])`; no expressions or SQL are
evaluated. For a concrete built-in payload view, use
`result.payload_as(QualityContractPayload).violation_count`.
