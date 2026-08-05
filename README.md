# qualipilot

Qualipilot runs configurable checks over tabular data and returns typed
JSON, HTML, or Markdown reports. It supports CSV, Parquet, JSONL, and NDJSON
files plus several dataframe backends. Columns must contain scalar values;
flatten nested arrays, maps, and objects before checking. LLM-generated
narrative is optional and disabled by default.

The project is beta software. Validate check semantics and performance
against representative data before using report severities as a release
gate.

## Install

Python 3.11-3.13 is supported.

```bash
pip install qualipilot
pip install "qualipilot[bedrock]"       # AWS Bedrock
pip install "qualipilot[ollama]"        # Ollama
pip install "qualipilot[openai]"        # OpenAI-compatible endpoint
pip install "qualipilot[dask]"          # Dask engine
pip install "qualipilot[duckdb]"        # DuckDB engine
pip install "qualipilot[gz]"            # GroundZero managed LLM connections
pip install "qualipilot[linking]"       # probabilistic linkage
pip install "qualipilot[spark]"         # Spark engine
```

From a source checkout, `./install.sh` and `.\install.ps1` create a local
virtual environment. Pass `--dev` or `-Dev` for editable development
installation. The `all` extra includes Spark; install only the extras you
use when possible.

## CLI

```bash
qualipilot check data.csv \
  --engine polars \
  --range amount=0,100000 \
  --output reports/data.quality.html \
  --fail-on warn
```

`--output` supports `.json`, `.html`, and `.md`. `--fail-on` returns a
nonzero exit code when a result reaches the selected severity, which makes
the command suitable for CI gates. Run `qualipilot check --help` for the
complete option set.

Configuration can also be stored in YAML or JSON:

```bash
qualipilot check examples/sample.csv --config examples/config.yaml
```

See [examples/config.yaml](https://github.com/Sarvesh-GanesanW/qualipilot/blob/main/examples/config.yaml)
for the configuration model. CLI-only controls such as `--fail-on` are not
configuration fields. The sample deliberately contains range and freshness
failures, so this command demonstrates the default nonzero quality gate.

## Python

```python
import pandas as pd

from qualipilot import DataQualityChecker, QualipilotConfig
from qualipilot.models.config import CheckConfig, ColumnRange

frame = pd.read_csv("orders.csv")
config = QualipilotConfig(
    engine="polars",
    checks=CheckConfig(
        column_ranges={"amount": ColumnRange(min=0, max=100_000)}
    ),
)

with DataQualityChecker(frame, config) as checker:
    report = checker.run()
print(report.to_json())
```

The context manager releases engine-owned resources such as DuckDB
connections. Dataframes and externally supplied Spark sessions remain owned
by the caller.

### GroundZero runtimes

The Spark and DuckDB runtime sessions expose the same thin adapter. Pass the
name of a managed LLM connection; Qualipilot selects its provider from that
connection's type and returns a `QualityReport`:

```python
from GZ.SparkUtils import sparkSession

spark = sparkSession("testapp", "FATAL")
df = spark.executeSnowflake("SourceSnowflake", "SELECT * FROM orders")
report = spark.runDataQualityChecks(
    df=df,
    connectionName="TestDataQuality",
)

# These variants load data through the same underlying Spark session.
table_report = spark.runTableDataQualityChecks(
    tableName="analytics.orders",
    connectionName="TestDataQuality",
)
query_report = spark.runQueryDataQualityChecks(
    query="SELECT * FROM analytics.orders WHERE status = 'open'",
    connectionName="TestDataQuality",
)

spark.saveDataQualityReport(report, "reports/orders.json")
```

The runtime adapter passes its underlying Spark session to Qualipilot. The
DuckDB adapter exposes the same methods and passes its existing DuckDB
connection, so neither runtime creates or closes a caller-owned session.

The direct API accepts the same connection name:

```python
from qualipilot import DataQualityChecker, LLMConfig, QualipilotConfig

report = DataQualityChecker(
    df,
    QualipilotConfig(llm=LLMConfig(connection_name="TestDataQuality")),
).run()
```

## Checks

| Check | Default | Purpose |
|---|---:|---|
| `missing_values` | on | null counts and percentages |
| `duplicates` | on | duplicate rows, optionally over a subset |
| `data_types` | on | column dtype inventory |
| `outliers` | on | numeric IQR outliers |
| `ranges` | on | configured numeric bounds |
| `cardinality` | on | distinct counts and optional top values |
| `freshness` | off | timestamp age and future timestamps |
| `linkage` | off | configured probabilistic duplicate detection |

Each check produces a `CheckResult` with an `ok`, `warn`, or `error`
severity, execution status, duration, and JSON-safe payload.

## Engines

| Engine | Extra | Notes |
|---|---|---|
| Polars | core | default for paths and ordinary in-memory frames |
| Pandas | core | explicit pandas execution |
| DuckDB | `duckdb` | SQL-backed execution |
| Dask | `dask` | partitioned dataframe execution |
| Spark | `spark` | requires a working Java/Spark environment |

`engine="auto"` selects by input type; paths and pandas frames currently
resolve to Polars. Backend parity is covered by tests, but memory use and
runtime depend on file format, check mix, and data distribution. Use the
scripts in `scripts/` to measure your own workload.

## Optional LLM reporting

Available direct providers are `bedrock`, `ollama`, and `openai` (for
compatible Chat Completions endpoints). `LLMConfig(connection_name="...")`
selects `gz` automatically and resolves the managed connection's actual
provider at call time. The provider receives a compact summary of the quality
report: column names and dtypes, aggregate check metrics, and check execution
status. Input paths, source versions, exception messages, row samples, and top
values are excluded. Keep the default `none` for fully local checks.

Bedrock requires an explicit, currently available model ID:

```bash
qualipilot check data.csv \
  --llm bedrock \
  --model "$BEDROCK_MODEL_ID" \
  --region us-east-1
```

The caller needs `bedrock:InvokeModel` for the selected foundation model or
inference profile. Model availability and identifiers vary by account and
region; do not bake a sample ID into long-lived configuration.

For a local Ollama example:

```bash
mkdir -p reports
export HOST_UID="$(id -u)" HOST_GID="$(id -g)"
docker compose -f docker/docker-compose.yml build qualipilot
docker compose -f docker/docker-compose.yml run --rm qualipilot
docker compose -f docker/docker-compose.yml down
```

The compose stack binds Ollama only to `127.0.0.1`, pulls the configured
model before the check starts, and does not mount cloud credentials.

## Record linkage

Install the `linking` extra and provide explicit blocking and comparison
rules:

```bash
qualipilot link customers.csv \
  --id customer_id \
  --compare "name:fuzzy:0.92,0.75" \
  --compare "postcode:exact" \
  --block "postcode" \
  --threshold 0.9 \
  --survivor-sort "updated_at:desc" \
  --output reports/customers.linkage.json \
  --deduplicated-output customers.deduplicated.parquet
```

String match keys are normalized for Unicode, case, and whitespace by
default. The cleaned output contains one survivor per cluster, fills missing
fields from duplicate records, and is accompanied by lineage and a
metadata-only audit written last as a commit marker. Consumers should verify
its output SHA-256 before reading the cleaned file. Blocking, thresholds,
survivor ranking, and conflict resolution remain domain-specific. Review
[the linkage guide](https://github.com/Sarvesh-GanesanW/qualipilot/blob/main/docs/LINKING.md)
before using clusters operationally.

## Deployment and development

The repository includes locked Docker builds and a Terraform module for an
S3-triggered Lambda deployment. The module deploys an ECR image by digest,
limits Lambda reads to `incoming/`, writes reports below `reports/`, and
routes exhausted asynchronous invocations to SQS. Follow
[the deployment guide](https://github.com/Sarvesh-GanesanW/qualipilot/blob/main/docs/DEPLOY.md);
the first apply intentionally creates the ECR repository before Lambda.

```bash
./install.sh --dev
make check
```

CI runs Ruff, strict MyPy, tests with coverage, dependency audit, package
build/smoke tests, Terraform validation and mock-plan tests, installer
parsing, and container smoke tests.

Additional documentation:

- [Architecture](https://github.com/Sarvesh-GanesanW/qualipilot/blob/main/docs/ARCHITECTURE.md)
- [Deployment](https://github.com/Sarvesh-GanesanW/qualipilot/blob/main/docs/DEPLOY.md)
- [Migrating from 2.x](https://github.com/Sarvesh-GanesanW/qualipilot/blob/main/docs/MIGRATION.md)
- [Changelog](https://github.com/Sarvesh-GanesanW/qualipilot/blob/main/CHANGELOG.md)

## License

MIT.
