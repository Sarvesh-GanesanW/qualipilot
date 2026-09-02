# qualipilot

Qualipilot runs configurable data-quality checks, probabilistic record
linkage, and named-entity extraction over tabular data. Quality checks return
typed JSON, HTML, or Markdown reports. CSV, Parquet, JSONL, and NDJSON are
supported across several dataframe backends. Columns must contain scalar
values; flatten nested arrays, maps, and objects before checking.
LLM-generated narrative is optional and disabled by default.

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
pip install "qualipilot[gz]"            # Groundzero managed LLM connections
pip install "qualipilot[linking]"       # probabilistic linkage
pip install "qualipilot[ner]"           # spaCy NER adapter
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

### Groundzero runtimes

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

Use `checks.severity_overrides` to replace the severity of a built-in check
when it finds a problem, for example `{"missing_values": "error", "ranges":
"warn"}`. Overrides do not turn successful checks into failures and cannot
hide execution failures. Outlier payloads identify whether their quantiles
are exact or approximate; Spark also reports its `0.001` relative error.

`expected_dtypes` accepts portable families: `integer`, `float`, `decimal`,
`numeric`, `boolean`, `string`, `categorical`, `binary`, `date`, `datetime`,
`time`, and `duration`. Native engine dtype names remain supported with a
case-insensitive exact match.

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

## Validation and benchmarks

These are point-in-time results from 2026-09-02, not compatibility promises or
service-level objectives. The full standards and residual-risk assessment is
in [docs/ASSURANCE.md](https://github.com/Sarvesh-GanesanW/qualipilot/blob/main/docs/ASSURANCE.md).

| Scope | Result |
|---|---|
| Full local quality gate | 588 passed, 1 Spark test skipped in the non-Spark environment; Ruff, formatting, strict MyPy, and 84.29% branch coverage passed |
| All five engines | 281 focused engine/check tests passed with Polars 1.44.1, Pandas 2.3.3, DuckDB 1.5.5, Dask 2026.8.0, and Spark 4.0.0 |
| Record linkage and consolidation | 118 focused tests passed across Polars and DuckDB |
| NER | 10 focused tests passed, including batching, offsets, label validation, and audit provenance |
| Lambda handler | 43 focused tests passed; Terraform format/validation and 21 module tests passed |
| Groundzero adapter | 41 connector tests passed in the adjacent Spark runtime repository |

### Spark 100-million-row benchmark

From a source checkout, run the checked benchmark in a Spark-enabled
environment:

```bash
python scripts/bench_spark.py
```

The recorded run used a Ryzen 7 7700X host with 30.53 GiB RAM, Python 3.12.13,
Java 21.0.11, Spark 4.0.0, `local[8]`, and 64 partitions. `spark.range`
generated 100,000,000 rows and four columns lazily. Only the dataset contract
and three range rules were enabled; there was no file I/O, cache, sampling, or
LLM call. Every run asserted one row-count action and one batched range
aggregation, with exact violation counts of 10,000,000, 20,000,000, and
20,000,000.

The cold quality-check wall time was 1.253 seconds. Five measured warm trials
were 0.286, 0.248, 0.209, 0.184, and 0.200 seconds (median 0.209 seconds).
After the cold run, one warm-up, and five trials, Linux reported a 1.02 GiB
Spark JVM high-water RSS and 198 MiB Python high-water RSS. These unusually
fast figures measure a synthetic, CPU-local aggregate with no storage or
network I/O; they are not a 100-million-row production SLA.

### Statistical smoke evaluations

On the FEBRL4 reference corpus (5,000 left records, 5,000 right records, and
5,000 known links), postcode/DOB/SSN blocking reduced 25 million possible
pairs to 30,021 candidates. The model compared fuzzy given name, surname, and
suburb plus exact street number and state. At a 0.9 threshold, both linkage
backends produced 4,905 true positives, no false positives, and 95 false
negatives: precision 1.000, recall 0.981, and F1 0.9904. Polars took 94 ms and
DuckDB 201 ms after data loading. This validates the evaluation path on one
reference corpus, not a universal threshold or accuracy claim.

An `en_core_web_sm` 3.8.0 interface smoke used eight hand-labeled documents
with 17 spans: 14 true positives, three false positives, and three false
negatives (precision/recall/F1 0.8235). A 5,000-short-document batching run
processed 1,042 documents/second. The sample is intentionally too small for a
model-quality claim; production NER requires a versioned, representative
domain holdout and per-label gates.

### Lambda container smoke and capacity run

The final Lambda image passed real-handler/fake-S3 processing, cached replay,
Runtime Interface Emulator startup, dependency checks, and a Trivy 0.70.0 scan
with zero known fixed HIGH or CRITICAL findings. A local 2 GiB cgroup run
processed a synthetic 5,000,000-row, 87.8 MB CSV through seven checks in 4.328
seconds, peaking at 1.70 GB cgroup memory. This is a local upper-envelope test,
not AWS cold-start, concurrency, billing, or availability evidence; select
Lambda memory and input limits from representative data.

Reproduce the automated portions with:

```bash
make check
python -m pytest -q tests/test_lambda_handler.py
terraform -chdir=deploy/terraform init -backend=false -lockfile=readonly
terraform -chdir=deploy/terraform test
```

## Optional LLM reporting

Available direct providers are `bedrock`, `ollama`, and `openai` (for
compatible Chat Completions and Responses endpoints).
`LLMConfig(connection_name="...")` selects `gz` automatically and resolves
the managed connection's actual provider at call time. Managed Anthropic, AWS
Bedrock, Azure OpenAI, Cohere, Fireworks AI, Gemini, Hugging Face, OpenAI,
Together AI, and xAI connections are supported, including complete Chat
Completions and Responses endpoint URLs. The provider receives a compact
summary of the quality report: column names and dtypes, aggregate check
metrics, and check execution status. Input paths, source versions, exception
messages, row samples, and top values are excluded. Keep the default `none`
for fully local checks.
Treat the generated narrative as untrusted advisory output: validate it before
acting, never execute instructions it contains, and never put secrets in a
custom `system_prompt`.

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
Linkage audits contain record identifiers, matched pairs, clusters, and
lineage; protect them like the source dataset and apply explicit access and
retention controls.

## Named-entity recognition

NER is a separate, optional local pipeline rather than a data-quality check.
Install spaCy through the `ner` extra and install a pipeline that matches your
language and domain. Qualipilot never downloads or silently selects a model:

```bash
pip install "qualipilot[ner]"
python -m spacy download en_core_web_sm  # convenient for local evaluation

qualipilot ner customer_notes.parquet \
  --text note \
  --id customer_id \
  --model en_core_web_sm \
  --output reports/customer-notes.entities.json
```

The audit contains source and model provenance, row identifiers, labels,
entity text, and half-open character offsets. The Python API batches documents
through `SpacyEntityRecognizer.extract_many()`. Statistical NER is
domain-dependent, so evaluate the pinned pipeline against representative
labeled data before using its output. See the
[NER guide](https://github.com/Sarvesh-GanesanW/qualipilot/blob/main/docs/NER.md).

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
- [Assurance and standards mapping](https://github.com/Sarvesh-GanesanW/qualipilot/blob/main/docs/ASSURANCE.md)
- [Deployment](https://github.com/Sarvesh-GanesanW/qualipilot/blob/main/docs/DEPLOY.md)
- [Named-entity recognition](https://github.com/Sarvesh-GanesanW/qualipilot/blob/main/docs/NER.md)
- [Migrating from 2.x](https://github.com/Sarvesh-GanesanW/qualipilot/blob/main/docs/MIGRATION.md)
- [Changelog](https://github.com/Sarvesh-GanesanW/qualipilot/blob/main/CHANGELOG.md)

## License

MIT.
