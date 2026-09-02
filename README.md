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

## Validation and benchmark matrix

These are point-in-time results from 2026-09-02, not compatibility promises or
service-level objectives. Release validation used Qualipilot v3.4.0 at commit
`a1bc30a`; the reproducible benchmark drivers and results use clean commit
`b17faeb`. The additional readiness gates and clean reruns below use commit
`db1ce77699f334af7258ebd7c1e8f4f654ce5436`. The full standards and
residual-risk assessment is in
[docs/ASSURANCE.md](https://github.com/Sarvesh-GanesanW/qualipilot/blob/main/docs/ASSURANCE.md).

| Scope | Result |
|---|---|
| v3.4.0 CI and release gate | 602 passed and 1 skipped in each non-Spark matrix environment; dedicated Spark 3.5.6 and 4.2.0 jobs each passed 17 tests. Ruff, formatting, strict MyPy, and the 70% coverage gate passed; the Ubuntu/Python 3.12 artifact recorded 87.31% line and 76.83% branch coverage |
| Post-release all-engine run | 573 passed; `tests/test_llm.py` was ignored and 18 additional LLM-named tests were deselected. Ruff, formatting, package MyPy, and explicit strict MyPy over seven benchmark drivers passed |
| Deterministic quality performance | 10/10 asserted cells passed: five engines across the 100-million-row range and 5-million-row full-check profiles |
| Persisted distributed quality | A local 100-million-row persisted-Parquet run passed for Dask multiprocessing and Spark standalone execution with exact range counts and recorded worker/executor evidence |
| Repeated-run resilience | All five engines passed 30-trial, 100,000-row full-check rehearsals with exact results and configured per-process high-water-memory thresholds |
| Record linkage | 12/12 input/backend cells passed; every warm-up and measured fit converged with `fit_status="ok"` and recovered all planted duplicates |
| NER | The pinned `en_core_web_sm` 3.8.0 pipeline passed fixed exact-span Few-NERD regression floors over 18,915 selected documents; five throughput trials also completed |
| Lambda operations | The fake-S3 performance benchmark passed; a separate actual-image RIE/MinIO gate passed boto3 download, an SSE-S3 request with returned `AES256` metadata, metadata-cache, provenance, and input-limit assertions under Docker limits |
| Groundzero adapter | 41 connector tests passed in the adjacent Spark runtime repository |

### Supported execution matrix

`N/A` means unsupported by design, not an untested or failed cell. `PASS`
means the benchmark executed and matched its expected assertions; the
synthetic datasets intentionally produce warn/error quality findings.

| Capability | Pandas | Polars | DuckDB | Dask | Spark |
|---|---:|---:|---:|---:|---:|
| Eight deterministic quality checks | PASS | PASS | PASS | PASS | PASS |
| Embedded dedup/linkage input | PASS; converted to Polars | PASS; native | N/A | N/A | N/A |
| Direct linkage input | PASS; converted to Polars | PASS; native | N/A | N/A | N/A |
| Linkage compute backend | N/A | PASS | PASS | N/A | N/A |
| Lambda dataframe engine | N/A | PASS | N/A | N/A | N/A |

NER consumes strings through spaCy and is independent of dataframe engines.
`engine="auto"` is a selector, not a sixth engine. Lambda accepts `auto` or
Polars, does not include NER, and rejects probabilistic linkage.

### Data-quality schemas

The comparable scale profile generated 100,000,000 rows with four non-null
`int64` columns. Pandas and Polars materialized a 2.98 GiB raw frame; DuckDB,
Dask, and Spark used generated lazy sources.

| Column | Expression | Inclusive rule | Exact violations |
|---|---|---:|---:|
| `id` | row index, 0 through 99,999,999 | contract only | 0 |
| `amount` | `id % 1000` | 0–899 | 10,000,000 |
| `quantity` | `id % 100` | 10–89 | 20,000,000 |
| `score` | `id % 200` | 20–179 | 20,000,000 |

Only dataset contract and the three range rules were enabled. There was no
file or network I/O, cache, sampling, top-value collection, or LLM call.

The full profile generated 5,000,000 rows and enabled dataset contract,
missing values, exact duplicates, dtype inventory, IQR outliers, ranges,
cardinality, and freshness.

| Column | Portable dtype | Expression and purpose |
|---|---|---|
| `id` | integer | row index |
| `entity_id` | integer | `id % 500000`; duplicate key, so all 5,000,000 rows are duplicate members |
| `amount` | float | null for the first 10 rows of every 1,000-row cycle, otherwise `id % 1000`; 50,000 nulls and 500,000 range violations |
| `quantity` | integer | `id % 100`; 1,000,000 range violations |
| `score` | float | 10,000 every 1,000th row, otherwise `id % 200`; 5,000 injected spikes and 1,000,000 range violations |
| `event_time` | UTC datetime | run start minus `id % 48` hours, plus two hours every 1,000th row; exercises age and future-time handling |

### Data-quality performance

The host was an AMD Ryzen 7 7700X with 16 logical CPUs and 30.53 GiB RAM,
running Python 3.12.13, Polars 1.44.1, Pandas 2.3.3, DuckDB 1.5.5, Dask
2026.8.0, Spark 4.0.0, and Java 21.0.11. Polars, DuckDB, Dask, and Spark were
configured for eight execution threads/slots; Dask and Spark used 64
partitions. Pandas used its native single-process execution. Each engine ran
in a separate process: first call, one discarded warm-up, then five measured
trials.

100-million-row range profile:

| Engine | Source | Setup/build or plan (s) | First call (s) | Median (s) | Logical M rows/s | Process high-water RSS |
|---|---|---:|---:|---:|---:|---:|
| Pandas | eager | 1.242 | 0.737 | 0.406 | 246.2 | 3.59 GiB |
| Polars | eager | 0.384 | 0.485 | 0.447 | 223.9 | 5.56 GiB |
| DuckDB | lazy generated SQL | 1.450 | 1.218 | 1.220 | 81.9 | 3.37 GiB |
| Dask | 64 delayed partitions | 0.352 | 0.564 | 0.567 | 176.4 | 0.74 GiB |
| Spark | `spark.range`, 64 partitions | 2.531 | 1.102 | 0.211 | 472.9 | 0.20 GiB Python + 0.91 GiB JVM |

Five-million-row full-check profile:

| Engine | IQR quantiles | Setup/build or plan (s) | First call (s) | Median (s) | Logical M rows/s | Process high-water RSS |
|---|---|---:|---:|---:|---:|---:|
| Pandas | exact | 0.272 | 0.475 | 0.473 | 10.58 | 1.06 GiB |
| Polars | exact | 0.084 | 0.185 | 0.176 | 28.43 | 0.99 GiB |
| DuckDB | exact | 0.333 | 0.763 | 0.728 | 6.87 | 1.89 GiB |
| Dask | approximate | 0.313 | 2.435 | 2.341 | 2.14 | 0.82 GiB |
| Spark | approximate, 0.001 relative error | 2.569 | 5.227 | 2.549 | 1.96 | 0.20 GiB Python + 3.24 GiB JVM |

Setup includes the selected engine import and runtime startup. The Qualipilot
module import occurs after setup and is not timed; checker initialization is
recorded separately in the JSON output and excluded from check times.
High-water RSS covers the process lifetime, including all imports and setup.
Logical rows/s is dataset rows divided by wall time, not physical scan
throughput: the full profile performs multiple engine actions. Eager and lazy
sources have different materialization costs, and the lazy sources were not
cached. These synthetic, CPU-local results are a correctness-backed workload
comparison, not an engine ranking or production SLA.

The generated-source 100-million-row matrix used Dask's threaded scheduler
and Spark `local[8]`; those timings do not establish a worker-process,
executor, or multi-host boundary. Persisted distributed execution was tested
separately below.

### Persisted distributed execution

A separate correctness gate read the same 100,000,000-row, four-column
`int64` schema from 64 local Snappy-Parquet files totaling 463,864,911 bytes.
Dataset generation was excluded and filesystem cache state was uncontrolled.

| Engine | Observed execution boundary | Quality wall time | Exact range violations | Result |
|---|---|---:|---|---:|
| Dask | two spawned child processes per checker action | 43.228 s | 10,000,000 / 20,000,000 / 20,000,000 | PASS |
| Spark | two standalone worker/executor JVMs; executor IDs `0` and `1`; 142 quality tasks | 32.591 s | 10,000,000 / 20,000,000 / 20,000,000 | PASS |

Both runs used one physical host and local disk. They establish persisted-data
and process/executor integration on this fixture, not multi-host execution,
remote-object-store behavior, network resilience, production capacity,
availability, or a service-level objective. The Spark driver also supports an
existing non-local master and a configurable minimum distinct-host assertion
for deployment-owned tests.

### Repeated-run resilience

A clean local rehearsal ran the 100,000-row full-check profile for 30 measured
trials after a cold call and warm-up. Every trial retained the expected check
results. Non-Spark processes used 2,048 MiB peak and 512 MiB post-warm-up HWM
growth thresholds; Spark used 4,096 MiB and 2,048 MiB for each observed
process. These are acceptance checks evaluated between completed trials, not
hard cgroup memory caps.

| Engine | Max/median time | Python peak / growth (MiB) | Spark driver JVM peak / growth (MiB) | Result |
|---|---:|---:|---:|---:|
| Pandas | 1.058 | 181.613 / 0.164 | — | PASS |
| Polars | 1.096 | 198.293 / 0.109 | — | PASS |
| DuckDB | 1.173 | 307.668 / 32.066 | — | PASS |
| Dask | 1.111 | 202.059 / 3.715 | — | PASS |
| Spark | 1.348 | 200.465 / 0.172 | 1,865.234 / 327.480 | PASS |

The time ratio was recorded but had no failure threshold. Memory is Linux
`/proc` high-water evidence for the observed parent/driver processes, not
aggregate host, Dask-worker, Spark-executor, or cgroup memory. The suite also
covers conditional-write retry, size/expansion rejection, failed-check
reporting, input-version races, atomic output restoration, and interruption
from an injected `KeyboardInterrupt` at the atomic-replacement boundary. These
short synthetic runs and injected failures are regression sentinels, not
long-duration leak, kernel-OOM recovery, multi-host failover, or
production-capacity evidence.

### Record-linkage schema and performance

The deterministic seed-7 dataset used the following schema. One percent of
base rows received a second record with one selected name character set to
`q` and the same postcode/DOB.

| Column | Type | Generation |
|---|---|---|
| `id` | integer | unique base and duplicate-record identifier |
| `name` | string | 12 lowercase ASCII characters; planted duplicate sets one selected character to `q` |
| `postcode` | string | base value is zero-padded `PC` plus `id % 2000`; planted duplicate copies its source value; used for blocking |
| `dob` | integer | seeded uniform year from 1950 through 2004 |

The timed region includes `RecordLinker` construction and execution, so Pandas
normalization is included. Each cell used one discarded warm-up and three
measured trials. Every cell asserted candidates, matched pairs, cluster count,
full membership, and 100% recovery of the planted duplicates.

| Base rows | Input | Compute | Candidates | Planted recall | Median (ms) |
|---:|---|---|---:|---:|---:|
| 5,000 | Pandas | Polars | 4,133 | 50/50 | 34.4 |
| 5,000 | Pandas | DuckDB | 4,133 | 50/50 | 43.5 |
| 5,000 | Polars | Polars | 4,133 | 50/50 | 34.8 |
| 5,000 | Polars | DuckDB | 4,133 | 50/50 | 42.7 |
| 25,000 | Pandas | Polars | 147,145 | 250/250 | 734.6 |
| 25,000 | Pandas | DuckDB | 147,145 | 250/250 | 717.0 |
| 25,000 | Polars | Polars | 147,145 | 250/250 | 738.9 |
| 25,000 | Polars | DuckDB | 147,145 | 250/250 | 716.6 |
| 100,000 | Pandas | Polars | 2,500,242 | 1,000/1,000 | 1,943.3 |
| 100,000 | Pandas | DuckDB | 2,500,242 | 1,000/1,000 | 1,174.3 |
| 100,000 | Polars | Polars | 2,500,242 | 1,000/1,000 | 1,978.0 |
| 100,000 | Polars | DuckDB | 2,500,242 | 1,000/1,000 | 1,151.2 |

The former default of 15 EM iterations was too small for the deterministic
5,000- and 25,000-row comparison-level distributions. The default is now 100
iterations at the existing `1e-3` tolerance. Focused regressions require both
distributions to converge after more than 15 but within 100 iterations. Every
warm-up and measured cell above returned `fit_status="ok"` without a fit
warning. This does not guarantee convergence on other datasets; diagnostics
remain mandatory and consolidation still rejects warning or rejected fits by
default.

This planted synthetic evaluation validates the pipeline and scaling path; it
does not establish accuracy, calibration, fairness, or a universal threshold
for real identities.

### NER quality and performance

The gate uses Few-NERD's CC BY-SA 4.0 supervised test distribution at revision
`205f3e9c9f3577ea2561d43f2f62dc249ab92d5b`, with source SHA-256
`b7ad746fcbeb68fcc235ba7142d7c3723ea2dc39930089e947284defecf300c6`.

| Source field | Type | Use |
|---|---|---|
| `id` | string | stable source-document identifier |
| `tokens` | list of strings | text reconstructed with one ASCII space between tokens |
| `fine_ner_tags` | list of integers | token-aligned reference labels |

Only person, organization, and location-GPE references are mapped to spaCy's
`PERSON`, `ORG`, and `GPE` labels. Entity-free sentences are retained;
sentences containing an incompatible entity label are excluded. The pinned
selection hash is
`ede4ba2d39f35c9a0843d803c65cddd68b794177a807a065b079f266a08a704c`.
This selected 18,915 of 37,648 documents with 35,396 reference entities.
Matching requires the exact reconstructed character span and mapped label.

| Metric | Observed | Regression floor |
|---|---:|---:|
| Micro precision | 0.552923 | 0.50 |
| Micro recall | 0.495875 | 0.45 |
| Micro F1 | 0.522848 | 0.47 |
| PERSON F1 | 0.671000 | 0.62 |
| ORG F1 | 0.331101 | 0.28 |
| GPE F1 | 0.581485 | 0.52 |

`en_core_web_sm` 3.8.0 under spaCy 3.8.16 passed every floor. CI pins the
model wheel download by SHA-256. The audit also records a deterministic digest
of installed model source/data files; its declared scope excludes
`__pycache__` and `.pyc`, so it is a drift sentinel rather than code-signing
or execution-authenticity evidence. Five 5,000-document throughput trials had
a 419.252 documents/second median and 904.2 MiB process high-water RSS.

These are fixed cross-domain regression floors over a label-filtered,
single-space reconstruction, not production accuracy, calibration, fairness,
language coverage, or fitness thresholds. ORG F1 is only 0.331101. Callers
must pin and validate their own model wheel and representative
domain/language holdout before production use.

### Lambda-handler schema and performance

The local handler workload was a 5,000,000-row, 69.46 MiB CSV with three
integer columns: `id` as row index, `amount = id % 1000`, and
`quantity = id % 100`. Seven checks ran with duplicate subset
`[amount, quantity]` and range rules of 0–899 and 10–89 respectively.

The real Lambda handler and real Polars checker ran in a fresh child process
with eight Polars workers. Module import took 0.281 seconds, the first handler
call after import took 3.243 seconds, and five post-warm-up trials had a 3.369
second median (1.484 million logical rows/second). Worker-lifetime high-water
RSS was 0.96 GiB; input generation ran in the parent and was excluded.

That 5-million-row timing remains a fake-S3, OS-cache-sensitive process
benchmark and is not an AWS Lambda or S3 measurement.

Separately, `scripts/check_lambda_container.py` exercised a revision-labeled
image built from `docker/Dockerfile.lambda`. In CI, the import smoke, RIE gate,
and scanner target the same image. The gate invoked the default handler through
Lambda RIE on an internal Docker bridge. The container ran non-root with a
read-only root filesystem, 1 CPU, 1 GiB memory, equal memory/swap limits, and a
256-PID limit. Boto3 read from pinned MinIO, the real Polars checker completed
all seven enabled checks, the report was read back with `AES256` server-side-
encryption metadata, and a second invocation reused the stored idempotency
metadata. A 1,025-byte object was rejected against a 1,024-byte cap before
download and produced no report.

The clean observed first request after RIE readiness was 1.060 seconds, the
cached request 0.009 seconds, and the rejected request 0.008 seconds; the
container memory snapshot was 128.7 MiB. These are diagnostic local timings,
not Lambda cold starts. Fake credentials and a service-specific MinIO endpoint
were isolated on a bridge with no external route; no external AWS service or
user AWS account was contacted. RIE and MinIO do not establish AWS IAM,
S3/KMS/TLS, CloudWatch, SQS destination, concurrency, managed-service failure,
or availability behavior.

Reproduce the benchmark matrix from a Spark-enabled environment:

```bash
for engine in pandas polars duckdb dask spark; do
  python scripts/bench_checker.py \
    --engine "$engine" --profile ranges \
    --output "reports/ranges-$engine.json"
  python scripts/bench_checker.py \
    --engine "$engine" --profile full \
    --output "reports/full-$engine.json"
done

python scripts/bench_linking.py --trials 3 \
  --output reports/linking.json
python scripts/bench_ner.py --model en_core_web_sm --docs 5000 --trials 5 \
  --output reports/ner.json
python scripts/bench_lambda.py --rows 5000000 --threads 8 --trials 5 \
  --output reports/lambda.json

python scripts/bench_distributed.py --engine dask \
  --rows 100000000 --partitions 64 --workers 2 \
  --data reports/distributed-input.parquet \
  --output reports/distributed-dask.json
python scripts/bench_distributed.py --engine spark \
  --rows 100000000 --partitions 64 --workers 2 \
  --data reports/distributed-input.parquet \
  --output reports/distributed-spark.json

for engine in pandas polars duckdb dask; do
  python scripts/bench_checker.py \
    --engine "$engine" --profile full --rows 100000 \
    --partitions 8 --threads 2 --trials 30 \
    --max-process-hwm-mib 2048 --max-hwm-growth-mib 512 \
    --output "reports/resilience-$engine.json"
done
python scripts/bench_checker.py \
  --engine spark --profile full --rows 100000 \
  --partitions 8 --threads 2 --trials 30 \
  --max-process-hwm-mib 4096 --max-hwm-growth-mib 2048 \
  --output reports/resilience-spark.json

python scripts/check_lambda_container.py \
  --output reports/lambda-rie.json
```

Reproduce the non-LLM verification:

```bash
python -m pytest -q --ignore=tests/test_llm.py -k 'not llm' --no-cov
ruff check .
ruff format --check .
mypy src/qualipilot
terraform -chdir=deploy/terraform init -backend=false -lockfile=readonly
terraform -chdir=deploy/terraform validate
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
