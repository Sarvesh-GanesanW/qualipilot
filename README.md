# qualipilot

Production-grade data quality checker for Python. Runs structural and
statistical checks on any tabular dataset (CSV / Parquet / JSON / Pandas /
Polars / Dask / cuDF) and, optionally, asks an LLM — **AWS Bedrock**,
**Ollama**, or any OpenAI-compatible endpoint — to narrate the findings.

* swap engines with one flag (Polars default, Pandas/Dask/cuDF on demand)
* swap LLM providers the same way (`--llm bedrock|ollama|openai|none`)
* one-click install, docker-compose for local runs, terraform for Lambda
* typed Pydantic results, deterministic JSON output, exit-code severity
  gate for CI pipelines

---

## Install

### One-click (recommended)

```bash
# macOS / Linux
./install.sh --all        # core + every optional extra
./install.sh --bedrock    # core + boto3
./install.sh --dev        # editable + dev + pre-commit

# Windows PowerShell
.\install.ps1 -Extras all
```

### Manual

```bash
pip install qualipilot                 # core
pip install "qualipilot[bedrock]"      # + boto3 for AWS Bedrock
pip install "qualipilot[ollama]"       # + httpx (already core)
pip install "qualipilot[dask]"         # + dask[dataframe]
pip install "qualipilot[all]"          # everything except cuDF
```

cuDF (GPU) needs the RAPIDS conda channel — see
[docs.rapids.ai/install](https://docs.rapids.ai/install).

---

## Quickstart (CLI)

```bash
qualipilot check data.csv \
    --engine polars \
    --range amount=0,100000 \
    --output reports/data.quality.html \
    --llm bedrock \
    --model provider.model-3-5-haiku-20241022-v1:0 \
    --region us-east-1 \
    --fail-on warn
```

* `--output` can be `.json`, `.html`, or `.md`; format is inferred.
* `--fail-on {ok,warn,error}` decides when the CLI returns a non-zero
  exit code — wire it straight into CI.
* All flags have `--config` equivalents; see `examples/config.yaml`.

## Quickstart (Python)

```python
import pandas as pd
from qualipilot import DataQualityChecker, QualipilotConfig
from qualipilot.models.config import CheckConfig, ColumnRange, LLMConfig

df = pd.read_csv("orders.csv")

config = QualipilotConfig(
    engine="polars",
    checks=CheckConfig(
        column_ranges={"amount": ColumnRange(min=0, max=100_000)},
    ),
    llm=LLMConfig(
        provider="bedrock",
        model="provider.model-3-5-haiku-20241022-v1:0",
        region="us-east-1",
    ),
)

report = DataQualityChecker(df, config).run()
print(report.to_json())
print(report.llm_report)
```

---

## What it checks

| Check | Default | Description |
|---|---|---|
| `missing_values` | on  | per-column null counts + percentage |
| `duplicates`     | on  | global duplicate rows (subset-aware) |
| `data_types`     | on  | dtype rollup per column |
| `outliers`       | on  | IQR rule, Q1/Q3 computed in one pass |
| `ranges`         | on  | user-supplied `[min, max]` per column |
| `cardinality`    | on  | distinct count + top-10 values |
| `freshness`      | off | max-timestamp vs `freshness_max_age_hours` |

Each check returns a typed `CheckResult` with severity `ok / warn /
error`, a duration, a JSON-safe payload, and any captured exception.

---

## Engines

| Engine | When to use |
|---|---|
| `polars` (default) | in-memory data up to ~10 GB — 8× faster than pandas |
| `pandas`  | legacy integrations that need pandas-native output |
| `dask`    | larger-than-memory data or multi-worker clusters |
| `cudf`    | single-node GPU acceleration (RAPIDS required) |

`--engine auto` inspects the input object and picks the fastest safe
backend (Polars for single-node, Dask for already-Dask frames, cuDF
when a GPU frame is handed in).

---

## LLM providers

| Provider | `--llm` | Required |
|---|---|---|
| None (default) | `none` | nothing |
| AWS Bedrock (Converse API) | `bedrock` | `boto3`, IAM `bedrock:Converse` |
| Ollama | `ollama` | running ollama server |
| OpenAI-compatible | `openai` | base URL + API key |

Bedrock uses the **Converse API**, so the same code path works for
Provider Model, Meta Llama, Mistral, Cohere, etc. — you just change
`model=...`.

---

## Deploy

### Docker (local Ollama stack)

```bash
docker compose -f docker/docker-compose.yml up --build
```

This brings up `ollama` and a `qualipilot` container wired to it, and
runs the sample check end-to-end.

### AWS Lambda (container image)

```bash
cd deploy/terraform
terraform init
terraform apply -var project=qualipilot -var aws_profile=sre-tea

# build + push the image to the ECR repo terraform just made
aws ecr get-login-password | docker login --username AWS --password-stdin \
    $(terraform output -raw ecr_repository_url | cut -d/ -f1)
docker build -f ../../docker/Dockerfile.lambda -t qualipilot-lambda:latest ../..
docker tag qualipilot-lambda:latest "$(terraform output -raw ecr_repository_url):latest"
docker push "$(terraform output -raw ecr_repository_url):latest"

aws lambda update-function-code \
    --function-name qualipilot \
    --image-uri "$(terraform output -raw ecr_repository_url):latest"
```

Invoke with:

```bash
aws lambda invoke \
    --function-name qualipilot \
    --payload '{"s3_uri":"s3://my-bucket/events.parquet"}' \
    response.json
```

Report lands at `s3://my-bucket/reports/events.quality.json`.

---

## Development

```bash
./install.sh --dev
make lint typecheck test
```

* Ruff for lint + format, MyPy in strict mode, pytest with coverage.
* Pre-commit runs the same locally before every commit.
* `pytest -m integration` runs tests that need real AWS/Bedrock credentials.

---

## Record linkage / probabilistic dedup

Beyond exact duplicates, qualipilot ships an in-house Fellegi-Sunter
linker — no external splink dependency. Polars blocking, rapidfuzz
string distance, numpy EM. 1M rows in ~10 s on a laptop.

```bash
qualipilot link customers.csv \
    --id customer_id \
    --compare "name:fuzzy:0.92,0.75" \
    --compare "postcode:exact" \
    --block "postcode" \
    --threshold 0.9
```

Full details: [`docs/LINKING.md`](docs/LINKING.md).

## Docs

* [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — module layout + data flow
* [`docs/LINKING.md`](docs/LINKING.md) — probabilistic dedup / linkage
* [`docs/DEEP_DIVE.md`](docs/DEEP_DIVE.md) — audit of the v1 codebase
* [`docs/DEPLOY.md`](docs/DEPLOY.md) — cloud + on-prem deployment notes
* [`docs/MIGRATION.md`](docs/MIGRATION.md) — upgrading from v1.x
* [`docs/RUST_CONVERSION.md`](docs/RUST_CONVERSION.md) — should we port to Rust? (tldr: hybrid, not rewrite)

## License

MIT.
