# Architecture

## Goals

1. **Swap backends, not code.** A pipeline that starts on pandas
   should run unmodified on Polars, Dask, or GPU just by changing
   one config field.
2. **LLM is optional.** Everything except the narrative step runs
   with zero network dependencies.
3. **Cloud-native.** The same wheel powers a CLI, a Lambda container,
   and an ad-hoc notebook.

## Layered layout

```
        +-----------------------------+
        |          CLI / SDK          |        qualipilot.cli, __init__
        +--------------+--------------+
                       |
                       v
        +-----------------------------+
        |       DataQualityChecker    |        qualipilot.checker
        |  (orchestrator + reporter)  |
        +------+---------+-----+------+
               |         |     |
     +---------+   +-----+     +------+
     v             v                  v
+-----------+  +-----------+   +-----------+
|  Checks   |  |  Engines  |   |   LLM     |
+-----------+  +-----------+   +-----------+
| missing   |  | polars    |   | bedrock   |
| duplicates|  | pandas    |   | ollama    |
| types     |  | dask      |   | openai    |
| outliers  |  | cudf      |   | null      |
| ranges    |  +-----------+   +-----------+
| cardinal. |
| freshness |
+-----------+
```

### Module responsibilities

| Module | Role |
|---|---|
| `qualipilot.engines` | thin wrapper around each dataframe lib; returns typed primitives (row count, null counts, quantiles, filters) |
| `qualipilot.checks`  | declarative checks consuming the `Engine` protocol; independent of the underlying backend |
| `qualipilot.llm`     | pluggable provider; each concrete class imports its SDK lazily |
| `qualipilot.reporting` | turns `QualityReport` into JSON / HTML / Markdown |
| `qualipilot.checker` | glues the above, produces the immutable `QualityReport` |
| `qualipilot.cli`     | Typer app with config-file + env + flag merging |
| `qualipilot.lambda_handler` | S3-triggered Lambda entry point |

## Data flow

1. User hands a file path / DataFrame + `QualipilotConfig`.
2. `build_engine` picks an `Engine` (auto by default).
3. `DataQualityChecker` instantiates the enabled `Check` subclasses
   and runs each in sequence. Every check times itself, traps
   exceptions, and returns a `CheckResult`.
4. An aggregate `QualityReport` (Pydantic) is produced. A SHA-1
   fingerprint of the config is attached so identical runs can be
   deduped downstream.
5. If an LLM provider is configured, the report is summarised and
   submitted; the narrative lives on `report.llm_report`.
6. Based on CLI flags, the report is serialised to JSON / HTML /
   Markdown and the CLI exits with a code derived from the worst
   observed severity.

## Why Polars is the default

* Pandas → Polars delivers roughly an 8× speedup on groupbys at 10M
  rows and ~50% lower peak memory ([benchmark](https://tildalice.io/pandas-polars-dask-10m-rows-benchmark/)).
* The lazy API lets us fold Q1/Q3 for every numeric column into a
  single aggregation — the v1 codebase ran two compute calls per
  column, which was the largest hot path.
* Arrow interop gives us zero-copy conversions to/from pandas,
  DuckDB, and Parquet.

## Why AWS Bedrock via Converse

The Converse API is Amazon's unified conversation interface across
Provider, Meta, Mistral, and Cohere. We only write to one request
shape, so switching from Model Haiku to Llama 3 is a `model=...`
change, not a rewrite. It also automatically handles multi-turn
state, tool use, and image input if we need them later.

Reference: [boto3 bedrock-runtime.converse](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html).

## Extension points

* **New check** — subclass `qualipilot.checks.base.Check`, implement
  `_execute`, add it to the registry in `checker._build_check_list`.
* **New engine** — subclass `qualipilot.engines.base.Engine`, register
  in `qualipilot.engines.build_engine`. All checks get it for free.
* **New LLM provider** — subclass `qualipilot.llm.base.LLMProvider`
  and register in `qualipilot.llm.build_provider`.
