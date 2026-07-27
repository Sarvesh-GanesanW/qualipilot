# Architecture

Qualipilot separates orchestration, dataframe execution, checks, optional
LLM narration, and rendering. The same `QualityReport` model is returned by
the Python API, rendered by the CLI, and uploaded by the Lambda handler.

```text
CLI / Python / Lambda
          |
          v
 DataQualityChecker
    /     |      \
checks  engines   LLM provider (optional)
    \     |      /
      QualityReport
          |
   JSON / HTML / Markdown
```

## Responsibilities

| Package | Responsibility |
|---|---|
| `qualipilot.checker` | select enabled checks and assemble a report |
| `qualipilot.checks` | evaluate one quality concern through the engine API |
| `qualipilot.engines` | adapt Polars, Pandas, DuckDB, Dask, or Spark |
| `qualipilot.models` | validate configuration and define serialized results |
| `qualipilot.llm` | send a compact summary to an explicitly configured provider |
| `qualipilot.reporting` | render HTML and Markdown; JSON comes from Pydantic |
| `qualipilot.linking` | normalization, matching, clustering, and record consolidation |
| `qualipilot.lambda_handler` | validate S3 events, bound downloads, and upload reports |

Optional dependencies are imported at their boundary so the core package
does not require every dataframe engine or cloud SDK.

## Check flow

1. A path or dataframe and `QualipilotConfig` enter
   `DataQualityChecker`.
2. `build_engine` resolves `auto` from the input type or constructs the
   requested backend.
3. Enabled checks run sequentially through the common engine methods.
4. Each check returns a `CheckResult`. A check-level exception is represented
   as a failed result instead of discarding the entire report.
5. Dataset metadata and results are assembled into `QualityReport`, with a
   configuration fingerprint for grouping runs by configuration.
6. If configured, an LLM provider receives a compact summary and its text is
   attached to the report.
7. The caller renders, saves, or gates on the result severities.

The fingerprint is provenance metadata, not a unique dataset or run ID.
Generated timestamps and check durations mean complete report JSON is not
byte-for-byte deterministic.

## Runtime boundaries

The CLI validates option choices and maps `--fail-on` to a process exit code.
Configuration files do not contain this process-level gate.
Remote Dask or Spark source URLs must not contain userinfo, query tokens, or
fragments because downstream reader errors may expose their input URL; use
the platform's external credential configuration instead.

The Lambda handler accepts a direct `s3_uri` event or native S3
`ObjectCreated` records. Before downloading, it verifies the file extension
and `ContentLength` against `QUALIPILOT_MAX_INPUT_BYTES`. It rejects report
objects as inputs. Versioned reads bind to the exact S3 version; mutable
objects use a conditional `If-Match` read and a streaming byte cap. Before
dataframe construction, it also caps columns, nested values, Parquet
expansion, and estimated text expansion. It writes encrypted JSON below
`reports/`, then applies the `QUALIPILOT_FAIL_ON`
quality gate. A crossed data-quality threshold returns a failed gate outcome
and emits a dedicated metric after upload. Only processing and
check-execution faults fail the invocation and use the SQS failure
destination. Terminal reports use deterministic keys and are reused on
replay. Reports with check-execution or LLM failures use unique
`reports/failures/` keys so a later replay can recover.

The included Terraform module narrows S3 access to the configured input
prefix and report prefix. Bedrock access is absent unless exact model or
inference-profile ARNs are supplied. Although the provider calls Bedrock's
Converse operation, AWS authorizes that request with
`bedrock:InvokeModel`.

## Extension points

- Add a check by subclassing `Check`, implementing `_execute`, and adding it
  to `DataQualityChecker._build_check_list`.
- Add an engine by implementing the `Engine` interface and registering its
  resolution and construction in `qualipilot.engines`.
- Add an LLM provider by implementing `LLMProvider` and registering it in
  `qualipilot.llm.build_provider`.

An extension should add contract tests against at least one existing
implementation and preserve the strict, JSON-safe result models.
