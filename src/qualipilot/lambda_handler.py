"""AWS Lambda entry point for objects stored in S3.

The handler accepts either a direct invocation::

    {"s3_uri": "s3://bucket/key.parquet", "config": {}}

or a native S3 notification containing one or more ``Records``. Reports are
always written below ``reports/`` in the input bucket.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, unquote_plus, urlparse
from uuid import uuid4

from qualipilot.checker import DataQualityChecker, config_fingerprint
from qualipilot.engines._file_formats import (
    require_unique_csv_columns,
    require_valid_json_lines,
)
from qualipilot.logging_setup import configure_logging
from qualipilot.models.config import QualipilotConfig

configure_logging(
    level=os.environ.get("QUALIPILOT_LOG_LEVEL", "INFO"),
    json_logs=True,
)
logger = logging.getLogger(__name__)

_SUPPORTED_SUFFIXES = {".csv", ".jsonl", ".ndjson", ".parquet", ".pq"}
_SEVERITY_RANK = {"ok": 0, "warn": 1, "error": 2}
_FAIL_ON_VALUES = {"none", "warn", "error"}
_DEFAULT_MAX_INPUT_BYTES = 256 * 1024 * 1024
_DEFAULT_MAX_DATASET_BYTES = 1024 * 1024 * 1024
_MAX_COLUMNS = 10_000
_TEXT_MEMORY_FACTOR = 8
_PACKAGE_VERSION = version("qualipilot")
_S3_CLIENT: Any | None = None


def _s3_client() -> Any:
    """Reuse the SDK connection pool across warm Lambda invocations."""
    global _S3_CLIENT  # noqa: PLW0603
    if _S3_CLIENT is None:
        import boto3

        _S3_CLIENT = boto3.client("s3")
    return _S3_CLIENT


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Process a direct invocation or native S3 notification."""
    _ = context
    if not isinstance(event, dict):
        raise ValueError("event must be a JSON object")
    config = _lambda_config(event)
    remaining_time = getattr(context, "get_remaining_time_in_millis", None)
    if not callable(remaining_time):
        remaining_time = None
    fail_on = _fail_on(event.get("fail_on"))
    max_input_bytes = _max_input_bytes()
    max_dataset_bytes = _max_dataset_bytes()
    inputs, skipped = _supported_inputs(event)
    if "output_key" in event and len(inputs) != 1:
        raise ValueError("output_key requires exactly one input object")
    if not inputs:
        return {"processed": 0, "skipped": skipped, "results": []}

    s3 = _s3_client()
    if "s3_uri" in event:
        results = [
            _process_object(
                s3,
                *inputs[0],
                config=config,
                output_key=event.get("output_key"),
                max_input_bytes=max_input_bytes,
                max_dataset_bytes=max_dataset_bytes,
                remaining_time=remaining_time,
            )
        ]
    else:
        results = _process_notifications(
            s3,
            inputs,
            config=config,
            max_input_bytes=max_input_bytes,
            max_dataset_bytes=max_dataset_bytes,
            remaining_time=remaining_time,
        )
    return _invocation_result(event, results, skipped, fail_on)


def _lambda_config(event: dict[str, Any]) -> QualipilotConfig:
    raw = event["config"] if "config" in event else _environment_config()
    if not isinstance(raw, dict):
        raise ValueError("event.config must be a JSON object")
    config = QualipilotConfig(**raw)
    if config.output_path is not None:
        raise ValueError("config.output_path is not supported in Lambda")
    if config.report_format != "json":
        raise ValueError("Lambda supports only JSON reports")
    if config.engine not in {"auto", "polars"}:
        raise ValueError("Lambda supports only auto or polars engines")
    if config.checks.linkage is not None:
        raise ValueError("Lambda does not support probabilistic linkage")
    if config.llm.provider not in {"none", "bedrock"}:
        raise ValueError("Lambda supports only none or bedrock LLM providers")
    if (
        config.llm.api_key is not None
        or config.llm.aws_profile is not None
        or config.llm.allow_insecure_http
    ):
        raise ValueError("Lambda does not accept local API credentials")
    deployment_region = os.environ.get("AWS_REGION")
    if config.llm.provider == "bedrock" and deployment_region:
        config.llm = config.llm.model_copy(
            update={"region": deployment_region}
        )
    return config


def _supported_inputs(
    event: dict[str, Any],
) -> tuple[list[tuple[str, str, str | None]], int]:
    inputs = _event_inputs(event)
    if "s3_uri" in event:
        return inputs, 0
    supported = [
        item
        for item in inputs
        if PurePosixPath(item[1]).suffix.lower() in _SUPPORTED_SUFFIXES
    ]
    skipped = len(inputs) - len(supported)
    if skipped:
        logger.warning("skipping %d unsupported S3 object(s)", skipped)
    return supported, skipped


def _process_notifications(
    s3: Any,
    inputs: list[tuple[str, str, str | None]],
    *,
    config: QualipilotConfig,
    max_input_bytes: int,
    max_dataset_bytes: int,
    remaining_time: Callable[[], int] | None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    errors: list[Exception] = []
    for bucket, key, version_id in inputs:
        try:
            results.append(
                _process_object(
                    s3,
                    bucket,
                    key,
                    version_id,
                    config=config,
                    output_key=None,
                    max_input_bytes=max_input_bytes,
                    max_dataset_bytes=max_dataset_bytes,
                    remaining_time=remaining_time,
                )
            )
        except Exception as exc:
            logger.error("failed to process s3://%s/%s: %s", bucket, key, exc)
            errors.append(exc)
    if errors:
        raise ExceptionGroup("one or more S3 objects failed", errors)
    return results


def _invocation_result(
    event: dict[str, Any],
    results: list[dict[str, Any]],
    skipped: int,
    fail_on: str,
) -> dict[str, Any]:
    quality_failures, llm_failures, execution_failures = _outcome_counts(
        results,
        fail_on,
    )
    metric_results = (
        results
        if "s3_uri" in event
        else [result for result in results if not result["cached"]]
    )
    _emit_outcome_metrics(*_outcome_counts(metric_results, fail_on))
    if execution_failures:
        outputs = ", ".join(
            f"s3://{result['bucket']}/{result['output_key']}"
            for result in results
            if result["execution_failures"]
        )
        raise RuntimeError(
            f"quality checks failed to execute; reports: {outputs}"
        )
    quality_gate = (
        "disabled"
        if fail_on == "none"
        else "failed"
        if quality_failures
        else "passed"
    )
    outcome = {
        "quality_gate": quality_gate,
        "quality_failures": quality_failures,
        "llm_failures": llm_failures,
        "execution_failures": execution_failures,
    }
    if "s3_uri" in event:
        return {**results[0], **outcome}
    return {
        "processed": len(results),
        "skipped": skipped,
        "results": results,
        **outcome,
    }


def _outcome_counts(
    results: list[dict[str, Any]],
    fail_on: str,
) -> tuple[int, int, int]:
    quality_failures = sum(
        _meets_threshold(result["summary"], fail_on) for result in results
    )
    llm_failures = sum(result["llm_status"] == "failed" for result in results)
    execution_failures = sum(
        result["execution_failures"] for result in results
    )
    return quality_failures, llm_failures, execution_failures


def _emit_outcome_metrics(
    quality_failures: int,
    llm_failures: int,
    execution_failures: int,
) -> None:
    function_name = os.environ.get(
        "AWS_LAMBDA_FUNCTION_NAME", "qualipilot-local"
    )
    print(
        json.dumps(
            {
                "_aws": {
                    "Timestamp": int(time.time() * 1000),
                    "CloudWatchMetrics": [
                        {
                            "Namespace": "Qualipilot",
                            "Dimensions": [["FunctionName"]],
                            "Metrics": [
                                {
                                    "Name": "QualityGateFailures",
                                    "Unit": "Count",
                                },
                                {
                                    "Name": "LLMGenerationFailures",
                                    "Unit": "Count",
                                },
                                {
                                    "Name": "CheckExecutionFailures",
                                    "Unit": "Count",
                                },
                            ],
                        }
                    ],
                },
                "FunctionName": function_name,
                "QualityGateFailures": quality_failures,
                "LLMGenerationFailures": llm_failures,
                "CheckExecutionFailures": execution_failures,
            },
            separators=(",", ":"),
        )
    )


def _process_object(
    s3: Any,
    bucket: str,
    key: str,
    version_id: str | None,
    *,
    config: QualipilotConfig,
    output_key: Any,
    max_input_bytes: int,
    max_dataset_bytes: int,
    remaining_time: Callable[[], int] | None,
) -> dict[str, Any]:
    _validate_input(bucket, key)
    object_args = {"Bucket": bucket, "Key": key}
    if version_id is not None:
        object_args["VersionId"] = version_id
    metadata = s3.head_object(**object_args)
    content_length = metadata.get("ContentLength")
    if not isinstance(content_length, int) or content_length < 0:
        raise ValueError("S3 object did not return a valid ContentLength")
    if content_length > max_input_bytes:
        raise ValueError(
            f"s3://{bucket}/{key} is {content_length} bytes; "
            f"limit is {max_input_bytes} bytes"
        )
    effective_version, expected_etag, source_version = _resolve_source_version(
        metadata, version_id
    )
    report_identity = _report_identity(
        bucket,
        key,
        source_version,
        config,
    )
    target_key = (
        _validate_output_key(output_key)
        if output_key is not None
        else _derive_output_key(key, report_identity)
    )
    if target_key == key:
        raise ValueError("output_key must not overwrite the input object")
    # The cache prevents repeated work after a report has completed. Concurrent
    # invocations may still repeat checks before the conditional write wins.
    cached = _cached_report(
        s3,
        bucket,
        key,
        target_key,
        report_identity,
    )
    if cached is not None:
        logger.info("reusing report at s3://%s/%s", bucket, target_key)
        return cached

    with tempfile.TemporaryDirectory() as tmpdir:
        local = Path(tmpdir) / f"input{Path(key).suffix.lower()}"
        logger.info("downloading s3://%s/%s", bucket, key)
        if expected_etag is not None:
            _download_mutable_object(
                s3,
                bucket,
                key,
                local,
                etag=expected_etag,
                version_id=effective_version,
                content_length=content_length,
                max_input_bytes=max_input_bytes,
            )
        else:
            s3.download_file(
                bucket,
                key,
                str(local),
                ExtraArgs={"VersionId": effective_version},
            )
        _validate_local_input(local, max_dataset_bytes)
        runtime_config = config.model_copy(deep=True)
        with DataQualityChecker(
            local,
            runtime_config,
            source=f"s3://{bucket}/{key}",
            source_version=source_version,
        ) as checker:
            report = checker.run(include_llm=False)
            if runtime_config.llm.provider == "bedrock":
                if _set_bedrock_budget(runtime_config, remaining_time):
                    checker.enrich_with_llm(report)
                else:
                    report.llm_provider = runtime_config.llm.provider
                    report.llm_model = runtime_config.llm.model
                    report.llm_status = "failed"
                    report.llm_error = (
                        "TimeoutError: insufficient Lambda time "
                        "remaining for Bedrock"
                    )

    return _store_report(
        s3,
        bucket,
        key,
        target_key,
        report_identity,
        report,
    )


def _resolve_source_version(
    metadata: dict[str, Any],
    requested_version: str | None,
) -> tuple[str | None, str | None, str]:
    effective_version = metadata.get("VersionId") or requested_version
    if effective_version is not None and not isinstance(
        effective_version, str
    ):
        raise ValueError("S3 object did not return a valid VersionId")
    if effective_version not in {None, "null"}:
        if not effective_version.strip('"'):
            raise ValueError("S3 object did not return a valid VersionId")
        return effective_version, None, effective_version

    etag = metadata.get("ETag")
    if not isinstance(etag, str) or not etag.strip('"'):
        raise ValueError("S3 object did not return a valid ETag")
    return effective_version, etag, etag.strip('"')


def _download_mutable_object(
    s3: Any,
    bucket: str,
    key: str,
    target: Path,
    *,
    etag: str,
    version_id: str | None,
    content_length: int,
    max_input_bytes: int,
) -> None:
    request = {"Bucket": bucket, "Key": key, "IfMatch": etag}
    if version_id == "null":
        request["VersionId"] = version_id
    try:
        response = s3.get_object(**request)
    except Exception as exc:
        if _aws_error_code(exc) in {"412", "PreconditionFailed"}:
            raise RuntimeError(
                f"s3://{bucket}/{key} changed before download"
            ) from exc
        raise

    downloaded_bytes = _write_object_body(
        response,
        target,
        source=f"s3://{bucket}/{key}",
        etag=etag,
        content_length=content_length,
        max_input_bytes=max_input_bytes,
    )
    if downloaded_bytes != content_length:
        raise RuntimeError(
            f"s3://{bucket}/{key} ended after {downloaded_bytes} bytes; "
            f"expected {content_length} bytes"
        )


def _write_object_body(
    response: dict[str, Any],
    target: Path,
    *,
    source: str,
    etag: str,
    content_length: int,
    max_input_bytes: int,
) -> int:
    body = response.get("Body")
    read = getattr(body, "read", None)
    close = getattr(body, "close", None)
    if not callable(read) or not callable(close):
        if callable(close):
            close()
        raise ValueError("S3 object did not return a readable body")
    downloaded_bytes = 0
    try:
        if (
            response.get("ETag") != etag
            or response.get("ContentLength") != content_length
        ):
            raise RuntimeError(f"{source} changed before download")
        with target.open("wb") as output:
            while chunk := read(1024 * 1024):
                if not isinstance(chunk, bytes):
                    raise ValueError("S3 object returned a non-binary body")
                downloaded_bytes += len(chunk)
                if (
                    downloaded_bytes > content_length
                    or downloaded_bytes > max_input_bytes
                ):
                    raise ValueError(f"{source} exceeded its validated size")
                output.write(chunk)
    finally:
        close()
    return downloaded_bytes


def _store_report(
    s3: Any,
    bucket: str,
    input_key: str,
    output_key: str,
    identity: str,
    report: Any,
) -> dict[str, Any]:
    report_json = report.to_json().encode("utf-8")
    execution_failures = sum(
        item.status == "failed" for item in report.results
    )
    retryable = report.llm_status == "failed" or execution_failures > 0
    target_key = (
        f"reports/failures/{identity}.{uuid4().hex}.json"
        if retryable
        else output_key
    )
    result = {
        "bucket": bucket,
        "input_key": input_key,
        "output_key": target_key,
        "summary": {
            item.name: item.severity
            for item in report.results
            if item.status == "completed"
        },
        "llm_status": report.llm_status,
        "execution_failures": execution_failures,
        "cached": False,
    }
    put_args = {
        "Bucket": bucket,
        "Key": target_key,
        "Body": report_json,
        "ContentType": "application/json",
        "ServerSideEncryption": "AES256",
        "IfNoneMatch": "*",
        "Metadata": {
            "qualipilot-identity": identity,
            "qualipilot-summary": json.dumps(
                result["summary"],
                separators=(",", ":"),
            ),
            "qualipilot-llm-status": result["llm_status"],
            "qualipilot-execution-failures": str(result["execution_failures"]),
        },
    }
    for attempt in range(2):
        try:
            s3.put_object(**put_args)
            break
        except Exception as exc:
            code = _aws_error_code(exc)
            if code in {"409", "ConditionalRequestConflict"} and attempt == 0:
                continue
            if code not in {"412", "PreconditionFailed"}:
                raise
            winner = _cached_report(
                s3,
                bucket,
                input_key,
                target_key,
                identity,
            )
            if winner is None:  # pragma: no cover - S3 consistency guard
                raise RuntimeError(
                    "conditional report write lost without winner"
                ) from exc
            return winner
    logger.info("report written to s3://%s/%s", bucket, target_key)
    return result


def _cached_report(
    s3: Any,
    bucket: str,
    input_key: str,
    output_key: str,
    identity: str,
) -> dict[str, Any] | None:
    try:
        response = s3.head_object(Bucket=bucket, Key=output_key)
    except Exception as exc:
        if _aws_error_code(exc) in {
            "403",
            "404",
            "AccessDenied",
            "NoSuchKey",
            "NotFound",
        }:
            # S3 reports a missing key as 403 without ListBucket permission.
            # The conditional write still prevents an inaccessible collision.
            return None
        raise
    metadata = response.get("Metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("existing report has no idempotency metadata")
    if metadata.get("qualipilot-identity") != identity:
        raise ValueError("output_key already belongs to a different input")
    try:
        summary = json.loads(metadata["qualipilot-summary"])
        llm_status = metadata["qualipilot-llm-status"]
        execution_failures = int(metadata["qualipilot-execution-failures"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("existing report metadata is invalid") from exc
    if (
        not isinstance(summary, dict)
        or any(severity not in _SEVERITY_RANK for severity in summary.values())
        or llm_status not in {"disabled", "completed", "failed"}
        or execution_failures < 0
    ):
        raise RuntimeError("existing report summary is invalid")
    return {
        "bucket": bucket,
        "input_key": input_key,
        "output_key": output_key,
        "summary": summary,
        "llm_status": llm_status,
        "execution_failures": execution_failures,
        "cached": True,
    }


def _aws_error_code(exc: Exception) -> str | None:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return None
    error = response.get("Error")
    return str(error.get("Code")) if isinstance(error, dict) else None


def _event_inputs(event: dict[str, Any]) -> list[tuple[str, str, str | None]]:
    s3_uri = event.get("s3_uri")
    records = event.get("Records")
    if s3_uri is not None and records is not None:
        raise ValueError(
            "provide either event.s3_uri or event.Records, not both"
        )
    if s3_uri is not None:
        if not isinstance(s3_uri, str):
            raise ValueError("event.s3_uri must be a string")
        bucket, key = _parse_s3_uri(s3_uri)
        return [(bucket, key, None)]
    if not isinstance(records, list) or not records:
        raise ValueError(
            "event.s3_uri or a non-empty event.Records is required"
        )

    inputs: list[tuple[str, str, str | None]] = []
    for index, record in enumerate(records):
        try:
            event_name = record["eventName"]
            bucket = record["s3"]["bucket"]["name"]
            key = unquote_plus(record["s3"]["object"]["key"])
            version_id = record["s3"]["object"].get("versionId")
        except (KeyError, TypeError) as exc:
            raise ValueError(f"invalid S3 record at index {index}") from exc
        if not isinstance(event_name, str) or not event_name.startswith(
            "ObjectCreated:"
        ):
            raise ValueError(f"unsupported S3 event at index {index}")
        if not isinstance(bucket, str) or not isinstance(key, str):
            raise ValueError(f"invalid S3 record at index {index}")
        if version_id is not None and not isinstance(version_id, str):
            raise ValueError(f"invalid S3 record at index {index}")
        inputs.append((bucket, key, version_id))
    return inputs


def _environment_config() -> dict[str, Any]:
    raw = os.environ.get("QUALIPILOT_CONFIG_JSON", "{}")
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("QUALIPILOT_CONFIG_JSON must be valid JSON") from exc
    if not isinstance(config, dict):
        raise ValueError("QUALIPILOT_CONFIG_JSON must be a JSON object")
    return config


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not parsed.path.lstrip("/")
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"expected s3://bucket/key, got {uri!r}")
    return parsed.netloc, unquote(parsed.path.lstrip("/"))


def _validate_input(bucket: str, key: str) -> None:
    if not bucket or not key or key.endswith("/"):
        raise ValueError("S3 input must identify a bucket and object key")
    if key.startswith("reports/"):
        raise ValueError("objects below reports/ cannot be used as inputs")
    suffix = PurePosixPath(key).suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(_SUPPORTED_SUFFIXES))
        raise ValueError(
            f"unsupported input type {suffix!r}; expected {supported}"
        )


def _validate_output_key(value: Any) -> str:
    if not isinstance(value, str) or not value.startswith("reports/"):
        raise ValueError("output_key must be a string below reports/")
    if (
        value.endswith("/")
        or "\x00" in value
        or ".." in PurePosixPath(value).parts
    ):
        raise ValueError("output_key must identify a safe S3 object key")
    if len(value.encode("utf-8")) > 1024:
        raise ValueError("output_key exceeds the S3 key length limit")
    return value


def _derive_output_key(input_key: str, identity: str) -> str:
    suffix = f".quality.{identity[:16]}.json"
    output = f"reports/{input_key}{suffix}"
    if len(output.encode()) <= 1024:
        return _validate_output_key(output)
    key_hash = hashlib.sha256(input_key.encode()).hexdigest()[:16]
    suffix = f".{key_hash}{suffix}"
    byte_budget = 1024 - len(b"reports/") - len(suffix.encode())
    mirrored = input_key.encode()[:byte_budget].decode("utf-8", "ignore")
    return _validate_output_key(f"reports/{mirrored}{suffix}")


def _report_identity(
    bucket: str,
    key: str,
    source_version: str,
    config: QualipilotConfig,
) -> str:
    payload = json.dumps(
        {
            "bucket": bucket,
            "key": key,
            "source_version": source_version,
            "config": config_fingerprint(config),
            "build": os.environ.get(
                "QUALIPILOT_BUILD_ID",
                _PACKAGE_VERSION,
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _fail_on(value: Any) -> str:
    configured = (
        value
        if value is not None
        else os.environ.get("QUALIPILOT_FAIL_ON", "error")
    )
    if (
        not isinstance(configured, str)
        or configured.lower() not in _FAIL_ON_VALUES
    ):
        allowed = ", ".join(sorted(_FAIL_ON_VALUES))
        raise ValueError(f"fail_on must be one of: {allowed}")
    return configured.lower()


def _max_input_bytes() -> int:
    return _positive_environment_integer(
        "QUALIPILOT_MAX_INPUT_BYTES",
        _DEFAULT_MAX_INPUT_BYTES,
    )


def _set_bedrock_budget(
    config: QualipilotConfig,
    remaining_time: Callable[[], int] | None,
) -> bool:
    if remaining_time is None:
        return True
    remaining_seconds = remaining_time() / 1000
    retries = min(config.llm.retries, 1)
    available = remaining_seconds - 60
    if available <= 20:
        return False
    timeout = min(
        config.llm.timeout_seconds,
        max(1.0, available / (retries + 1) - 10),
    )
    config.llm = config.llm.model_copy(
        update={"retries": retries, "timeout_seconds": timeout}
    )
    return True


def _max_dataset_bytes() -> int:
    return _positive_environment_integer(
        "QUALIPILOT_MAX_DATASET_BYTES",
        _DEFAULT_MAX_DATASET_BYTES,
    )


def _positive_environment_integer(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_local_input(path: Path, max_dataset_bytes: int) -> None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        _validate_text_size(path, max_dataset_bytes)
        columns = require_unique_csv_columns(path)
        if columns is not None and len(columns) > _MAX_COLUMNS:
            raise ValueError("CSV input exceeds the 10,000-column limit")
        return
    if suffix in {".jsonl", ".ndjson"}:
        _validate_text_size(path, max_dataset_bytes)
        json_columns = require_valid_json_lines(path)
        if len(json_columns) > _MAX_COLUMNS:
            raise ValueError("JSONL input exceeds the 10,000-column limit")
        return
    if suffix not in {".parquet", ".pq"}:
        return
    from pyarrow import parquet

    parquet_file = parquet.ParquetFile(path)
    metadata = parquet_file.metadata
    uncompressed_bytes = sum(
        metadata.row_group(index).total_byte_size
        for index in range(metadata.num_row_groups)
    )
    if metadata.num_columns > _MAX_COLUMNS:
        raise ValueError("Parquet input exceeds the 10,000-column limit")
    if any(field.type.num_fields for field in parquet_file.schema_arrow):
        raise ValueError("Parquet input must contain only scalar columns")
    if uncompressed_bytes > max_dataset_bytes:
        raise ValueError(
            f"Parquet input expands to {uncompressed_bytes} bytes; "
            f"limit is {max_dataset_bytes} bytes"
        )


def _validate_text_size(path: Path, max_dataset_bytes: int) -> None:
    estimated_bytes = path.stat().st_size * _TEXT_MEMORY_FACTOR
    if estimated_bytes > max_dataset_bytes:
        raise ValueError(
            f"Text input may expand to {estimated_bytes} bytes; "
            f"limit is {max_dataset_bytes} bytes"
        )


def _meets_threshold(summary: dict[str, str], fail_on: str) -> bool:
    if fail_on == "none":
        return False
    threshold = _SEVERITY_RANK[fail_on]
    return any(
        _SEVERITY_RANK[severity] >= threshold for severity in summary.values()
    )


if __name__ == "__main__":  # pragma: no cover
    import sys

    invocation = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    print(json.dumps(handler(invocation, None), indent=2, default=str))
