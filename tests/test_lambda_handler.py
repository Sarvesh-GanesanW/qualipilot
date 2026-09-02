"""Tests for the Lambda/S3 operational boundary."""

from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from qualipilot import lambda_handler


class _S3Error(Exception):
    def __init__(self, code: str) -> None:
        self.response = {"Error": {"Code": code}}


class _S3:
    def __init__(
        self,
        *,
        size: int | None = None,
        body: bytes = b"id\n1\n",
        version_id: str | None = None,
        missing_report_code: str = "404",
        put_errors: list[str] | None = None,
    ) -> None:
        self.size = len(body) if size is None else size
        self.body = body
        self.etag = '"etag-1"'
        self.version_id = version_id
        self.missing_report_code = missing_report_code
        self.put_errors = list(put_errors or [])
        self.downloads: list[tuple[str, str]] = []
        self.head_calls: list[dict[str, Any]] = []
        self.get_kwargs: list[dict[str, Any]] = []
        self.download_kwargs: list[dict[str, Any]] = []
        self.puts: list[dict[str, Any]] = []
        self.put_attempts = 0
        self.client_calls: list[str] = []
        self.reports: dict[str, dict[str, Any]] = {}

    def head_object(self, **kwargs: Any) -> dict[str, Any]:
        self.head_calls.append(kwargs)
        key = kwargs["Key"]
        if key.startswith("reports/"):
            try:
                return self.reports[key]
            except KeyError as exc:
                raise _S3Error(self.missing_report_code) from exc
        response: dict[str, Any] = {
            "ContentLength": self.size,
            "ETag": self.etag,
        }
        if self.version_id is not None:
            response["VersionId"] = self.version_id
        return response

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        self.get_kwargs.append(kwargs)
        if kwargs["IfMatch"] != self.etag:
            raise _S3Error("PreconditionFailed")
        self.downloads.append((kwargs["Bucket"], kwargs["Key"]))
        return {
            "Body": BytesIO(self.body),
            "ContentLength": self.size,
            "ETag": self.etag,
        }

    def download_file(
        self, bucket: str, key: str, filename: str, **kwargs: Any
    ) -> None:
        self.downloads.append((bucket, key))
        self.download_kwargs.append(kwargs)
        target = Path(filename)
        if target.suffix in {".parquet", ".pq"}:
            import pyarrow as pa
            import pyarrow.parquet as pq

            pq.write_table(pa.table({"id": [1]}), target)
        else:
            target.write_text("id\n1\n", encoding="utf-8")

    def put_object(self, **kwargs: Any) -> None:
        self.put_attempts += 1
        if self.put_errors:
            raise _S3Error(self.put_errors.pop(0))
        if kwargs["Key"] in self.reports and kwargs.get("IfNoneMatch") == "*":
            raise _S3Error("PreconditionFailed")
        self.puts.append(kwargs)
        self.reports[kwargs["Key"]] = {"Metadata": kwargs["Metadata"]}


class _Checker:
    severity = "ok"
    status = "completed"
    llm_status = "disabled"
    last_config: Any = None
    last_kwargs: ClassVar[dict[str, Any]] = {}

    def __init__(self, data: Any, config: Any, **kwargs: Any) -> None:
        _ = data
        type(self).last_config = config
        type(self).last_kwargs = kwargs

    def run(self, *, include_llm: bool = True) -> Any:
        result = SimpleNamespace(
            name="missing_values",
            severity=self.severity,
            status=self.status,
        )
        return SimpleNamespace(
            results=[result],
            llm_status=self.llm_status if include_llm else "disabled",
            to_json=lambda: '{"ok": true}',
        )

    def enrich_with_llm(self, report: Any) -> None:
        report.llm_status = self.llm_status

    def __enter__(self) -> _Checker:
        return self

    def __exit__(self, *_: object) -> None:
        pass


def _install_fakes(monkeypatch: pytest.MonkeyPatch, s3: _S3) -> None:
    def client(service: str) -> _S3:
        s3.client_calls.append(service)
        return s3

    boto3 = SimpleNamespace(client=client)
    monkeypatch.setitem(sys.modules, "boto3", boto3)
    monkeypatch.setattr(lambda_handler, "_S3_CLIENT", None)
    monkeypatch.setattr(lambda_handler, "DataQualityChecker", _Checker)


def test_direct_event_writes_non_colliding_encrypted_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s3 = _S3()
    _install_fakes(monkeypatch, s3)

    response = lambda_handler.handler(
        {
            "s3_uri": "s3://quality/incoming%20files/data.csv",
            "fail_on": "none",
        },
        None,
    )

    assert response["output_key"].startswith(
        "reports/incoming files/data.csv.quality."
    )
    assert response["output_key"].endswith(".json")
    assert s3.downloads == [("quality", "incoming files/data.csv")]
    assert s3.puts[0]["ServerSideEncryption"] == "AES256"


def test_native_s3_event_decodes_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    s3 = _S3()
    _install_fakes(monkeypatch, s3)
    event = {
        "Records": [
            {
                "eventName": "ObjectCreated:Put",
                "s3": {
                    "bucket": {"name": "quality"},
                    "object": {"key": "incoming%2Fdata+set.csv"},
                },
            }
        ],
        "fail_on": "none",
    }

    response = lambda_handler.handler(event, None)

    assert response["processed"] == 1
    assert s3.downloads == [("quality", "incoming/data set.csv")]


def test_native_event_skips_unsupported_objects() -> None:
    event = {
        "Records": [
            {
                "eventName": "ObjectCreated:Put",
                "s3": {
                    "bucket": {"name": "quality"},
                    "object": {"key": "incoming/data.json"},
                },
            }
        ]
    }

    assert lambda_handler.handler(event, None) == {
        "processed": 0,
        "skipped": 1,
        "results": [],
    }


def test_native_event_processes_every_supported_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s3 = _S3()
    _install_fakes(monkeypatch, s3)
    event = {
        "Records": [
            {
                "eventName": "ObjectCreated:Put",
                "s3": {
                    "bucket": {"name": "quality"},
                    "object": {"key": f"incoming/{name}.csv"},
                },
            }
            for name in ("first", "second")
        ],
        "fail_on": "none",
    }

    response = lambda_handler.handler(event, None)

    assert response["processed"] == 2
    assert response["skipped"] == 0
    assert len(response["results"]) == 2


def test_duplicate_bedrock_records_keep_stable_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s3 = _S3()
    _install_fakes(monkeypatch, s3)
    config = lambda_handler.QualipilotConfig(
        llm={
            "provider": "bedrock",
            "model": "profile-id",
            "timeout_seconds": 60,
            "retries": 3,
        }
    )
    fingerprint = lambda_handler.config_fingerprint(config)

    results = lambda_handler._process_notifications(
        s3,
        [
            ("quality", "incoming/data.csv", None),
            ("quality", "incoming/data.csv", None),
        ],
        config=config,
        max_input_bytes=100,
        max_dataset_bytes=100,
        remaining_time=lambda: 120_000,
    )

    assert results[0]["output_key"] == results[1]["output_key"]
    assert results[1]["cached"] is True
    assert lambda_handler.config_fingerprint(config) == fingerprint
    assert len(s3.downloads) == 1


def test_versioned_event_reads_and_reports_exact_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s3 = _S3(version_id="version-1")
    _install_fakes(monkeypatch, s3)
    event = {
        "Records": [
            {
                "eventName": "ObjectCreated:Put",
                "s3": {
                    "bucket": {"name": "quality"},
                    "object": {
                        "key": "incoming/data.csv",
                        "versionId": "version-1",
                    },
                },
            }
        ],
        "fail_on": "none",
    }

    response = lambda_handler.handler(event, None)
    result = response["results"][0]

    assert s3.head_calls[0] == (
        {
            "Bucket": "quality",
            "Key": "incoming/data.csv",
            "VersionId": "version-1",
        }
    )
    assert s3.download_kwargs == [{"ExtraArgs": {"VersionId": "version-1"}}]
    assert result["output_key"].startswith(
        "reports/incoming/data.csv.quality."
    )
    assert _Checker.last_kwargs == {
        "source": "s3://quality/incoming/data.csv",
        "source_version": "version-1",
    }


def test_unversioned_download_is_bound_to_validated_etag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ChangingS3(_S3):
        def get_object(self, **kwargs: Any) -> dict[str, Any]:
            self.etag = '"etag-2"'
            return super().get_object(**kwargs)

    s3 = ChangingS3()
    _install_fakes(monkeypatch, s3)

    with pytest.raises(RuntimeError, match="changed before download"):
        lambda_handler.handler(
            {"s3_uri": "s3://quality/input.csv", "fail_on": "none"},
            None,
        )

    assert s3.get_kwargs[0]["IfMatch"] == '"etag-1"'
    assert s3.puts == []


def test_mutable_null_version_uses_etag_condition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s3 = _S3(version_id="null")
    _install_fakes(monkeypatch, s3)

    lambda_handler.handler(
        {"s3_uri": "s3://quality/input.csv", "fail_on": "none"},
        None,
    )

    assert s3.get_kwargs == [
        {
            "Bucket": "quality",
            "Key": "input.csv",
            "IfMatch": '"etag-1"',
            "VersionId": "null",
        }
    ]
    assert s3.download_kwargs == []
    assert _Checker.last_kwargs["source_version"] == "etag-1"


def test_streamed_object_cannot_exceed_validated_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s3 = _S3(size=5, body=b"id\n1\nx")
    _install_fakes(monkeypatch, s3)

    with pytest.raises(ValueError, match="exceeded its validated size"):
        lambda_handler.handler(
            {"s3_uri": "s3://quality/input.csv", "fail_on": "none"},
            None,
        )

    assert s3.puts == []


def test_duplicate_event_reuses_existing_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    s3 = _S3()
    _install_fakes(monkeypatch, s3)
    event = {
        "s3_uri": "s3://quality/input.csv",
        "fail_on": "none",
    }

    first = lambda_handler.handler(event, None)
    second = lambda_handler.handler(event, None)

    assert first["cached"] is False
    assert second["cached"] is True
    assert s3.client_calls == ["s3"]
    assert len(s3.downloads) == 1
    assert len(s3.puts) == 1
    metrics = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if '"_aws"' in line
    ]
    assert metrics[-1]["QualityGateFailures"] == 0
    assert metrics[-1]["LLMGenerationFailures"] == 0
    assert metrics[-1]["CheckExecutionFailures"] == 0


def test_duplicate_s3_notification_does_not_repeat_outcome_metric(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    s3 = _S3()
    _install_fakes(monkeypatch, s3)
    monkeypatch.setattr(_Checker, "severity", "warn")
    event = {
        "Records": [
            {
                "eventName": "ObjectCreated:Put",
                "s3": {
                    "bucket": {"name": "quality"},
                    "object": {"key": "incoming/data.csv"},
                },
            }
        ],
        "fail_on": "warn",
    }
    lambda_handler.handler(event, None)

    response = lambda_handler.handler(event, None)

    metrics = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if '"_aws"' in line
    ]
    assert response["quality_gate"] == "failed"
    assert response["results"][0]["cached"] is True
    assert metrics[-1]["QualityGateFailures"] == 0


def test_missing_report_does_not_require_bucket_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s3 = _S3(missing_report_code="AccessDenied")
    _install_fakes(monkeypatch, s3)

    response = lambda_handler.handler(
        {"s3_uri": "s3://quality/input.csv", "fail_on": "none"},
        None,
    )

    assert response["cached"] is False
    assert len(s3.puts) == 1


def test_conditional_write_retries_s3_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s3 = _S3(put_errors=["ConditionalRequestConflict"])
    _install_fakes(monkeypatch, s3)

    response = lambda_handler.handler(
        {"s3_uri": "s3://quality/input.csv", "fail_on": "none"},
        None,
    )

    assert response["cached"] is False
    assert s3.put_attempts == 2
    assert len(s3.puts) == 1


def test_custom_output_key_cannot_alias_another_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s3 = _S3()
    _install_fakes(monkeypatch, s3)
    output_key = "reports/custom.json"

    lambda_handler.handler(
        {
            "s3_uri": "s3://quality/first.csv",
            "output_key": output_key,
            "fail_on": "none",
        },
        None,
    )

    with pytest.raises(ValueError, match="different input"):
        lambda_handler.handler(
            {
                "s3_uri": "s3://quality/second.csv",
                "output_key": output_key,
                "fail_on": "none",
            },
            None,
        )

    assert len(s3.downloads) == 1


def test_environment_supplies_native_event_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s3 = _S3()
    _install_fakes(monkeypatch, s3)
    monkeypatch.setenv(
        "QUALIPILOT_CONFIG_JSON",
        '{"engine":"polars","checks":{"outliers":false}}',
    )

    lambda_handler.handler(
        {"s3_uri": "s3://quality/input.csv", "fail_on": "none"},
        None,
    )

    assert _Checker.last_config.engine == "polars"
    assert not _Checker.last_config.checks.outliers


def test_bedrock_uses_lambda_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch, _S3())
    monkeypatch.setenv("AWS_REGION", "eu-west-1")

    lambda_handler.handler(
        {
            "s3_uri": "s3://quality/input.csv",
            "config": {
                "llm": {
                    "provider": "bedrock",
                    "model": "profile-id",
                }
            },
            "fail_on": "none",
        },
        None,
    )

    assert _Checker.last_config.llm.region == "eu-west-1"


def test_bedrock_reserves_time_to_upload_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fakes(monkeypatch, _S3())
    context = SimpleNamespace(get_remaining_time_in_millis=lambda: 120_000)

    lambda_handler.handler(
        {
            "s3_uri": "s3://quality/input.csv",
            "config": {
                "llm": {
                    "provider": "bedrock",
                    "model": "profile-id",
                    "timeout_seconds": 60,
                    "retries": 3,
                }
            },
            "fail_on": "none",
        },
        context,
    )

    assert _Checker.last_config.llm.retries == 1
    assert _Checker.last_config.llm.timeout_seconds == 20


def test_bedrock_is_skipped_but_report_is_uploaded_near_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s3 = _S3()
    _install_fakes(monkeypatch, s3)
    context = SimpleNamespace(get_remaining_time_in_millis=lambda: 70_000)

    response = lambda_handler.handler(
        {
            "s3_uri": "s3://quality/input.csv",
            "config": {
                "llm": {
                    "provider": "bedrock",
                    "model": "profile-id",
                }
            },
            "fail_on": "none",
        },
        context,
    )

    assert len(s3.puts) == 1
    assert response["llm_failures"] == 1


def test_rejects_oversized_object_before_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s3 = _S3(size=101)
    _install_fakes(monkeypatch, s3)
    monkeypatch.setenv("QUALIPILOT_MAX_INPUT_BYTES", "100")

    with pytest.raises(ValueError, match="limit is 100 bytes"):
        lambda_handler.handler(
            {"s3_uri": "s3://quality/input.csv", "fail_on": "none"}, None
        )

    assert s3.downloads == []
    assert s3.puts == []


def test_rejects_parquet_that_expands_beyond_memory_budget(
    tmp_path: Path,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "large.parquet"
    pq.write_table(pa.table({"value": ["x" * 100]}), path)

    with pytest.raises(ValueError, match="expands to"):
        lambda_handler._validate_local_input(path, 1)


def test_rejects_text_that_may_expand_beyond_memory_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "large.csv"
    path.write_text("id\n1\n", encoding="utf-8")
    monkeypatch.setattr(
        lambda_handler,
        "require_unique_csv_columns",
        lambda _: pytest.fail("header parsed before size rejection"),
    )

    with pytest.raises(ValueError, match="Text input may expand"):
        lambda_handler._validate_local_input(path, 39)


def test_rejects_wide_csv_before_dataframe_load(tmp_path: Path) -> None:
    path = tmp_path / "wide.csv"
    path.write_text(
        ",".join(f"column_{index}" for index in range(10_001)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="10,000-column"):
        lambda_handler._validate_local_input(path, 10_000_000)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('{"id": 1}\nnot-json\n', "invalid JSONL record on line 2"),
        ("[1, 2]\n", "must be an object"),
        ('{"items": [1, 2]}\n', "only scalar values"),
        ('{"id": 1, "id": 2}\n', "duplicate object keys"),
    ],
)
def test_rejects_invalid_jsonl_before_dataframe_load(
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        lambda_handler._validate_local_input(path, 10_000)


def test_rejects_wide_jsonl_before_dataframe_load(tmp_path: Path) -> None:
    path = tmp_path / "wide.jsonl"
    path.write_text(
        json.dumps({f"column_{index}": index for index in range(10_001)}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="10,000-column"):
        lambda_handler._validate_local_input(path, 10_000_000)


def test_rejects_nested_parquet_before_dataframe_load(
    tmp_path: Path,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "nested.parquet"
    pq.write_table(pa.table({"items": [[1, 2]]}), path)

    with pytest.raises(ValueError, match="only scalar columns"):
        lambda_handler._validate_local_input(path, 10_000)


def test_threshold_failure_occurs_after_report_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s3 = _S3()
    _install_fakes(monkeypatch, s3)
    monkeypatch.setattr(_Checker, "severity", "warn")

    response = lambda_handler.handler(
        {"s3_uri": "s3://quality/input.csv", "fail_on": "warn"}, None
    )

    assert len(s3.puts) == 1
    assert response["quality_gate"] == "failed"
    assert response["quality_failures"] == 1


def test_cached_report_emits_current_quality_gate_metric(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    s3 = _S3()
    _install_fakes(monkeypatch, s3)
    monkeypatch.setattr(_Checker, "severity", "warn")
    lambda_handler.handler(
        {"s3_uri": "s3://quality/input.csv", "fail_on": "none"},
        None,
    )

    response = lambda_handler.handler(
        {"s3_uri": "s3://quality/input.csv", "fail_on": "warn"},
        None,
    )

    metrics = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if '"_aws"' in line
    ]
    assert response["cached"] is True
    assert response["quality_gate"] == "failed"
    assert metrics[-1]["QualityGateFailures"] == 1


def test_llm_failure_occurs_after_report_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s3 = _S3()
    _install_fakes(monkeypatch, s3)
    monkeypatch.setattr(_Checker, "llm_status", "failed")

    response = lambda_handler.handler(
        {
            "s3_uri": "s3://quality/input.csv",
            "config": {
                "llm": {
                    "provider": "bedrock",
                    "model": "profile-id",
                }
            },
            "fail_on": "none",
        },
        None,
    )

    assert len(s3.puts) == 1
    assert response["llm_failures"] == 1


def test_retry_after_llm_failure_writes_terminal_custom_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s3 = _S3()
    _install_fakes(monkeypatch, s3)
    event = {
        "s3_uri": "s3://quality/input.csv",
        "output_key": "reports/custom.json",
        "config": {
            "llm": {
                "provider": "bedrock",
                "model": "profile-id",
            }
        },
        "fail_on": "none",
    }
    monkeypatch.setattr(_Checker, "llm_status", "failed")
    first = lambda_handler.handler(event, None)
    monkeypatch.setattr(_Checker, "llm_status", "completed")

    second = lambda_handler.handler(event, None)
    third = lambda_handler.handler(event, None)

    assert first["llm_failures"] == 1
    assert first["output_key"].startswith("reports/failures/")
    assert second["llm_failures"] == 0
    assert second["cached"] is False
    assert second["output_key"] == "reports/custom.json"
    assert third["cached"] is True
    assert third["output_key"] == "reports/custom.json"
    assert len(s3.downloads) == 2
    assert all(item["IfNoneMatch"] == "*" for item in s3.puts)


def test_check_execution_failure_raises_after_report_upload(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    s3 = _S3()
    _install_fakes(monkeypatch, s3)
    monkeypatch.setattr(_Checker, "status", "failed")

    with pytest.raises(RuntimeError, match="failed to execute"):
        lambda_handler.handler(
            {"s3_uri": "s3://quality/input.csv", "fail_on": "none"},
            None,
        )

    assert len(s3.puts) == 1
    metric = next(
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if '"_aws"' in line
    )
    assert metric["QualityGateFailures"] == 0
    assert metric["CheckExecutionFailures"] == 1


def test_retry_after_check_execution_failure_writes_terminal_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s3 = _S3()
    _install_fakes(monkeypatch, s3)
    event = {
        "s3_uri": "s3://quality/input.csv",
        "fail_on": "none",
    }
    monkeypatch.setattr(_Checker, "status", "failed")
    with pytest.raises(RuntimeError, match="failed to execute"):
        lambda_handler.handler(event, None)
    assert s3.puts[0]["Key"].startswith("reports/failures/")
    monkeypatch.setattr(_Checker, "status", "completed")

    response = lambda_handler.handler(event, None)

    assert response["cached"] is False
    assert response["execution_failures"] == 0
    assert len(s3.downloads) == 2
    assert not s3.puts[-1]["Key"].startswith("reports/failures/")


def test_derived_output_key_handles_long_unicode_input() -> None:
    input_key = f"incoming/{'é' * 490}.csv"

    output = lambda_handler._derive_output_key(input_key, "a" * 64)

    assert len(input_key.encode()) <= 1024
    assert len(output.encode()) <= 1024
    assert output.startswith("reports/")
    assert output.endswith(".quality.aaaaaaaaaaaaaaaa.json")


@pytest.mark.parametrize(
    ("event", "message"),
    [
        (
            {"s3_uri": "https://example.com/data.csv"},
            "expected s3://bucket/key",
        ),
        (
            {"s3_uri": "s3://quality/reports/data.csv"},
            "reports/ cannot be used",
        ),
        (
            {
                "s3_uri": "s3://quality/data.csv",
                "output_key": "data.quality.json",
            },
            "below reports/",
        ),
        (
            {
                "s3_uri": "s3://quality/data.csv",
                "config": {"output_path": "/tmp/out.json"},
            },
            "not supported in Lambda",
        ),
        (
            {
                "s3_uri": "s3://quality/data.csv",
                "config": {"report_format": "html"},
            },
            "only JSON reports",
        ),
        (
            {
                "s3_uri": "s3://quality/data.csv",
                "config": {"engine": "pandas"},
            },
            "only auto or polars",
        ),
        (
            {
                "s3_uri": "s3://quality/data.csv",
                "config": {
                    "llm": {
                        "provider": "openai",
                        "model": "remote",
                        "base_url": "https://api.example.com",
                    }
                },
            },
            "only none or bedrock",
        ),
        (
            {
                "Records": [
                    {
                        "eventName": "ObjectCreated:Put",
                        "s3": {
                            "bucket": {"name": "quality"},
                            "object": {"key": "one.csv"},
                        },
                    },
                    {
                        "eventName": "ObjectCreated:Put",
                        "s3": {
                            "bucket": {"name": "quality"},
                            "object": {"key": "two.csv"},
                        },
                    },
                ],
                "output_key": "reports/custom.json",
            },
            "exactly one input",
        ),
    ],
)
def test_rejects_unsafe_events(
    monkeypatch: pytest.MonkeyPatch,
    event: dict[str, Any],
    message: str,
) -> None:
    _install_fakes(monkeypatch, _S3())

    with pytest.raises(ValueError, match=message):
        lambda_handler.handler(event, None)
