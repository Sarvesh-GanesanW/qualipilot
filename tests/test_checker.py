"""End-to-end orchestrator tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from qualipilot import DataQualityChecker, QualipilotConfig
from qualipilot.checker import (
    _build_llm_prompt,
    _summarise_payload,
    config_fingerprint,
)
from qualipilot.models.config import CheckConfig, ColumnRange, LLMConfig
from qualipilot.models.results import CheckResult, DatasetStats, QualityReport


def test_full_run_produces_all_sections(
    dirty_pandas: pd.DataFrame,
) -> None:
    cfg = QualipilotConfig(
        checks=CheckConfig(
            column_ranges={"amount": ColumnRange(min=0, max=100)}
        )
    )
    report = DataQualityChecker(dirty_pandas, cfg).run()
    names = {r.name for r in report.results}
    assert {
        "missing_values",
        "duplicates",
        "data_types",
        "outliers",
        "ranges",
        "cardinality",
    }.issubset(names)
    assert report.dataset.row_count > 0
    assert report.config_hash is not None


def test_save_writes_json(tmp_path: Path, dirty_pandas: pd.DataFrame) -> None:
    cfg = QualipilotConfig(output_path=tmp_path / "out.json")
    DataQualityChecker(dirty_pandas, cfg).run()
    assert (tmp_path / "out.json").exists()


def test_configured_markdown_output_is_honoured(
    tmp_path: Path, dirty_pandas: pd.DataFrame
) -> None:
    output = tmp_path / "out.txt"
    config = QualipilotConfig(
        output_path=output,
        report_format="markdown",
    )

    DataQualityChecker(dirty_pandas, config).run()

    assert output.read_text(encoding="utf-8").startswith(
        "# Data Quality Report"
    )


def test_engine_override(dirty_pandas: pd.DataFrame) -> None:
    cfg = QualipilotConfig(engine="pandas")
    report = DataQualityChecker(dirty_pandas, cfg).run()
    assert report.dataset.engine == "pandas"


def test_report_records_input_provenance(tmp_csv: Path) -> None:
    report = DataQualityChecker(tmp_csv).run()

    assert report.dataset.source == str(tmp_csv)
    assert report.package_version


def test_report_source_redacts_url_credentials(
    tidy_pandas: pd.DataFrame,
) -> None:
    report = DataQualityChecker(
        tidy_pandas,
        source=(
            "https://user:secret@example.com/data.csv"
            "?X-Amz-Credential=private#fragment"
        ),
    ).run()

    assert report.dataset.source == "https://example.com/data.csv"


def test_output_cannot_overwrite_input(tmp_csv: Path) -> None:
    config = QualipilotConfig(output_path=tmp_csv.resolve())

    with pytest.raises(ValueError, match="must not overwrite"):
        DataQualityChecker(tmp_csv, config)


def test_non_string_column_names_are_rejected() -> None:
    frame = pd.DataFrame([[1]], columns=[1])

    with pytest.raises(ValueError, match="column names must be strings"):
        DataQualityChecker(frame, QualipilotConfig(engine="pandas"))


def test_duplicate_column_names_are_rejected() -> None:
    frame = pd.DataFrame([[1, 2]], columns=["value", "value"])

    with pytest.raises(ValueError, match="column names must be unique"):
        DataQualityChecker(frame, QualipilotConfig(engine="duckdb"))


def test_checker_context_closes_duckdb_engine(
    tidy_pandas: pd.DataFrame,
) -> None:
    with DataQualityChecker(
        tidy_pandas,
        QualipilotConfig(engine="duckdb"),
    ) as checker:
        engine = checker.engine
        checker.run()

    with pytest.raises(Exception, match="closed"):
        engine.row_count()


def test_exit_severity_helpers(dirty_pandas: pd.DataFrame) -> None:
    cfg = QualipilotConfig(
        checks=CheckConfig(
            column_ranges={"amount": ColumnRange(min=0, max=100)}
        )
    )
    report = DataQualityChecker(dirty_pandas, cfg).run()
    assert report.failed_checks()  # ranges must fail
    assert any(r.severity == "warn" for r in report.warning_checks())


def test_config_fingerprint_is_canonical_and_excludes_secrets() -> None:
    first = QualipilotConfig(
        checks=CheckConfig(
            column_ranges={
                "a": ColumnRange(min=0, max=1),
                "b": ColumnRange(min=0, max=2),
            }
        ),
        llm=LLMConfig(
            provider="openai",
            model="test-model",
            base_url="https://api.example.com/v1",
            api_key="first-secret",
        ),
    )
    second = QualipilotConfig(
        checks=CheckConfig(
            column_ranges={
                "b": ColumnRange(min=0, max=2),
                "a": ColumnRange(min=0, max=1),
            }
        ),
        llm=LLMConfig(
            provider="openai",
            model="test-model",
            base_url="https://api.example.com/v1",
            api_key="second-secret",
        ),
    )

    assert config_fingerprint(first) == config_fingerprint(second)


def test_llm_initialisation_failure_is_structured(
    tidy_pandas: pd.DataFrame,
) -> None:
    config = QualipilotConfig(
        llm=LLMConfig(
            provider="openai",
            model="test-model",
            base_url="https://api.example.com/v1",
            api_key="secret",
        )
    )

    with patch(
        "qualipilot.llm.build_provider",
        side_effect=ImportError("provider unavailable"),
    ):
        report = DataQualityChecker(tidy_pandas, config).run()

    assert report.llm_status == "failed"
    assert report.llm_report is None
    assert report.llm_error == "ImportError: provider unavailable"


def test_llm_summary_keeps_findings_but_removes_samples() -> None:
    summary = _summarise_payload(
        {
            "per_column": [
                {
                    "column": "amount",
                    "outlier_count": 3,
                    "sample": [{"amount": 10_000}],
                    "top_values": [["private", 1]],
                }
            ]
        }
    )

    item = summary["per_column"]["items"][0]
    assert item["column"] == "amount"
    assert item["outlier_count"] == 3
    assert "sample" not in item
    assert "top_values" not in item


def test_llm_summary_prioritizes_late_findings_before_cap() -> None:
    per_column = [
        {"column": f"clean_{index}", "null_count": 0} for index in range(60)
    ]
    per_column.append({"column": "broken", "null_count": 1})

    summary = _summarise_payload({"per_column": per_column})

    assert summary["per_column"]["items"][0]["column"] == "broken"
    assert summary["per_column"]["total_count"] == 61
    assert summary["per_column"]["omitted_count"] == 11


@pytest.mark.parametrize(
    "finding",
    [
        {"column": "stale", "is_stale": True},
        {"column": "future", "is_future": True},
        {"column": "constant", "distinct_count": 1},
    ],
)
def test_llm_summary_prioritizes_each_actionable_shape(
    finding: dict[str, object],
) -> None:
    per_column = [
        {"column": f"clean_{index}", "distinct_count": 2}
        for index in range(60)
    ]
    per_column.append(finding)

    summary = _summarise_payload({"per_column": per_column})

    assert summary["per_column"]["items"][0] == finding


def test_llm_prompt_excludes_source_and_values() -> None:
    report = QualityReport(
        dataset=DatasetStats(
            row_count=1,
            column_count=1,
            columns=["email"],
            dtypes={"email": "String"},
            engine="polars",
            source="s3://secret-bucket/customers.csv",
            source_version="private-version",
        ),
        results=[
            CheckResult(
                name="cardinality",
                severity="error",
                duration_seconds=0,
                status="failed",
                error="could not read s3://secret-bucket/customers.csv",
                payload={
                    "per_column": [
                        {
                            "column": "email",
                            "top_values": [["alice@example.com", 1]],
                        }
                    ]
                },
            )
        ],
    )

    prompt = _build_llm_prompt(report)

    assert "alice@example.com" not in prompt
    assert "secret-bucket" not in prompt
    assert "private-version" not in prompt
    assert "email" in prompt
    assert '"status": "failed"' in prompt


def test_result_payload_is_json_safe() -> None:
    result = CheckResult(
        name="binary",
        severity="warn",
        duration_seconds=0,
        payload={"sample": [{("value", 1): b"\xff"}]},
    )

    assert "xff" in result.model_dump_json()
    assert "('value', 1)" in result.payload["sample"][0]


def test_run_reads_structural_metadata_once(
    tidy_pandas: pd.DataFrame,
) -> None:
    config = QualipilotConfig(
        engine="pandas",
        checks=CheckConfig(
            missing_values=False,
            duplicates=False,
            data_types=False,
            outliers=False,
            ranges=False,
            cardinality=False,
        ),
    )
    checker = DataQualityChecker(tidy_pandas, config)

    with (
        patch.object(
            checker.engine,
            "row_count",
            wraps=checker.engine.row_count,
        ) as row_count,
        patch.object(
            checker.engine,
            "columns",
            wraps=checker.engine.columns,
        ) as columns,
        patch.object(
            checker.engine,
            "dtypes",
            wraps=checker.engine.dtypes,
        ) as dtypes,
    ):
        checker.run()

    row_count.assert_called_once_with()
    columns.assert_called_once_with()
    dtypes.assert_called_once_with()
