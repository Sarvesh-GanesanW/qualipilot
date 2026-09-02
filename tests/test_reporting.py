"""Tests for the markdown + html report renderers.

These are tighter than golden-file comparisons: we assert the human
summary surfaces affected columns by name, not just the count.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from pydantic import ValidationError

from qualipilot import DataQualityChecker, QualipilotConfig
from qualipilot.models.config import CheckConfig, ColumnRange
from qualipilot.models.results import CheckResult, DatasetStats, QualityReport
from qualipilot.reporting import render_html, render_markdown


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_check_result_normalizes_non_finite_payloads(value: float) -> None:
    result = CheckResult(
        name="custom",
        severity="warn",
        duration_seconds=0,
        payload={"value": value},
    )

    assert result.payload["value"] in {"nan", "inf"}
    assert json.loads(result.model_dump_json())["payload"]["value"] in {
        "nan",
        "inf",
    }


def _report(dirty_pandas: pd.DataFrame):
    cfg = QualipilotConfig(
        checks=CheckConfig(
            column_ranges={"amount": ColumnRange(min=0, max=100)},
        ),
    )
    return DataQualityChecker(dirty_pandas, cfg).run()


def test_markdown_lists_columns_with_nulls(
    dirty_pandas: pd.DataFrame,
) -> None:
    md = render_markdown(_report(dirty_pandas))
    # `amount` is the column with the null in the fixture
    assert "`amount`" in md
    assert "Nulls" in md


def test_markdown_lists_outlier_columns_with_bounds(
    dirty_pandas: pd.DataFrame,
) -> None:
    md = render_markdown(_report(dirty_pandas))
    # the IQR table must appear with the affected column and bounds
    assert "Bounds (IQR)" in md
    assert "amount" in md


def test_markdown_lists_range_violations(
    dirty_pandas: pd.DataFrame,
) -> None:
    md = render_markdown(_report(dirty_pandas))
    assert "Violations" in md
    # the range we configured surfaces in the affected table
    assert r"\[0.0, 100.0\]" in md


def test_markdown_outlier_phrasing_says_numeric(
    dirty_pandas: pd.DataFrame,
) -> None:
    md = render_markdown(_report(dirty_pandas))
    # previous phrasing was "columns evaluated: 3" which read as
    # "only 3 of 6 columns checked" — we now say "numeric columns".
    assert "numeric columns scanned" in md


def test_html_renders_affected_columns_table(
    dirty_pandas: pd.DataFrame,
) -> None:
    h = render_html(_report(dirty_pandas))
    assert "<th>Outliers</th>" in h
    assert "<th>Bounds (IQR)</th>" in h
    assert "<th>Violations</th>" in h


def test_html_keeps_raw_payload_collapsed(
    dirty_pandas: pd.DataFrame,
) -> None:
    h = render_html(_report(dirty_pandas))
    assert "<details>" in h
    assert "raw payload" in h


def test_markdown_escapes_untrusted_cells_and_html() -> None:
    report = QualityReport(
        dataset=DatasetStats(
            row_count=2,
            column_count=1,
            columns=["id"],
            dtypes={"id": "str"},
            engine="test",
        ),
        results=[
            CheckResult(
                name="duplicates",
                severity="warn",
                duration_seconds=0,
                payload={
                    "total_duplicate_rows": 2,
                    "sample": [
                        {
                            "id": (
                                "safe | forged\n# heading "
                                "![track](https://attacker.invalid/pixel)"
                            )
                        }
                    ],
                },
            )
        ],
        llm_report=(
            "<details open>spoof</details>\n"
            "[external](https://example.com)\n# forged heading"
        ),
        llm_status="completed",
        llm_provider="openai",
        llm_model="example-model",
    )

    markdown = render_markdown(report)

    assert r"safe \| forged&#10;# heading" in markdown
    assert r"\!\[track\]\(https://attacker.invalid/pixel\)" in markdown
    assert "![track](" not in markdown
    assert "<details" not in markdown
    assert "    &lt;details open&gt;" in markdown
    assert "    [external](https://example.com)" in markdown
    assert "    # forged heading" in markdown
    assert "AI-generated advisory; validate before acting" in markdown
    assert "`openai`" in markdown
    assert "`example-model`" in markdown

    rendered_html = render_html(report)
    assert "AI-generated advisory" in rendered_html
    assert "openai / example-model" in rendered_html


def test_html_surfaces_escaped_execution_errors() -> None:
    report = QualityReport(
        dataset=DatasetStats(
            row_count=1,
            column_count=1,
            columns=["id"],
            dtypes={"id": "int"},
            engine="test",
        ),
        results=[
            CheckResult(
                name="cardinality",
                severity="error",
                status="failed",
                duration_seconds=0,
                error="<unsupported>",
            )
        ],
    )

    rendered = render_html(report)

    assert "Execution failed" in rendered
    assert "&lt;unsupported&gt;" in rendered
    assert "<unsupported>" not in rendered


def test_linkage_renderers_show_the_match_threshold() -> None:
    report = QualityReport(
        dataset=DatasetStats(
            row_count=2,
            column_count=2,
            columns=["id", "name"],
            dtypes={"id": "int", "name": "str"},
            engine="polars",
        ),
        results=[
            CheckResult(
                name="linkage",
                severity="warn",
                duration_seconds=0,
                payload={
                    "candidate_pairs": 1,
                    "matched_pairs": 1,
                    "match_threshold_probability": 0.83,
                    "duplicate_clusters": 1,
                    "records_in_duplicate_groups": 2,
                },
            )
        ],
    )

    assert "match threshold: 0.83" in render_markdown(report)
    assert "match threshold: 0.83" in render_html(report)


def test_outlier_renderers_explain_skipped_columns() -> None:
    report = _single_result_report(
        CheckResult(
            name="outliers",
            severity="warn",
            duration_seconds=0,
            payload={
                "per_column": [
                    {
                        "column": "amount",
                        "skipped": "non-finite quantile bounds",
                    }
                ]
            },
        )
    )

    for rendered in (render_markdown(report), render_html(report)):
        assert "amount" in rendered
        assert "non-finite quantile bounds" in rendered


def test_html_freshness_does_not_invent_an_age_for_empty_columns() -> None:
    report = _single_result_report(
        CheckResult(
            name="freshness",
            severity="error",
            duration_seconds=0,
            payload={
                "per_column": [
                    {
                        "column": "event_time",
                        "max_timestamp": None,
                        "is_stale": True,
                        "note": "no non-null values",
                    }
                ]
            },
        )
    )

    rendered = render_html(report)

    assert "no non-null values" in rendered
    assert "0.0h old" not in rendered


def test_html_dataset_contract_shows_dtype_mismatches() -> None:
    report = _single_result_report(
        CheckResult(
            name="dataset_contract",
            severity="error",
            duration_seconds=0,
            payload={
                "row_count": 1,
                "min_rows": 1,
                "missing_required_columns": [],
                "dtype_mismatches": [
                    {
                        "column": "id",
                        "expected": "Int64",
                        "actual": "String",
                    }
                ],
            },
        )
    )

    rendered = render_html(report)

    assert "Expected dtype" in rendered
    assert "<code>id</code>" in rendered
    assert "<code>Int64</code>" in rendered
    assert "<code>String</code>" in rendered


def test_markdown_retains_known_and_extension_payload_details() -> None:
    report = QualityReport(
        dataset=DatasetStats(
            row_count=1,
            column_count=1,
            columns=["value"],
            dtypes={"value": "str"},
            engine="test",
        ),
        results=[
            CheckResult(
                name="cardinality",
                severity="ok",
                duration_seconds=0,
                payload={
                    "per_column": [
                        {
                            "column": "value",
                            "distinct_count": 1,
                            "top_values": [["requested-value", 1]],
                        }
                    ]
                },
            ),
            CheckResult(
                name="extension_check",
                severity="ok",
                duration_seconds=0,
                payload={"extension_detail": "preserved"},
            ),
        ],
    )

    rendered = render_markdown(report)

    assert rendered.count("Raw payload:") == 2
    assert "requested-value" in rendered
    assert "extension_detail" in rendered
    assert "preserved" in rendered


def test_report_deserialization_ignores_compatible_unknown_fields() -> None:
    report = json.loads(
        _single_result_report(
            CheckResult(
                name="custom",
                severity="ok",
                duration_seconds=0,
            )
        ).to_json()
    )
    report["future_report_field"] = "ignored"
    report["dataset"]["future_dataset_field"] = "ignored"
    report["results"][0]["future_result_field"] = "ignored"

    parsed = QualityReport.from_json(json.dumps(report))

    assert parsed.schema_version == "1.0"
    assert not hasattr(parsed, "future_report_field")
    assert not hasattr(parsed.dataset, "future_dataset_field")
    assert not hasattr(parsed.results[0], "future_result_field")


def test_report_deserialization_rejects_an_unknown_schema_version() -> None:
    report = json.loads(
        _single_result_report(
            CheckResult(
                name="custom",
                severity="ok",
                duration_seconds=0,
            )
        ).to_json()
    )
    report["schema_version"] = "2.0"

    with pytest.raises(ValidationError, match="schema_version"):
        QualityReport.from_json(json.dumps(report))


def test_report_construction_still_rejects_unknown_fields() -> None:
    report = _single_result_report(
        CheckResult(
            name="custom",
            severity="ok",
            duration_seconds=0,
        )
    ).model_dump()
    report["future_report_field"] = "not accepted at strict boundary"

    with pytest.raises(ValidationError, match="future_report_field"):
        QualityReport.model_validate(report)


def _single_result_report(result: CheckResult) -> QualityReport:
    return QualityReport(
        dataset=DatasetStats(
            row_count=1,
            column_count=1,
            columns=["value"],
            dtypes={"value": "str"},
            engine="test",
        ),
        results=[result],
    )
