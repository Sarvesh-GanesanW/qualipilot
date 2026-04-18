"""End-to-end orchestrator tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from datapilot import DatapilotConfig, DataQualityChecker
from datapilot.models.config import CheckConfig, ColumnRange


def test_full_run_produces_all_sections(
    dirty_pandas: pd.DataFrame,
) -> None:
    cfg = DatapilotConfig(
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


def test_save_writes_json(
    tmp_path: Path, dirty_pandas: pd.DataFrame
) -> None:
    cfg = DatapilotConfig(output_path=tmp_path / "out.json")
    DataQualityChecker(dirty_pandas, cfg).run()
    assert (tmp_path / "out.json").exists()


def test_engine_override(dirty_pandas: pd.DataFrame) -> None:
    cfg = DatapilotConfig(engine="pandas")
    report = DataQualityChecker(dirty_pandas, cfg).run()
    assert report.dataset.engine == "pandas"


def test_exit_severity_helpers(dirty_pandas: pd.DataFrame) -> None:
    cfg = DatapilotConfig(
        checks=CheckConfig(
            column_ranges={"amount": ColumnRange(min=0, max=100)}
        )
    )
    report = DataQualityChecker(dirty_pandas, cfg).run()
    assert report.failed_checks()  # ranges must fail
    assert any(r.severity == "warn" for r in report.warning_checks())
