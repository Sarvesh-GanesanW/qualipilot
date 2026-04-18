"""Checks operate on the ``Engine`` protocol; we test each in isolation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from datapilot.checks import (
    CardinalityCheck,
    CheckContext,
    DataTypesCheck,
    DuplicatesCheck,
    FreshnessCheck,
    MissingValuesCheck,
    OutliersCheck,
    RangesCheck,
)
from datapilot.engines import PolarsEngine
from datapilot.models.config import CheckConfig, ColumnRange


def _ctx(df: pd.DataFrame, cfg: CheckConfig | None = None) -> CheckContext:
    return CheckContext(
        engine=PolarsEngine.from_any(df),
        config=cfg or CheckConfig(),
    )


def test_missing_check_warns_on_nulls(
    dirty_pandas: pd.DataFrame,
) -> None:
    result = MissingValuesCheck().run(_ctx(dirty_pandas))
    assert result.severity == "warn"
    assert result.payload["total_null_count"] >= 1


def test_missing_check_ok_when_clean(
    tidy_pandas: pd.DataFrame,
) -> None:
    result = MissingValuesCheck().run(_ctx(tidy_pandas))
    assert result.severity == "ok"
    assert result.payload["total_null_count"] == 0


def test_duplicates_flagged(dirty_pandas: pd.DataFrame) -> None:
    result = DuplicatesCheck().run(_ctx(dirty_pandas))
    assert result.severity == "warn"
    assert result.payload["total_duplicate_rows"] >= 2


def test_duplicates_with_subset(dirty_pandas: pd.DataFrame) -> None:
    cfg = CheckConfig(duplicate_subset=["category"])
    result = DuplicatesCheck().run(_ctx(dirty_pandas, cfg))
    assert result.payload["subset"] == ["category"]


def test_data_types_rollup(dirty_pandas: pd.DataFrame) -> None:
    result = DataTypesCheck().run(_ctx(dirty_pandas))
    assert result.severity == "ok"
    assert "rollup" in result.payload


def test_outliers_flagged(dirty_pandas: pd.DataFrame) -> None:
    result = OutliersCheck().run(_ctx(dirty_pandas))
    assert result.severity == "warn"
    per_col = result.payload["per_column"]
    amount = next(c for c in per_col if c["column"] == "amount")
    assert amount["outlier_count"] >= 1


def test_ranges_errors_on_violation(
    dirty_pandas: pd.DataFrame,
) -> None:
    cfg = CheckConfig(column_ranges={"amount": ColumnRange(min=0, max=100)})
    result = RangesCheck().run(_ctx(dirty_pandas, cfg))
    assert result.severity == "error"
    amount = result.payload["per_column"][0]
    assert amount["violation_count"] >= 1


def test_ranges_ok_when_not_configured(
    dirty_pandas: pd.DataFrame,
) -> None:
    result = RangesCheck().run(_ctx(dirty_pandas))
    assert result.severity == "ok"


def test_cardinality_detects_constant_column() -> None:
    df = pd.DataFrame({"a": [1] * 10, "b": range(10)})
    result = CardinalityCheck().run(_ctx(df))
    assert result.severity == "warn"
    const = next(c for c in result.payload["per_column"] if c["column"] == "a")
    assert const["distinct_count"] == 1


def test_freshness_flags_old_data(
    stale_timestamps_pandas: pd.DataFrame,
) -> None:
    cfg = CheckConfig(
        freshness=True,
        freshness_columns=["event_ts"],
        freshness_max_age_hours=24.0,
    )
    result = FreshnessCheck().run(_ctx(stale_timestamps_pandas, cfg))
    assert result.severity == "error"


def test_freshness_ok_for_fresh_data() -> None:
    now = datetime.now(UTC)
    df = pd.DataFrame({"event_ts": [now, now - timedelta(minutes=10)]})
    cfg = CheckConfig(
        freshness=True,
        freshness_columns=["event_ts"],
        freshness_max_age_hours=24.0,
    )
    result = FreshnessCheck().run(_ctx(df, cfg))
    assert result.severity == "ok"


@pytest.mark.parametrize(
    "check_cls",
    [
        MissingValuesCheck,
        DuplicatesCheck,
        DataTypesCheck,
        OutliersCheck,
        RangesCheck,
        CardinalityCheck,
    ],
)
def test_checks_never_raise(
    check_cls: type, tidy_pandas: pd.DataFrame
) -> None:
    result = check_cls().run(_ctx(tidy_pandas))
    assert result.error is None
