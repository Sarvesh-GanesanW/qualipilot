"""Checks operate on the ``Engine`` protocol; we test each in isolation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from qualipilot.checks import (
    CardinalityCheck,
    CheckContext,
    DatasetContractCheck,
    DataTypesCheck,
    DuplicatesCheck,
    FreshnessCheck,
    MissingValuesCheck,
    OutliersCheck,
    RangesCheck,
)
from qualipilot.engines import PolarsEngine
from qualipilot.models.config import CheckConfig, ColumnRange


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


def test_dataset_contract_rejects_empty_input() -> None:
    config = CheckConfig(min_rows=1)
    result = DatasetContractCheck().run(_ctx(pd.DataFrame(), config))

    assert result.severity == "error"
    assert result.payload["row_count"] == 0


def test_dataset_contract_reports_missing_required_columns(
    tidy_pandas: pd.DataFrame,
) -> None:
    config = CheckConfig(required_columns=["id", "created_at"])
    result = DatasetContractCheck().run(_ctx(tidy_pandas, config))

    assert result.severity == "error"
    assert result.payload["missing_required_columns"] == ["created_at"]


def test_dataset_contract_reports_dtype_mismatches(
    tidy_pandas: pd.DataFrame,
) -> None:
    config = CheckConfig(expected_dtypes={"id": "Float64"})
    result = DatasetContractCheck().run(_ctx(tidy_pandas, config))

    assert result.severity == "error"
    assert result.payload["dtype_mismatches"] == [
        {"column": "id", "expected": "Float64", "actual": "Int64"}
    ]


def test_expected_dtype_implies_required_column(
    tidy_pandas: pd.DataFrame,
) -> None:
    config = CheckConfig(expected_dtypes={"created_at": "Datetime"})
    result = DatasetContractCheck().run(_ctx(tidy_pandas, config))

    assert result.severity == "error"
    assert result.payload["missing_required_columns"] == ["created_at"]


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
    assert result.payload["sample"] == []


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


def test_outliers_skip_non_finite_bounds() -> None:
    result = OutliersCheck().run(
        _ctx(pd.DataFrame({"value": [0.0, float("inf")]}))
    )

    assert result.severity == "warn"
    assert result.status == "completed"
    assert result.payload["per_column"] == [
        {
            "column": "value",
            "skipped": "non-finite quantile bounds",
        }
    ]


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


def test_ranges_errors_on_non_numeric_column(
    dirty_pandas: pd.DataFrame,
) -> None:
    """Range constraint on a string column was silently ok before."""
    cfg = CheckConfig(
        column_ranges={"category": ColumnRange(min=0, max=10)},
    )
    result = RangesCheck().run(_ctx(dirty_pandas, cfg))
    assert result.severity == "error"
    note = result.payload["per_column"][0]["note"]
    assert "non-numeric" in note


def test_ranges_errors_on_missing_column(
    dirty_pandas: pd.DataFrame,
) -> None:
    """Configured column not present in dataset must fail closed."""
    cfg = CheckConfig(
        column_ranges={"made_up": ColumnRange(min=0, max=10)},
    )
    result = RangesCheck().run(_ctx(dirty_pandas, cfg))
    assert result.severity == "error"
    assert "not present" in result.payload["per_column"][0]["note"]


def test_cardinality_detects_constant_column() -> None:
    df = pd.DataFrame({"a": [1] * 10, "b": range(10)})
    result = CardinalityCheck().run(_ctx(df))
    assert result.severity == "warn"
    const = next(c for c in result.payload["per_column"] if c["column"] == "a")
    assert const["distinct_count"] == 1
    assert const["top_values"] == []


def test_cardinality_surfaces_engine_failures(
    tidy_pandas: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _ctx(tidy_pandas)

    def fail_distinct_counts(columns: list[str]) -> dict[str, int]:
        raise TypeError(f"unsupported columns: {columns}")

    monkeypatch.setattr(
        context.engine,
        "distinct_counts",
        fail_distinct_counts,
    )
    result = CardinalityCheck().run(context)

    assert result.status == "failed"
    assert result.severity == "error"
    assert "unsupported columns" in (result.error or "")


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


def test_freshness_rejects_future_data() -> None:
    future = datetime.now(UTC) + timedelta(hours=2)
    df = pd.DataFrame({"event_ts": [future]})
    config = CheckConfig(
        freshness=True,
        freshness_columns=["event_ts"],
        freshness_max_age_hours=24,
    )

    result = FreshnessCheck().run(_ctx(df, config))

    assert result.severity == "error"
    assert result.payload["per_column"][0]["is_future"] is True


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
