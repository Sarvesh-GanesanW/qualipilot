"""Engine parity tests — polars and pandas must agree on every metric."""

from __future__ import annotations

import pandas as pd
import pytest

from datapilot.engines import PandasEngine, PolarsEngine, build_engine


@pytest.mark.parametrize(
    "engine_cls",
    [PolarsEngine, PandasEngine],
)
def test_row_and_column_count(
    engine_cls: type, dirty_pandas: pd.DataFrame
) -> None:
    engine = engine_cls.from_any(dirty_pandas)
    assert engine.row_count() == len(dirty_pandas)
    assert set(engine.columns()) == set(dirty_pandas.columns)


def test_null_counts_match(dirty_pandas: pd.DataFrame) -> None:
    polars_counts = PolarsEngine.from_any(dirty_pandas).null_counts()
    pandas_counts = PandasEngine.from_any(dirty_pandas).null_counts()
    assert polars_counts == pandas_counts


def test_duplicate_count_matches(dirty_pandas: pd.DataFrame) -> None:
    polars_dup = PolarsEngine.from_any(dirty_pandas).duplicate_count()
    pandas_dup = PandasEngine.from_any(dirty_pandas).duplicate_count()
    assert polars_dup == pandas_dup
    assert polars_dup >= 2  # we duplicated a row in the fixture


def test_quantile_parity(dirty_pandas: pd.DataFrame) -> None:
    polars_q = PolarsEngine.from_any(dirty_pandas).quantiles(
        ["amount"], qs=(0.25, 0.75)
    )
    pandas_q = PandasEngine.from_any(dirty_pandas).quantiles(
        ["amount"], qs=(0.25, 0.75)
    )
    # polars vs pandas quantile algorithms can disagree slightly on
    # small samples, so allow a tolerance
    for q in (0.25, 0.75):
        assert abs(polars_q["amount"][q] - pandas_q["amount"][q]) < 1.0


def test_count_outside_bounds(dirty_pandas: pd.DataFrame) -> None:
    eng = PolarsEngine.from_any(dirty_pandas)
    assert eng.count_outside("amount", 0, 100) == 1


def test_build_engine_dispatch(dirty_pandas: pd.DataFrame) -> None:
    # auto on a pandas df should upgrade to polars for single-node speed
    eng = build_engine(dirty_pandas, kind="auto")
    assert eng.name == "polars"

    eng = build_engine(dirty_pandas, kind="pandas")
    assert eng.name == "pandas"


def test_read_csv_path(tmp_csv) -> None:
    eng = build_engine(str(tmp_csv), kind="polars")
    assert eng.row_count() > 0
