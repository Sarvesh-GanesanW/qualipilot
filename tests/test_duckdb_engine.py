"""Tests for the DuckDB-backed Engine.

DuckDB is the in-process columnar SQL engine; we make sure every
``Engine`` method we call from checks works against it and matches
the polars/pandas reference numbers on the same fixture.
"""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

duckdb = pytest.importorskip("duckdb")

from qualipilot.engines import build_engine  # noqa: E402
from qualipilot.engines.duckdb_engine import DuckDBEngine  # noqa: E402
from qualipilot.engines.pandas_engine import PandasEngine  # noqa: E402
from qualipilot.engines.polars_engine import PolarsEngine  # noqa: E402


def test_from_pandas_dataframe(dirty_pandas: pd.DataFrame) -> None:
    eng = DuckDBEngine.from_any(dirty_pandas)
    assert eng.name == "duckdb"
    assert eng.row_count() == len(dirty_pandas)
    assert set(eng.columns()) == set(dirty_pandas.columns)


def test_from_polars_dataframe(dirty_pandas: pd.DataFrame) -> None:
    eng = DuckDBEngine.from_any(pl.from_pandas(dirty_pandas))
    assert eng.row_count() == len(dirty_pandas)


def test_from_csv_path(tmp_csv) -> None:
    eng = DuckDBEngine.from_any(str(tmp_csv))
    assert eng.row_count() > 0


def test_unsupported_input_type_raises() -> None:
    with pytest.raises(TypeError):
        DuckDBEngine.from_any(42)


def test_unsupported_file_extension_raises(tmp_path) -> None:
    bad = tmp_path / "not-supported.xlsx"
    bad.write_text("ignored", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported file type"):
        DuckDBEngine.from_any(str(bad))


def test_dtypes_and_numeric_columns(dirty_pandas: pd.DataFrame) -> None:
    eng = DuckDBEngine.from_any(dirty_pandas)
    dtypes = eng.dtypes()
    assert set(dtypes.keys()) == set(dirty_pandas.columns)
    assert "amount" in eng.numeric_columns()
    assert "category" not in eng.numeric_columns()


def test_null_counts_parity(dirty_pandas: pd.DataFrame) -> None:
    duck = DuckDBEngine.from_any(dirty_pandas).null_counts()
    pol = PolarsEngine.from_any(dirty_pandas).null_counts()
    assert duck == pol


def test_distinct_count(dirty_pandas: pd.DataFrame) -> None:
    eng = DuckDBEngine.from_any(dirty_pandas)
    # category has 3 distinct values: x, y, z
    assert eng.distinct_count("category") == 3


def test_top_values(dirty_pandas: pd.DataFrame) -> None:
    eng = DuckDBEngine.from_any(dirty_pandas)
    top = eng.top_values("category", n=2)
    assert len(top) == 2
    # z is the most frequent in the fixture
    assert top[0][0] == "z"
    assert top[0][1] >= top[1][1]


def test_quantiles_parity(dirty_pandas: pd.DataFrame) -> None:
    duck_q = DuckDBEngine.from_any(dirty_pandas).quantiles(
        ["amount"], qs=(0.25, 0.75)
    )
    pd_q = PandasEngine.from_any(dirty_pandas).quantiles(
        ["amount"], qs=(0.25, 0.75)
    )
    for q in (0.25, 0.75):
        assert abs(duck_q["amount"][q] - pd_q["amount"][q]) < 1.0


def test_quantiles_empty_inputs(dirty_pandas: pd.DataFrame) -> None:
    eng = DuckDBEngine.from_any(dirty_pandas)
    assert eng.quantiles([], qs=(0.5,)) == {}
    assert eng.quantiles(["amount"], qs=()) == {}


def test_describe(dirty_pandas: pd.DataFrame) -> None:
    eng = DuckDBEngine.from_any(dirty_pandas)
    desc = eng.describe()
    assert "amount" in desc
    assert "min" in desc["amount"] or "mean" in desc["amount"]


def test_duplicate_count_global(dirty_pandas: pd.DataFrame) -> None:
    duck = DuckDBEngine.from_any(dirty_pandas).duplicate_count()
    pol = PolarsEngine.from_any(dirty_pandas).duplicate_count()
    assert duck == pol
    assert duck >= 2


def test_sample_duplicates_returns_dicts(
    dirty_pandas: pd.DataFrame,
) -> None:
    eng = DuckDBEngine.from_any(dirty_pandas)
    sample = eng.sample_duplicates(n=5)
    assert isinstance(sample, list)
    if sample:
        assert isinstance(sample[0], dict)


def test_count_outside(dirty_pandas: pd.DataFrame) -> None:
    eng = DuckDBEngine.from_any(dirty_pandas)
    # the fixture has one outlier amount of 10_000
    assert eng.count_outside("amount", 0, 100) == 1


def test_sample_outside(dirty_pandas: pd.DataFrame) -> None:
    eng = DuckDBEngine.from_any(dirty_pandas)
    rows = eng.sample_outside("amount", 0, 100, 10)
    assert any(r.get("amount", 0) > 100 for r in rows)


def test_build_engine_dispatches_duckdb(
    dirty_pandas: pd.DataFrame,
) -> None:
    eng = build_engine(dirty_pandas, kind="duckdb")
    assert eng.name == "duckdb"


def test_max_datetime(stale_timestamps_pandas: pd.DataFrame) -> None:
    # duckdb requires pytz for tz-aware datetime values; skip if absent
    pytest.importorskip("pytz")
    eng = DuckDBEngine.from_any(stale_timestamps_pandas)
    assert eng.max_datetime("event_ts") is not None


def test_unique_view_names_per_engine(dirty_pandas: pd.DataFrame) -> None:
    """Two engines in the same process must not collide on view name."""
    a = DuckDBEngine.from_any(dirty_pandas)
    b = DuckDBEngine.from_any(dirty_pandas)
    # different view names so the registrations stay isolated
    assert a._view != b._view
    # both still functional
    assert a.row_count() == b.row_count() == len(dirty_pandas)
