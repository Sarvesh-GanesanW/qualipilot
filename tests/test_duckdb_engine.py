"""Tests for the DuckDB-backed Engine.

DuckDB is the in-process columnar SQL engine; we make sure every
``Engine`` method we call from checks works against it and matches
the polars/pandas reference numbers on the same fixture.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

duckdb = pytest.importorskip("duckdb")

from qualipilot import DataQualityChecker, QualipilotConfig  # noqa: E402
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


def test_rejects_polars_int128() -> None:
    if not hasattr(pl, "Int128"):
        pytest.skip("Polars does not expose Int128")
    frame = pl.DataFrame({"value": pl.Series([2**100], dtype=pl.Int128)})

    with pytest.raises(ValueError, match="does not support 128-bit"):
        DuckDBEngine.from_any(frame)


def test_from_csv_path(tmp_csv) -> None:
    eng = DuckDBEngine.from_any(str(tmp_csv))
    assert eng.row_count() > 0


def test_csv_null_tokens_match(tmp_path: Path) -> None:
    path = tmp_path / "tokens.csv"
    path.write_text('value\nNA\n""\nfoo\n', encoding="utf-8")

    engine = DuckDBEngine.from_any(path)

    assert engine.null_counts() == {"value": 1}
    assert engine.top_values("value") == [("NA", 1), ("foo", 1)]


def test_json_array_files_are_not_supported(tmp_path: Path) -> None:
    path = tmp_path / "records.json"
    pd.DataFrame({"id": [1, 2]}).to_json(path)

    with pytest.raises(ValueError, match="unsupported file type"):
        DuckDBEngine.from_any(path)


def test_json_array_cannot_disguise_itself_as_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text('[{"id":1},{"id":2}]\n', encoding="utf-8")

    with pytest.raises(ValueError, match="must be an object"):
        DuckDBEngine.from_any(path)


def test_jsonl_duplicate_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicates.jsonl"
    path.write_text('{"id":1,"id":2}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate object keys"):
        DuckDBEngine.from_any(path)


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


def test_runtime_relation_is_checked_without_owning_its_connection() -> None:
    connection = duckdb.connect()
    relation = connection.sql(
        "SELECT * FROM (VALUES (1, 5.0), (1, 50.0), (2, NULL)) "
        "AS quality_input(id, amount)"
    )

    engine = build_engine(relation, duckdb_connection=connection)
    assert engine.name == "duckdb"
    assert engine.row_count() == 3
    assert engine.null_counts() == {"id": 0, "amount": 1}
    assert engine.count_outside("amount", 0, 10) == 1
    assert engine.sample_outside("amount", 0, 10, 1) == [
        {"id": 1, "amount": 50.0}
    ]

    engine.close()
    assert connection.sql("SELECT 1").fetchone() == (1,)
    connection.close()


def test_relation_rejects_a_different_borrowed_connection() -> None:
    source_connection = duckdb.connect()
    other_connection = duckdb.connect()
    try:
        relation = source_connection.sql("SELECT 1 AS id")

        with pytest.raises(
            duckdb.InvalidInputException,
            match="another Connection",
        ):
            build_engine(
                relation,
                duckdb_connection=other_connection,
            )
    finally:
        source_connection.close()
        other_connection.close()


def test_checker_uses_borrowed_duckdb_connection(
    dirty_pandas: pd.DataFrame,
) -> None:
    connection = duckdb.connect()
    try:
        with DataQualityChecker(
            dirty_pandas,
            QualipilotConfig(engine="duckdb"),
            duckdb_connection=connection,
        ) as checker:
            assert checker.run(include_llm=False).dataset.engine == "duckdb"

        assert connection.sql("SELECT 1").fetchone() == (1,)
    finally:
        connection.close()


def test_borrowed_connection_supports_in_memory_inputs() -> None:
    connection = duckdb.connect()
    inputs = [
        pd.DataFrame({"id": [1, 2]}),
        pl.DataFrame({"id": [1, 2]}),
        pa.table({"id": [1, 2]}),
    ]
    try:
        for data in inputs:
            with DuckDBEngine.from_any(
                data,
                duckdb_connection=connection,
            ) as engine:
                assert engine.row_count() == 2
        assert connection.sql("SELECT 1").fetchone() == (1,)
    finally:
        connection.close()


def test_borrowed_connection_supports_file_inputs(tmp_csv: Path) -> None:
    connection = duckdb.connect()
    try:
        with DuckDBEngine.from_any(
            tmp_csv,
            duckdb_connection=connection,
        ) as engine:
            assert engine.row_count() > 0
        assert connection.sql("SELECT 1").fetchone() == (1,)
    finally:
        connection.close()


def test_relation_binder_ignores_quoted_question_marks() -> None:
    connection = duckdb.connect()
    relation = connection.sql('SELECT 1 AS "amount?"')

    with build_engine(relation) as engine:
        assert engine.null_counts() == {"amount?": 0}
        assert engine.top_values("amount?", 1) == [("1", 1)]

    connection.close()


def test_max_datetime(stale_timestamps_pandas: pd.DataFrame) -> None:
    eng = DuckDBEngine.from_any(stale_timestamps_pandas)
    assert eng.max_datetime("event_ts") is not None


def test_private_connections_isolate_engines(
    dirty_pandas: pd.DataFrame,
) -> None:
    a = DuckDBEngine.from_any(dirty_pandas)
    b = DuckDBEngine.from_any(dirty_pandas)
    assert a._con is not b._con
    assert a.row_count() == b.row_count() == len(dirty_pandas)


def test_malicious_column_name_cannot_inject_sql() -> None:
    column = 'x") FROM (SELECT 1 AS x); SELECT 999; --'
    frame = pd.DataFrame([[2]], columns=[column])
    engine = DuckDBEngine.from_any(frame)
    assert engine.distinct_count(column) == 1
    assert engine.null_counts() == {column: 0}


def test_native_nan_is_treated_as_missing() -> None:
    frame = pl.DataFrame({"value": [1.0, float("nan")]})
    engine = DuckDBEngine.from_any(frame)
    assert engine.null_counts() == {"value": 1}
    assert engine.distinct_count("value") == 1
    assert engine.top_values("value") == [("1.0", 1)]


def test_nan_and_null_are_the_same_duplicate_key() -> None:
    frame = pl.DataFrame(
        {"value": [None, float("nan")], "other": [1, 1]},
        schema={"value": pl.Float64, "other": pl.Int64},
    )
    engine = DuckDBEngine.from_any(frame)

    assert engine.duplicate_count() == 2
    assert len(engine.sample_duplicates(10)) == 2


def test_nested_columns_are_rejected() -> None:
    frame = pl.DataFrame({"id": [1], "nested": [["value"]]})

    with pytest.raises(TypeError, match=r"flatten.*nested"):
        DuckDBEngine.from_any(frame)


def test_parquet_nan_is_excluded_from_ranges(tmp_path: Path) -> None:
    path = tmp_path / "values.parquet"
    pl.DataFrame({"value": [float("nan"), 0.0, 100.0]}).write_parquet(path)
    engine = DuckDBEngine.from_any(path)

    assert engine.count_outside("value", -1, 10) == 1
    assert engine.sample_outside("value", -1, 10, 10) == [{"value": 100.0}]


def test_duplicate_sample_does_not_collide_with_n_column() -> None:
    frame = pd.DataFrame({"value": [1, 1], "_n": ["original", "original"]})
    engine = DuckDBEngine.from_any(frame)
    assert engine.sample_duplicates(5) == frame.to_dict(orient="records")


def test_timestamp_ns_is_datetime() -> None:
    table = pa.table({"event_ts": pa.array([], type=pa.timestamp("ns"))})
    engine = DuckDBEngine.from_any(table)
    assert engine.datetime_columns() == ["event_ts"]


def test_context_manager_closes_connection(
    dirty_pandas: pd.DataFrame,
) -> None:
    with DuckDBEngine.from_any(dirty_pandas, threads=1) as engine:
        assert engine.row_count() == len(dirty_pandas)
    with pytest.raises(duckdb.ConnectionException):
        engine.row_count()
