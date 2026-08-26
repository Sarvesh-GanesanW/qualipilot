"""Engine parity tests — polars and pandas must agree on every metric."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from qualipilot.checker import DataQualityChecker
from qualipilot.engines import PandasEngine, PolarsEngine, build_engine
from qualipilot.engines.dask_engine import DaskEngine
from qualipilot.engines.duckdb_engine import DuckDBEngine
from qualipilot.models.config import CheckConfig, QualipilotConfig


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


@pytest.mark.parametrize(
    "engine_cls",
    [PolarsEngine, PandasEngine, DuckDBEngine, DaskEngine],
)
def test_batched_counts_match_scalar_metrics(engine_cls: type) -> None:
    frame = pd.DataFrame(
        {
            "first": [0.0, 20.0, float("nan"), None],
            "second": [-5, 5, 50, 5],
            "label": ["a", "a", None, "b"],
        }
    )
    engine = engine_cls.from_any(frame)

    assert engine.distinct_counts(["first", "second", "label"]) == {
        "first": 2,
        "second": 3,
        "label": 2,
    }
    assert engine.counts_outside({"first": (0, 10), "second": (0, 10)}) == {
        "first": 1,
        "second": 2,
    }
    assert engine.distinct_count("label") == 2
    assert engine.count_outside("second", 0, 10) == 2


def test_build_engine_dispatch(dirty_pandas: pd.DataFrame) -> None:
    # auto on a pandas df should upgrade to polars for single-node speed
    eng = build_engine(dirty_pandas, kind="auto")
    assert eng.name == "polars"

    eng = build_engine(dirty_pandas, kind="pandas")
    assert eng.name == "pandas"


def test_build_engine_rejects_execution_context_backend_mismatches(
    dirty_pandas: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match=r"spark_session.*spark engine"):
        build_engine(
            dirty_pandas,
            kind="pandas",
            spark_session=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(
        ValueError,
        match=r"duckdb_connection.*duckdb engine",
    ):
        build_engine(
            dirty_pandas,
            kind="pandas",
            duckdb_connection=object(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "kind", ["auto", "polars", "pandas", "duckdb", "dask"]
)
def test_arrow_table_dispatch(kind: str) -> None:
    table = pa.table({"id": [1, 2], "name": ["Alice", "Bob"]})

    engine = build_engine(table, kind=kind)
    try:
        assert engine.row_count() == 2
        assert engine.columns() == ["id", "name"]
    finally:
        close = getattr(engine, "close", None)
        if close is not None:
            close()


def test_read_csv_path(tmp_csv) -> None:
    eng = build_engine(str(tmp_csv), kind="polars")
    assert eng.row_count() > 0


@pytest.mark.parametrize("engine_cls", [PolarsEngine, PandasEngine])
def test_csv_null_tokens_match(
    tmp_path: Path,
    engine_cls: type[PolarsEngine] | type[PandasEngine],
) -> None:
    path = tmp_path / "tokens.csv"
    path.write_text('value\nNA\n""\nfoo\n', encoding="utf-8")

    engine = engine_cls.from_any(path)

    assert engine.null_counts() == {"value": 1}
    assert engine.top_values("value") == [("NA", 1), ("foo", 1)]


@pytest.mark.parametrize("engine_cls", [PolarsEngine, PandasEngine])
def test_csv_duplicate_headers_are_rejected(
    tmp_path: Path,
    engine_cls: type[PolarsEngine] | type[PandasEngine],
) -> None:
    path = tmp_path / "duplicates.csv"
    path.write_text("value,value\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="column names must be unique"):
        engine_cls.from_any(path)


@pytest.mark.parametrize(
    "engine_cls",
    [PolarsEngine, PandasEngine, DuckDBEngine, DaskEngine],
)
@pytest.mark.parametrize(
    ("header", "message"),
    [
        ("id,,value", "must not be blank"),
        ("id, value", "surrounding whitespace"),
    ],
)
def test_ambiguous_csv_headers_are_rejected(
    tmp_path: Path,
    engine_cls: type,
    header: str,
    message: str,
) -> None:
    path = tmp_path / "ambiguous.csv"
    path.write_text(f"{header}\n1,2,3\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        engine_cls.from_any(path)


@pytest.mark.parametrize(
    "engine_cls",
    [PolarsEngine, PandasEngine, DuckDBEngine, DaskEngine],
)
@pytest.mark.parametrize(
    "row",
    ["1,Alice,extra", "1"],
)
def test_inconsistent_csv_row_width_is_rejected(
    tmp_path: Path,
    engine_cls: type,
    row: str,
) -> None:
    path = tmp_path / "malformed.csv"
    path.write_text(f"id,name\n{row}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"has \d+ fields; expected 2"):
        engine_cls.from_any(path)


@pytest.mark.parametrize(
    "engine_cls",
    [PolarsEngine, PandasEngine, DuckDBEngine, DaskEngine],
)
def test_blank_csv_records_are_rejected(
    tmp_path: Path,
    engine_cls: type,
) -> None:
    path = tmp_path / "blank-row.csv"
    path.write_text("id,name\n1,Alice\n\n2,Bob\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must not be blank"):
        engine_cls.from_any(path)


@pytest.mark.parametrize(
    "engine_cls",
    [PolarsEngine, PandasEngine, DuckDBEngine, DaskEngine],
)
@pytest.mark.parametrize(
    "row",
    ['1,"Alice', '1,"Alice"junk'],
)
def test_malformed_csv_quotes_are_rejected(
    tmp_path: Path,
    engine_cls: type,
    row: str,
) -> None:
    path = tmp_path / "malformed.csv"
    path.write_text(f"id,name\n{row}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid CSV syntax"):
        engine_cls.from_any(path)


@pytest.mark.parametrize(
    "engine_cls",
    [PolarsEngine, PandasEngine, DuckDBEngine, DaskEngine],
)
def test_csv_nul_bytes_are_rejected(
    tmp_path: Path,
    engine_cls: type,
) -> None:
    path = tmp_path / "nul.csv"
    path.write_text("id,name\n1,Ali\0ce\n", encoding="utf-8")

    with pytest.raises(ValueError, match="NUL bytes"):
        engine_cls.from_any(path)


@pytest.mark.parametrize(
    "engine_cls",
    [PolarsEngine, PandasEngine, DuckDBEngine, DaskEngine],
)
def test_csv_delimiter_is_always_comma(
    tmp_path: Path,
    engine_cls: type,
) -> None:
    path = tmp_path / "semicolon.csv"
    path.write_text("id;name\n1;Alice\n", encoding="utf-8")

    engine = engine_cls.from_any(path)

    assert engine.columns() == ["id;name"]


@pytest.mark.parametrize(
    "engine_cls",
    [PolarsEngine, PandasEngine, DuckDBEngine, DaskEngine],
)
@pytest.mark.parametrize("content", ["", "\n\n", "{}\n"])
def test_jsonl_without_a_schema_is_rejected(
    tmp_path: Path,
    engine_cls: type,
    content: str,
) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"at least one (?:object|column)",
    ):
        engine_cls.from_any(path)


@pytest.mark.parametrize(
    "engine_cls",
    [PolarsEngine, PandasEngine, DuckDBEngine, DaskEngine],
)
def test_jsonl_all_null_columns_keep_their_schema(
    tmp_path: Path,
    engine_cls: type,
) -> None:
    path = tmp_path / "nulls.jsonl"
    path.write_text('{"value":null}\n{"value":null}\n', encoding="utf-8")

    engine = engine_cls.from_any(path)

    assert engine.columns() == ["value"]
    assert engine.null_counts() == {"value": 2}


@pytest.mark.parametrize(
    "engine_cls",
    [PolarsEngine, PandasEngine, DuckDBEngine, DaskEngine],
)
def test_jsonl_unpaired_surrogates_are_rejected(
    tmp_path: Path,
    engine_cls: type,
) -> None:
    path = tmp_path / "surrogate.jsonl"
    path.write_text('{"value":"\\ud800"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="valid Unicode"):
        engine_cls.from_any(path)


@pytest.mark.parametrize("engine_cls", [PolarsEngine, PandasEngine])
def test_json_array_files_are_not_supported(
    tmp_path: Path,
    engine_cls: type[PolarsEngine] | type[PandasEngine],
) -> None:
    path = tmp_path / "records.json"
    pd.DataFrame({"id": [1, 2]}).to_json(path)

    with pytest.raises(ValueError, match="unsupported file type"):
        engine_cls.from_any(path)


@pytest.mark.parametrize("engine_cls", [PolarsEngine, PandasEngine])
def test_jsonl_duplicate_keys_are_rejected(
    tmp_path: Path,
    engine_cls: type[PolarsEngine] | type[PandasEngine],
) -> None:
    path = tmp_path / "duplicates.jsonl"
    path.write_text('{"id":1,"id":2}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate object keys"):
        engine_cls.from_any(path)


@pytest.mark.parametrize(
    "engine_cls",
    [PolarsEngine, PandasEngine, DuckDBEngine, DaskEngine],
)
@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_json_numeric_constants_are_rejected(
    tmp_path: Path,
    engine_cls: type,
    constant: str,
) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(f'{{"value":{constant}}}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSONL record"):
        engine_cls.from_any(path)


@pytest.mark.parametrize(
    "engine_cls",
    [PolarsEngine, PandasEngine, DuckDBEngine, DaskEngine],
)
def test_jsonl_records_require_one_stable_schema(
    tmp_path: Path,
    engine_cls: type,
) -> None:
    path = tmp_path / "late-column.jsonl"
    path.write_text(
        "".join('{"a":1}\n' for _ in range(101)) + '{"a":1,"b":2}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must use identical columns"):
        engine_cls.from_any(path)


@pytest.mark.parametrize(
    "engine_cls",
    [PolarsEngine, PandasEngine, DuckDBEngine, DaskEngine],
)
def test_jsonl_string_tokens_are_preserved(
    tmp_path: Path,
    engine_cls: type,
) -> None:
    path = tmp_path / "strings.jsonl"
    path.write_text(
        '{"code":"001","sentinel":"NaN","date":"1900-01-01"}\n'
        '{"code":"1","sentinel":"Infinity","date":"2020-01-01"}\n',
        encoding="utf-8",
    )

    engine = engine_cls.from_any(path)

    assert engine.distinct_count("code") == 2
    assert engine.null_counts()["sentinel"] == 0
    assert set(engine.top_values("date")) == {
        ("1900-01-01", 1),
        ("2020-01-01", 1),
    }


@pytest.mark.parametrize(
    "engine_cls",
    [PolarsEngine, PandasEngine, DuckDBEngine, DaskEngine],
)
def test_jsonl_rejects_mixed_scalar_types(
    tmp_path: Path,
    engine_cls: type,
) -> None:
    path = tmp_path / "mixed.jsonl"
    path.write_text('{"value":"1"}\n{"value":1}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="must use one scalar type"):
        engine_cls.from_any(path)


def test_nulls_are_excluded_from_cardinality_and_top_values() -> None:
    frame = pd.DataFrame({"value": ["a", None, "a"]})
    for engine_cls in (PolarsEngine, PandasEngine):
        engine = engine_cls.from_any(frame)
        assert engine.distinct_count("value") == 1
        assert engine.top_values("value") == [("a", 2)]


def test_native_nan_is_missing_across_local_engines() -> None:
    frame = pl.DataFrame({"value": [1.0, float("nan")]})
    for engine_cls in (PolarsEngine, PandasEngine):
        engine = engine_cls.from_any(frame)
        assert engine.null_counts() == {"value": 1}
        assert engine.distinct_count("value") == 1
        assert engine.top_values("value") == [("1.0", 1)]


@pytest.mark.parametrize("engine_cls", [PandasEngine, DaskEngine])
def test_timedelta_columns_are_not_profiled_as_numeric(
    engine_cls: type,
) -> None:
    frame = pd.DataFrame({"duration": pd.to_timedelta([1, 2], unit="D")})

    assert engine_cls.from_any(frame).numeric_columns() == []


@pytest.mark.parametrize(
    "engine_cls",
    [PolarsEngine, PandasEngine, DuckDBEngine, DaskEngine],
)
def test_decimal_parquet_columns_are_numeric(
    tmp_path: Path,
    engine_cls: type,
) -> None:
    path = tmp_path / "decimal.parquet"
    pq.write_table(
        pa.table(
            {
                "amount": pa.array(
                    [Decimal("50.00"), Decimal("200.00")],
                    type=pa.decimal128(10, 2),
                )
            }
        ),
        path,
    )

    engine = engine_cls.from_any(path)

    assert engine.numeric_columns() == ["amount"]
    assert engine.count_outside("amount", 0, 100) == 1
    quantiles = engine.quantiles(["amount"], (0.25, 0.75))
    assert quantiles["amount"][0.25] < quantiles["amount"][0.75]


@pytest.mark.parametrize("kind", ["polars", "pandas", "duckdb", "dask"])
def test_date32_parquet_supports_default_freshness(
    tmp_path: Path,
    kind: str,
) -> None:
    path = tmp_path / "date.parquet"
    pq.write_table(
        pa.table(
            {
                "event_date": pa.array(
                    [datetime.now(UTC).date()],
                    type=pa.date32(),
                )
            }
        ),
        path,
    )
    config = QualipilotConfig(
        engine=kind,  # type: ignore[arg-type]
        checks=CheckConfig(
            freshness=True,
            freshness_max_age_hours=48,
        ),
    )

    report = DataQualityChecker(path, config).run()
    freshness = next(
        result for result in report.results if result.name == "freshness"
    )

    assert freshness.status == "completed"
    assert freshness.severity == "ok"


def test_string_datetime_max_uses_instants_across_engines(
    tmp_path: Path,
) -> None:
    path = tmp_path / "offsets.csv"
    path.write_text(
        "event_ts\n2026-08-26T00:00:00+14:00\n2026-08-25T23:30:00-12:00\n",
        encoding="utf-8",
    )
    expected = datetime(2026, 8, 26, 11, 30, tzinfo=UTC)

    for engine_cls in (
        PolarsEngine,
        PandasEngine,
        DuckDBEngine,
        DaskEngine,
    ):
        engine = engine_cls.from_any(path)
        try:
            assert engine.max_datetime_instant("event_ts") == expected
        finally:
            close = getattr(engine, "close", None)
            if close is not None:
                close()


def test_object_numeric_and_date_families_match_across_engines() -> None:
    frame = pd.DataFrame(
        {
            "amount": pd.Series([1, 2], dtype=object),
            "event_date": pd.Series(
                [date(2026, 8, 25), date(2026, 8, 26)],
                dtype=object,
            ),
        }
    )

    for engine_cls in (
        PolarsEngine,
        PandasEngine,
        DuckDBEngine,
        DaskEngine,
    ):
        engine = engine_cls.from_any(frame)
        try:
            assert engine.dtype_family("amount") == "integer"
            assert engine.numeric_columns() == ["amount"]
            assert engine.count_outside("amount", 0, 10) == 0
            assert engine.dtype_family("event_date") == "date"
            assert engine.datetime_columns() == ["event_date"]
        finally:
            close = getattr(engine, "close", None)
            if close is not None:
                close()


def test_portable_dtype_contract_matches_every_engine() -> None:
    frame = pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["Alice", "Bob"],
            "active": [True, False],
            "segment": pd.Series(["retail", "business"], dtype="category"),
            "event_ts": pd.to_datetime(
                ["2026-08-25T00:00:00Z", "2026-08-26T00:00:00Z"]
            ),
        }
    )
    checks = CheckConfig(
        missing_values=False,
        duplicates=False,
        data_types=False,
        outliers=False,
        ranges=False,
        cardinality=False,
        expected_dtypes={
            "id": "integer",
            "name": "string",
            "active": "boolean",
            "segment": "categorical",
            "event_ts": "datetime",
        },
    )

    for kind in ("polars", "pandas", "duckdb", "dask"):
        with DataQualityChecker(
            frame,
            QualipilotConfig(engine=kind, checks=checks),  # type: ignore[arg-type]
        ) as checker:
            report = checker.run(include_llm=False)
        contract = next(
            result
            for result in report.results
            if result.name == "dataset_contract"
        )
        assert contract.severity == "ok"
        assert contract.payload["dtype_mismatches"] == []


def test_nan_and_null_are_the_same_duplicate_key() -> None:
    frame = pl.DataFrame(
        {"value": [None, float("nan")], "other": [1, 1]},
        schema={"value": pl.Float64, "other": pl.Int64},
    )
    for engine_cls in (PolarsEngine, PandasEngine):
        engine = engine_cls.from_any(frame)
        assert engine.duplicate_count() == 2
        assert len(engine.sample_duplicates(10)) == 2


def test_object_nulls_are_the_same_pandas_duplicate_key() -> None:
    frame = pd.DataFrame(
        {"value": pd.Series([None, float("nan")], dtype=object)}
    )
    engine = PandasEngine.from_any(frame)

    assert engine.duplicate_count(["value"]) == 2
    assert len(engine.sample_duplicates(10, ["value"])) == 2


@pytest.mark.parametrize("engine_cls", [PolarsEngine, PandasEngine])
def test_nested_columns_are_rejected(engine_cls: type) -> None:
    frame = pd.DataFrame({"id": [1], "nested": [["value"]]})

    with pytest.raises(TypeError, match=r"flatten.*nested"):
        engine_cls.from_any(frame)


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        (["Name", "name"], "unique"),
        (["nul\0name"], "NUL"),
        (["\ud800"], "valid Unicode"),
    ],
)
@pytest.mark.parametrize(
    "engine_cls",
    [PolarsEngine, PandasEngine, DuckDBEngine, DaskEngine],
)
def test_ambiguous_in_memory_column_names_are_rejected(
    columns: list[str],
    message: str,
    engine_cls: type,
) -> None:
    frame = pd.DataFrame([[1] * len(columns)], columns=columns)

    with pytest.raises(ValueError, match=message):
        engine_cls.from_any(frame)


@pytest.mark.parametrize(
    "kind", ["auto", "polars", "pandas", "duckdb", "dask"]
)
@pytest.mark.parametrize(
    "series",
    [
        pd.Series([1, "1"], dtype=object),
        pd.Series([b"a", "a"], dtype=object),
        pd.Series(pd.Categorical([1, "1"])),
        pd.Series([1 + 2j, 2 + 3j]),
        pd.Series(pd.period_range("2026-01", periods=2, freq="M")),
        pd.Series(pd.arrays.IntervalArray.from_tuples([(0, 1), (1, 2)])),
    ],
    ids=[
        "mixed-scalars",
        "bytes-and-text",
        "mixed-category",
        "complex",
        "period",
        "interval",
    ],
)
def test_nonportable_pandas_columns_are_rejected(
    kind: str,
    series: pd.Series,
) -> None:
    frame = pd.DataFrame({"value": series})

    with pytest.raises(TypeError, match="unsupported pandas column types"):
        build_engine(frame, kind=kind)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("values", "arrow_type"),
    [
        ([[1], [1]], pa.list_(pa.int64())),
        (
            [{"value": 1}, {"value": 1}],
            pa.struct([("value", pa.int64())]),
        ),
    ],
)
def test_pandas_arrow_nested_columns_are_rejected(
    values: list[object],
    arrow_type: pa.DataType,
) -> None:
    frame = pd.DataFrame(
        {
            "nested": pd.Series(
                values,
                dtype=pd.ArrowDtype(arrow_type),
            )
        }
    )

    with pytest.raises(TypeError, match=r"flatten.*nested"):
        PandasEngine.from_any(frame)


def test_local_engines_use_linear_quantiles() -> None:
    frame = pd.DataFrame({"value": [0, 0, 1, 3]})
    for engine_cls in (PolarsEngine, PandasEngine):
        quantiles = engine_cls.from_any(frame).quantiles(["value"])
        assert quantiles["value"] == {0.25: 0.0, 0.75: 1.5}


def test_empty_duplicate_subset_means_all_columns() -> None:
    frame = pd.DataFrame({"value": [1, 1]})
    for engine_cls in (PolarsEngine, PandasEngine):
        assert engine_cls.from_any(frame).duplicate_count([]) == 2


def test_polars_ranges_exclude_nan() -> None:
    frame = pl.DataFrame({"value": [float("nan"), 0.0, 100.0]})
    engine = PolarsEngine.from_any(frame)

    assert engine.count_outside("value", -1, 10) == 1
    assert engine.sample_outside("value", -1, 10, 10) == [{"value": 100.0}]
