"""Runtime contract tests for the optional Spark engine."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyspark")

from pyspark.sql import SparkSession

from qualipilot import CheckConfig, DataQualityChecker, QualipilotConfig
from qualipilot.checks import CheckContext, OutliersCheck
from qualipilot.engines.spark_engine import SparkEngine
from qualipilot.models.config import (
    ComparisonRule,
    ForeignKeyRule,
    QualityContract,
)


@pytest.fixture(scope="module")
def spark() -> Iterator[SparkSession]:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("PYSPARK_PYTHON", sys.executable)
        session = (
            SparkSession.builder.master("local[1]")
            .appName("qualipilot-tests")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        yield session
        session.stop()


def test_spark_engine_contract(spark: SparkSession) -> None:
    engine = SparkEngine(
        spark.createDataFrame(
            [(1, 10.0), (1, 10.0), (2, None)],
            ["id", "amount"],
        )
    )

    assert engine.row_count() == 3
    assert engine.columns() == ["id", "amount"]
    assert engine.null_counts() == {"id": 0, "amount": 1}
    assert engine.duplicate_count() == 2


def test_checker_uses_supplied_spark_session(spark: SparkSession) -> None:
    supplied_session = spark.newSession()

    checker = DataQualityChecker(
        pd.DataFrame({"id": [1, 2]}),
        QualipilotConfig(engine="spark"),
        spark_session=supplied_session,
    )

    assert checker.engine._df.sparkSession is supplied_session
    assert checker.run(include_llm=False).dataset.engine == "spark"


def test_spark_dataframe_rejects_a_different_session(
    spark: SparkSession,
) -> None:
    frame = spark.createDataFrame([(1,)], ["id"])

    with pytest.raises(ValueError, match="different Spark session"):
        SparkEngine.from_any(
            frame,
            spark_session=spark.newSession(),
        )


def test_spark_ranges_exclude_nan(spark: SparkSession) -> None:
    engine = SparkEngine(
        spark.createDataFrame(
            [(float("nan"),), (0.0,), (100.0,)],
            ["value"],
        )
    )

    assert engine.count_outside("value", -1, 10) == 1
    assert engine.sample_outside("value", -1, 10, 10) == [{"value": 100.0}]


def test_spark_treats_nan_and_null_as_the_same_duplicate_key(
    spark: SparkSession,
) -> None:
    engine = SparkEngine(
        spark.createDataFrame(
            [(None, 1), (float("nan"), 1)],
            "value double, other long",
        )
    )

    assert engine.duplicate_count() == 2
    assert len(engine.sample_duplicates(10)) == 2


def test_spark_quality_contract_uses_native_filters_and_antijoin(
    spark: SparkSession,
) -> None:
    config = QualipilotConfig(
        engine="spark",
        checks=CheckConfig(
            missing_values=False,
            duplicates=False,
            data_types=False,
            outliers=False,
            ranges=False,
            cardinality=False,
            rule_packs=[
                QualityContract(
                    name="orders",
                    comparisons=[
                        ComparisonRule(
                            name="sequence",
                            left_column="end",
                            operator="ge",
                            right_column="start",
                        )
                    ],
                    foreign_keys=[
                        ForeignKeyRule(
                            name="customer",
                            columns=["customer_id"],
                            reference="customers",
                            reference_columns=["id"],
                        )
                    ],
                )
            ],
        ),
    )
    report = DataQualityChecker(
        spark.createDataFrame(
            [(1, 2, 1), (2, 1, 2)], "customer_id long, end long, start long"
        ),
        config,
        spark_session=spark,
        references={"customers": spark.createDataFrame([(1,)], "id long")},
    ).run(include_llm=False)
    result = next(
        item for item in report.results if item.name == "quality_contract"
    )
    assert result.typed_payload().violation_count == 2


def test_spark_rejects_nested_columns(spark: SparkSession) -> None:
    frame = spark.createDataFrame([(1, ["value"])], ["id", "nested"])

    with pytest.raises(TypeError, match=r"flatten.*nested"):
        SparkEngine(frame)


def test_spark_batches_quantiles_across_columns(
    spark: SparkSession,
) -> None:
    engine = SparkEngine(
        spark.createDataFrame(
            [(0.0, 10.0), (2.0, 20.0), (4.0, 30.0)],
            ["first", "second"],
        )
    )

    assert engine.quantiles(["first", "second"], (0.5,)) == {
        "first": {0.5: 2.0},
        "second": {0.5: 20.0},
    }

    result = OutliersCheck().run(
        CheckContext(engine=engine, config=CheckConfig())
    )
    assert result.payload["quantile_provenance"] == {
        "method": "approximate",
        "relative_error": 0.001,
    }


def test_spark_batched_counts_match_scalar_metrics(
    spark: SparkSession,
) -> None:
    engine = SparkEngine(
        spark.createDataFrame(
            [
                (0.0, -5, "a"),
                (20.0, 5, "a"),
                (float("nan"), 50, None),
                (None, 5, "b"),
            ],
            "first double, second long, label string",
        )
    )

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


@pytest.mark.parametrize("suffix", [".jsonl", ".ndjson"])
def test_spark_json_inputs(
    spark: SparkSession, tmp_path: Path, suffix: str
) -> None:
    path = tmp_path / f"records{suffix}"
    records = [{"id": 1}, {"id": 2}]
    text = "\n".join(json.dumps(record) for record in records)
    path.write_text(text, encoding="utf-8")

    engine = SparkEngine.from_any(path, spark=spark)

    assert engine.row_count() == 2
    assert engine.columns() == ["id"]


def test_spark_json_array_files_are_not_supported(
    spark: SparkSession, tmp_path: Path
) -> None:
    path = tmp_path / "records.json"
    path.write_text('{"id":{"0":1,"1":2}}', encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported file type"):
        SparkEngine.from_any(path, spark=spark)


def test_spark_jsonl_duplicate_keys_are_rejected(
    spark: SparkSession,
    tmp_path: Path,
) -> None:
    path = tmp_path / "duplicates.jsonl"
    path.write_text('{"id":1,"id":2}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate object keys"):
        SparkEngine.from_any(path, spark=spark)


def test_spark_string_datetime_max_uses_offset_instants(
    spark: SparkSession,
) -> None:
    engine = SparkEngine(
        spark.createDataFrame(
            [
                ("2026-08-26T00:00:00+14:00",),
                ("2026-08-25T23:30:00-12:00",),
            ],
            ["event_ts"],
        )
    )

    assert engine.max_datetime_instant("event_ts") == datetime(
        2026,
        8,
        26,
        11,
        30,
        tzinfo=UTC,
    )


def test_spark_accepts_pandas_object_numeric_and_date_columns(
    spark: SparkSession,
) -> None:
    frame = pd.DataFrame(
        {
            "amount": pd.Series([1, 2], dtype=object),
            "event_date": pd.Series(
                [date(2026, 8, 25), date(2026, 8, 26)],
                dtype=object,
            ),
        }
    )

    engine = SparkEngine.from_any(frame, spark_session=spark)

    assert engine.dtype_family("amount") == "integer"
    assert engine.numeric_columns() == ["amount"]
    assert engine.count_outside("amount", 0, 10) == 0
    assert engine.dtype_family("event_date") == "date"
    assert engine.datetime_columns() == ["event_date"]


def test_spark_freshness_fails_closed_without_datetime_columns(
    spark: SparkSession,
) -> None:
    frame = spark.createDataFrame([(1,), (2,)], ["id"])
    config = QualipilotConfig(
        engine="spark",
        checks=CheckConfig(freshness=True),
    )

    report = DataQualityChecker(frame, config).run(include_llm=False)
    freshness = next(
        result for result in report.results if result.name == "freshness"
    )

    assert freshness.status == "completed"
    assert freshness.severity == "error"
    assert freshness.payload["per_column"] == []


@pytest.mark.parametrize(
    "path", ["s3a://bucket/data.csv", "hdfs://data.jsonl"]
)
def test_spark_rejects_remote_text_without_raw_validation(
    spark: SparkSession,
    path: str,
) -> None:
    with pytest.raises(ValueError, match="remote CSV and JSONL"):
        SparkEngine.from_any(path, spark=spark)


@pytest.mark.parametrize(
    ("operator", "expected"),
    [("eq", 1), ("ne", 1), ("gt", 2), ("ge", 1), ("lt", 1), ("le", 0)],
)
def test_spark_contract_comparison_operators_and_null_policy(
    spark: SparkSession, operator: str, expected: int
) -> None:
    config = QualipilotConfig(
        engine="spark",
        checks=CheckConfig(
            missing_values=False,
            duplicates=False,
            data_types=False,
            outliers=False,
            ranges=False,
            cardinality=False,
            rule_packs=[
                QualityContract(
                    name="operators",
                    comparisons=[
                        ComparisonRule(
                            name="comparison",
                            left_column="left",
                            operator=operator,  # type: ignore[arg-type]
                            right_column="right",
                        )
                    ],
                )
            ],
        ),
    )
    report = DataQualityChecker(
        spark.createDataFrame([(1, 1), (2, 3)], ["left", "right"]),
        config,
        spark_session=spark,
    ).run(include_llm=False)
    assert (
        next(
            item for item in report.results if item.name == "quality_contract"
        )
        .typed_payload()
        .violation_count
        == expected
    )


def test_spark_contract_quotes_dots_and_backticks_in_reference_keys(
    spark: SparkSession,
) -> None:
    config = QualipilotConfig(
        engine="spark",
        checks=CheckConfig(
            missing_values=False,
            duplicates=False,
            data_types=False,
            outliers=False,
            ranges=False,
            cardinality=False,
            rule_packs=[
                QualityContract(
                    name="quoted",
                    comparisons=[
                        ComparisonRule(
                            name="comparison",
                            left_column="x.y",
                            operator="eq",
                            right_column="x`z",
                        )
                    ],
                    foreign_keys=[
                        ForeignKeyRule(
                            name="foreign",
                            columns=["x.y", "x`z"],
                            reference="reference",
                            reference_columns=["r.x", "r`z"],
                        )
                    ],
                )
            ],
        ),
    )
    report = DataQualityChecker(
        spark.createDataFrame([(1, 1), (2, 3)], ["x.y", "x`z"]),
        config,
        spark_session=spark,
        references={
            "reference": spark.createDataFrame([(1, 1)], ["r.x", "r`z"])
        },
    ).run(include_llm=False)
    assert (
        next(
            item for item in report.results if item.name == "quality_contract"
        )
        .typed_payload()
        .violation_count
        == 2
    )


def test_spark_nested_jsonl_is_opt_in_and_records_provenance(
    spark: SparkSession, tmp_path: Path
) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"id":1,"nested":{"b":1,"a":2},"items":[2,1],"plain":"value"}',
                '{"id":2,"nested":null,"items":null,"plain":"other"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="only scalar"):
        SparkEngine.from_any(path, spark_session=spark)
    engine = SparkEngine.from_any(path, spark_session=spark, allow_nested=True)
    assert engine.row_count() == 2
    assert engine.nested_normalized_columns == ["items", "nested"]
    assert [row.asDict() for row in engine._df.orderBy("id").collect()] == [
        {
            "id": 1,
            "items": "[2,1]",
            "nested": '{"a":2,"b":1}',
            "plain": "value",
        },
        {"id": 2, "items": None, "nested": None, "plain": "other"},
    ]
    report = DataQualityChecker(
        path,
        QualipilotConfig(
            engine="spark", checks=CheckConfig(allow_nested=True)
        ),
        spark_session=spark,
    ).run(include_llm=False)
    assert report.dataset.nested_normalized_columns == ["items", "nested"]


@pytest.mark.parametrize("reference_has_nan", [False, True])
@pytest.mark.parametrize(("nulls_pass", "expected"), [(True, 1), (False, 2)])
def test_spark_foreign_key_treats_nan_as_missing_per_policy(
    spark: SparkSession,
    reference_has_nan: bool,
    nulls_pass: bool,
    expected: int,
) -> None:
    reference_rows = [(1.0,)]
    if reference_has_nan:
        reference_rows.append((float("nan"),))
    config = QualipilotConfig(
        engine="spark",
        checks=CheckConfig(
            missing_values=False,
            duplicates=False,
            data_types=False,
            outliers=False,
            ranges=False,
            cardinality=False,
            rule_packs=[
                QualityContract(
                    name="keys",
                    foreign_keys=[
                        ForeignKeyRule(
                            name="key",
                            columns=["key"],
                            reference="reference",
                            reference_columns=["id"],
                            nulls_pass=nulls_pass,
                        )
                    ],
                )
            ],
        ),
    )
    report = DataQualityChecker(
        spark.createDataFrame([(1.0,), (float("nan"),), (2.0,)], "key double"),
        config,
        spark_session=spark,
        references={
            "reference": spark.createDataFrame(reference_rows, "id double")
        },
    ).run(include_llm=False)
    assert (
        next(
            item for item in report.results if item.name == "quality_contract"
        )
        .typed_payload()
        .violation_count
        == expected
    )
