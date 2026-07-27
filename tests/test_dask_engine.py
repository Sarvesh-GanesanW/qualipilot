"""Focused contract tests for the optional Dask engine."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import dask
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

dd = pytest.importorskip("dask.dataframe")
DaskEngine = pytest.importorskip("qualipilot.engines.dask_engine").DaskEngine


def test_duplicates_are_global_across_partitions() -> None:
    frame = pd.DataFrame(
        {
            "id": [1, 2, 3, 1],
            "value": ["duplicate", "a", "b", "duplicate"],
        }
    )
    engine = DaskEngine.from_any(frame, npartitions=4)
    assert engine.duplicate_count() == 2
    assert engine.sample_duplicates(10) == [
        {"id": 1, "value": "duplicate"},
        {"id": 1, "value": "duplicate"},
    ]


def test_duplicate_subset_is_global() -> None:
    frame = pd.DataFrame({"id": [1, 2, 3, 4], "group": ["x", "a", "b", "x"]})
    engine = DaskEngine.from_any(frame, npartitions=4)
    assert engine.duplicate_count(["group"]) == 2
    assert len(engine.sample_duplicates(10, ["group"])) == 2


def test_samples_scan_all_partitions() -> None:
    frame = pd.DataFrame({"value": [1, 1, 1, 100]})
    engine = DaskEngine.from_any(frame, npartitions=4)
    assert engine.count_outside("value", 0, 10) == 1
    assert engine.sample_outside("value", 0, 10, 10) == [{"value": 100}]


def test_top_values_breaks_ties_by_value() -> None:
    frame = pd.DataFrame({"value": ["z", "z", "c", "b", "a"]})
    engine = DaskEngine.from_any(frame, npartitions=3)

    assert engine.top_values("value", 3) == [
        ("z", 2),
        ("a", 1),
        ("b", 1),
    ]


def test_null_duplicate_keys_are_included() -> None:
    frame = pd.DataFrame({"group": [None, "x", None], "value": [1, 2, 1]})
    engine = DaskEngine.from_any(frame, npartitions=3)
    assert engine.duplicate_count() == 2
    assert len(engine.sample_duplicates(10)) == 2


def test_object_null_duplicate_samples_are_included() -> None:
    frame = pd.DataFrame(
        {"group": pd.Series([None, float("nan")], dtype=object)}
    )
    engine = DaskEngine.from_any(frame, npartitions=2)

    assert engine.duplicate_count() == 2
    assert len(engine.sample_duplicates(10)) == 2


def test_nested_columns_are_rejected_without_string_conversion(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested.jsonl"
    path.write_text(
        '{"id":1,"nested":["value"]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="only scalar values"):
        DaskEngine.from_any(path)


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
def test_arrow_nested_columns_are_rejected(
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
        DaskEngine.from_any(frame, npartitions=2)


def test_direct_dask_complex_columns_are_rejected() -> None:
    frame = dd.from_pandas(
        pd.DataFrame({"value": [1 + 2j, 2 + 3j]}),
        npartitions=2,
    )

    with pytest.raises(TypeError, match="unsupported pandas column types"):
        DaskEngine.from_any(frame)


def test_quantile_endpoints_are_monotonic() -> None:
    values = np.random.default_rng(0).normal(size=10)
    engine = DaskEngine.from_any(
        pd.DataFrame({"value": values}),
        npartitions=10,
    )

    quantiles = engine.quantiles(["value"], (0.0, 0.1, 1.0))["value"]

    assert quantiles[0.0] <= quantiles[0.1] <= quantiles[1.0]


def test_decimal_detection_uses_one_compute() -> None:
    with patch(
        "qualipilot.engines.dask_engine.dd.compute",
        wraps=dd.compute,
    ) as compute:
        engine = DaskEngine.from_any(
            pd.DataFrame(
                {
                    "first": [Decimal("1.0"), None],
                    "second": [None, Decimal("2.0")],
                    "text": ["a", "b"],
                }
            ),
            npartitions=2,
        )
        assert engine.numeric_columns() == ["first", "second"]

    compute.assert_called_once()


def test_direct_dask_input_rejects_mixed_partition_families() -> None:
    with dask.config.set({"dataframe.convert-string": False}):
        frame = dd.from_pandas(
            pd.DataFrame({"value": pd.Series([1, "one"], dtype=object)}),
            npartitions=2,
        )

    with pytest.raises(
        TypeError,
        match="unsupported pandas column types",
    ):
        DaskEngine(frame)


def test_direct_dask_input_rejects_nested_columns() -> None:
    with dask.config.set({"dataframe.convert-string": False}):
        frame = dd.from_pandas(
            pd.DataFrame({"nested": pd.Series([[1], [2]], dtype=object)}),
            npartitions=2,
        )

    with pytest.raises(TypeError, match=r"flatten.*nested"):
        DaskEngine(frame)


@pytest.mark.parametrize(
    ("suffix", "lines"),
    [(".jsonl", True), (".ndjson", True)],
)
def test_json_inputs(tmp_path: Path, suffix: str, lines: bool) -> None:
    path = tmp_path / f"records{suffix}"
    pd.DataFrame({"id": [1, 2]}).to_json(
        path,
        orient="records",
        lines=lines,
    )

    with patch(
        "qualipilot.engines.dask_engine.dd.read_json",
        wraps=dd.read_json,
    ) as read_json:
        engine = DaskEngine.from_any(path)

    assert engine.row_count() == 2
    if lines:
        assert read_json.call_args.kwargs["blocksize"] == 64 * 1024 * 1024


def test_csv_null_tokens_match(tmp_path: Path) -> None:
    path = tmp_path / "tokens.csv"
    path.write_text('value\nNA\n""\nfoo\n', encoding="utf-8")

    engine = DaskEngine.from_any(path)

    assert engine.null_counts() == {"value": 1}
    assert engine.top_values("value") == [("NA", 1), ("foo", 1)]


def test_json_array_files_are_not_supported(tmp_path: Path) -> None:
    path = tmp_path / "records.json"
    pd.DataFrame({"id": [1, 2]}).to_json(path)

    with pytest.raises(ValueError, match="unsupported file type"):
        DaskEngine.from_any(path)


def test_remote_csv_url_is_preserved() -> None:
    with (
        patch(
            "qualipilot.engines.dask_engine.require_unique_csv_columns"
        ) as validate,
        patch(
            "qualipilot.engines.dask_engine.dd.read_csv",
            return_value=dd.from_pandas(
                pd.DataFrame({"id": [1]}), npartitions=1
            ),
        ) as read_csv,
    ):
        DaskEngine.from_any("s3://bucket/data.csv")

    validate.assert_called_once_with("s3://bucket/data.csv")
    assert read_csv.call_args.args[0] == "s3://bucket/data.csv"


def test_remote_urls_must_not_embed_credentials() -> None:
    with pytest.raises(ValueError, match="configure credentials outside"):
        DaskEngine.from_any(
            "https://user:secret@example.com/data.parquet?token=private"
        )


def test_file_uri_csv_duplicate_headers_are_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "duplicates.csv"
    path.write_text("id,id\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="column names must be unique"):
        DaskEngine.from_any(path.as_uri())


def test_jsonl_duplicate_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicates.jsonl"
    path.write_text('{"id":1,"id":2}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate object keys"):
        DaskEngine.from_any(path)
