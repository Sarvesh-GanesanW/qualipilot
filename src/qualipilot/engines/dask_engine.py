"""Dask engine for partitioned dataframe workloads."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import dask
import pandas as pd

from qualipilot.engines._file_formats import (
    require_safe_remote_url,
    require_unique_csv_columns,
    require_valid_json_lines,
)
from qualipilot.engines.base import (
    Engine,
    as_utc_datetime,
    is_date_arrow_dtype,
    is_decimal_arrow_dtype,
    is_nested_arrow_dtype,
    is_unsupported_pandas_dtype,
    object_dtype_family,
    reject_nested_columns,
    validate_column_names,
    validate_pandas_columns,
)

try:
    import dask.dataframe as dd
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "dask is required for DaskEngine; "
        "install with `pip install qualipilot[dask]`"
    ) from exc


class DaskEngine(Engine):
    """``Engine`` backed by a ``dask.dataframe.DataFrame``."""

    name = "dask"

    def __init__(self, df: dd.DataFrame) -> None:
        validate_column_names(list(df.columns))
        unsupported_dtypes = [
            f"{column} ({dtype})"
            for column, dtype in df.dtypes.items()
            if is_unsupported_pandas_dtype(dtype)
        ]
        if unsupported_dtypes:
            raise TypeError(
                "unsupported pandas column types: "
                + ", ".join(unsupported_dtypes)
                + "; cast them to a portable string, numeric, date, or "
                "binary dtype"
            )
        (
            nested_object_columns,
            self._object_families,
            unsupported_object_columns,
        ) = _inspect_object_columns(df)
        reject_nested_columns(
            [
                *[
                    column
                    for column, dtype in df.dtypes.items()
                    if is_nested_arrow_dtype(dtype)
                ],
                *nested_object_columns,
            ]
        )
        if unsupported_object_columns:
            raise TypeError(
                "unsupported pandas column types: "
                + ", ".join(unsupported_object_columns)
                + "; cast them to a portable string, numeric, date, or "
                "binary dtype"
            )
        self._df = df

    @classmethod
    def from_any(cls, data: Any, *, npartitions: int = 4) -> DaskEngine:
        if isinstance(data, dd.DataFrame):
            return cls(data)
        if isinstance(data, pd.DataFrame):
            validate_pandas_columns(data)
            with dask.config.set({"dataframe.convert-string": False}):
                return cls(
                    dd.from_pandas(  # type: ignore[no-untyped-call]
                        data,
                        npartitions=npartitions,
                    )
                )
        if type(data).__module__.startswith("polars"):
            with dask.config.set({"dataframe.convert-string": False}):
                return cls(
                    dd.from_pandas(  # type: ignore[no-untyped-call]
                        data.to_pandas(),
                        npartitions=npartitions,
                    )
                )
        if type(data).__module__.startswith("pyarrow"):
            return cls.from_any(
                data.to_pandas(),
                npartitions=npartitions,
            )
        if isinstance(data, str | Path):
            raw = str(data)
            remote = "://" in raw
            if remote:
                require_safe_remote_url(raw)
            suffix = Path(urlparse(raw).path if remote else raw).suffix.lower()
            if suffix == ".csv":
                require_unique_csv_columns(raw)
                return cls(
                    dd.read_csv(
                        raw,
                        keep_default_na=False,
                        na_values=[""],
                        skip_blank_lines=False,
                    )
                )
            if suffix in {".parquet", ".pq"}:
                return cls(dd.read_parquet(raw))
            if suffix in {".jsonl", ".ndjson"}:
                scalar_types = require_valid_json_lines(raw)
                dtypes = {
                    "string": "string",
                    "integer": "Int64",
                    "number": "Float64",
                    "boolean": "boolean",
                }
                with dask.config.set({"dataframe.convert-string": False}):
                    return cls(
                        dd.read_json(
                            raw,
                            lines=True,
                            blocksize=64 * 1024 * 1024,
                            dtype={
                                column: dtypes[family]
                                for column, family in scalar_types.items()
                            },
                            convert_dates=False,
                        )
                    )
            raise ValueError(f"unsupported file type: {suffix}")
        raise TypeError(f"cannot build DaskEngine from {type(data).__name__}")

    def row_count(self) -> int:
        return int(self._df.shape[0].compute())

    def columns(self) -> list[str]:
        return list(self._df.columns)

    def dtypes(self) -> dict[str, str]:
        return {c: str(dt) for c, dt in self._df.dtypes.items()}

    def dtype_family(self, column: str) -> str:
        families = self._object_families.get(column)
        if families is not None:
            return object_dtype_family(families)
        return super().dtype_family(column)

    def numeric_columns(self) -> list[str]:
        return [
            column
            for column in self._df.columns
            if self.dtype_family(column) in {"integer", "float", "decimal"}
        ]

    def datetime_columns(self) -> list[str]:
        return [
            column
            for column in self._df.columns
            if self.dtype_family(column) in {"date", "datetime"}
        ]

    def null_counts(self) -> dict[str, int]:
        result = self._df.isna().sum().compute()
        return {c: int(v) for c, v in result.items()}

    def distinct_count(self, column: str) -> int:
        return self.distinct_counts([column])[column]

    def distinct_counts(self, columns: list[str]) -> dict[str, int]:
        if not columns:
            return {}
        values = dd.compute(  # type: ignore[no-untyped-call]
            *(self._df[column].nunique() for column in columns)
        )
        return {
            column: int(value)
            for column, value in zip(columns, values, strict=True)
        }

    def top_values(
        self,
        column: str,
        n: int = 10,
    ) -> list[tuple[str, int]]:
        count_name = "__qualipilot_count__"
        text_name = "__qualipilot_text__"
        while count_name == column:
            count_name += "_"
        while text_name in {column, count_name}:
            text_name += "_"
        counts = (
            self._df[column]
            .value_counts(dropna=True)
            .rename(count_name)
            .to_frame()
            .reset_index()
        )
        ranked = counts.assign(**{text_name: counts[column].astype(str)})
        rows = ranked.sort_values(
            [count_name, text_name],
            ascending=[False, True],
        ).head(n, npartitions=-1)
        return [
            (str(row[column]), int(row[count_name]))
            for _, row in rows.iterrows()
        ]

    def quantiles(
        self,
        columns: list[str],
        qs: tuple[float, ...] = (0.25, 0.75),
    ) -> dict[str, dict[float, float]]:
        if not columns or not qs:
            return {}
        if any(not math.isfinite(q) or not 0 <= q <= 1 for q in qs):
            raise ValueError("quantiles must be finite and between 0 and 1")
        # Dask quantiles are approximate; collecting for exact quantiles
        # would defeat this engine's larger-than-memory contract.
        interior = sorted({q for q in qs if 0 < q < 1})
        tasks: list[tuple[str, str, Any]] = []
        for column in columns:
            series = self._df[column]
            if is_decimal_arrow_dtype(
                series.dtype
            ) or pd.api.types.is_object_dtype(series.dtype):
                series = series.astype(float)
            if 0 in qs:
                tasks.append((column, "min", series.min()))
            if interior:
                tasks.append((column, "quantiles", series.quantile(interior)))
            if 1 in qs:
                tasks.append((column, "max", series.max()))
        computed = dd.compute(  # type: ignore[no-untyped-call]
            *(task for _, _, task in tasks)
        )
        out: dict[str, dict[float, float]] = {column: {} for column in columns}
        for (column, kind, _), value in zip(tasks, computed, strict=True):
            if kind == "min":
                out[column][0.0] = float(value)
            elif kind == "max":
                out[column][1.0] = float(value)
            else:
                out[column].update(
                    {float(q): float(value.loc[q]) for q in interior}
                )
        return out

    @property
    def quantile_provenance(self) -> dict[str, str | float]:
        return {"method": "approximate"}

    def duplicate_count(self, subset: list[str] | None = None) -> int:
        counts, count_column = self._duplicate_counts(subset)
        duplicate_rows = counts[counts[count_column] > 1][count_column].sum()
        result = duplicate_rows.compute()
        return 0 if pd.isna(result) else int(result)

    def sample_duplicates(
        self,
        n: int,
        subset: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        columns = subset or self.columns()
        counts, count_column = self._duplicate_counts(columns)
        duplicate_keys = counts[counts[count_column] > 1][columns].astype(
            {column: self._df.dtypes[column] for column in columns}
        )
        rows = self._df.merge(  # type: ignore[no-untyped-call]
            duplicate_keys,
            on=columns,
            how="inner",
        )
        return cast(
            list[dict[str, Any]],
            rows.head(n, npartitions=-1, compute=True).to_dict(
                orient="records"
            ),
        )

    def count_outside(
        self,
        column: str,
        low: float,
        high: float,
    ) -> int:
        return self.counts_outside({column: (low, high)})[column]

    def counts_outside(
        self,
        ranges: dict[str, tuple[float, float]],
    ) -> dict[str, int]:
        if not ranges:
            return {}
        columns = list(ranges)
        values = dd.compute(  # type: ignore[no-untyped-call]
            *(
                (
                    (self._df[column] < ranges[column][0])
                    | (self._df[column] > ranges[column][1])
                ).sum()
                for column in columns
            )
        )
        return {
            column: int(value)
            for column, value in zip(columns, values, strict=True)
        }

    def sample_outside(
        self,
        column: str,
        low: float,
        high: float,
        n: int,
    ) -> list[dict[str, Any]]:
        series = self._df[column]
        mask = (series < low) | (series > high)
        return cast(
            list[dict[str, Any]],
            self._df[mask]
            .head(n, npartitions=-1, compute=True)
            .to_dict(orient="records"),
        )

    def max_datetime(self, column: str) -> Any:
        series = self._df[column]
        if is_date_arrow_dtype(series.dtype):
            series = series.astype("datetime64[ns]")
        val = series.max().compute()
        return None if pd.isna(val) else val

    def max_datetime_instant(
        self,
        column: str,
        naive_timezone: str = "UTC",
    ) -> Any:
        series = self._df[column]
        if not (
            pd.api.types.is_object_dtype(series.dtype)
            or isinstance(series.dtype, pd.StringDtype)
        ):
            return super().max_datetime_instant(column, naive_timezone)
        maxima = series.map_partitions(
            _datetime_partition_max,
            naive_timezone,
            meta=pd.Series(dtype="datetime64[ns, UTC]"),
            clear_divisions=True,
        )
        value = maxima.max().compute()
        return None if pd.isna(value) else as_utc_datetime(value, "UTC")

    def _duplicate_counts(
        self,
        subset: list[str] | None,
    ) -> tuple[dd.DataFrame, str]:
        columns = subset or self.columns()
        count_column = "__qualipilot_count__"
        while count_column in self.columns():
            count_column += "_"
        counts = (
            self._df.groupby(columns, dropna=False)
            .size()
            .rename(count_column)
            .reset_index()
        )
        return counts, count_column


def _inspect_object_columns(
    frame: dd.DataFrame,
) -> tuple[list[str], dict[str, set[str]], list[str]]:
    object_columns = [
        column
        for column, dtype in frame.dtypes.items()
        if pd.api.types.is_object_dtype(dtype)
    ]
    if not object_columns:
        return [], {}, []
    results = dd.compute(  # type: ignore[no-untyped-call]
        *(_object_stats(frame[column]) for column in object_columns)
    )
    nested: list[str] = []
    object_families: dict[str, set[str]] = {}
    unsupported: list[str] = []
    for column, stats in zip(object_columns, results, strict=True):
        if stats["nested"].any():
            nested.append(column)
            continue
        families = {
            family
            for value in stats["families"]
            for family in str(value).split("|")
            if family
        }
        object_families[column] = families
        if not _portable_object_families(families):
            family = "+".join(sorted(families)) or "empty"
            unsupported.append(f"{column} (object/{family})")
    return nested, object_families, unsupported


def _object_stats(series: Any) -> Any:
    meta = pd.DataFrame(
        {
            "nested": pd.Series(dtype=bool),
            "families": pd.Series(dtype=object),
        }
    )

    def inspect_partition(partition: pd.Series) -> pd.DataFrame:
        values = partition.dropna()
        scalar = values.map(pd.api.types.is_scalar)
        families = {
            pd.api.types.infer_dtype([value], skipna=True)
            for value in values[scalar]
        }
        return pd.DataFrame(
            {
                "nested": [not bool(scalar.all())],
                "families": ["|".join(sorted(families))],
            }
        )

    return series.map_partitions(inspect_partition, meta=meta)


def _portable_object_families(families: set[str]) -> bool:
    if not families:
        return True
    if len(families) == 1:
        return families <= {
            "boolean",
            "bytes",
            "date",
            "datetime",
            "datetime64",
            "decimal",
            "floating",
            "integer",
            "string",
            "time",
            "timedelta",
            "timedelta64",
        }
    return families in (
        {"integer", "floating"},
        {"date", "datetime"},
    )


def _datetime_partition_max(
    series: pd.Series,
    naive_timezone: str,
) -> pd.Series:
    values = (
        as_utc_datetime(value, naive_timezone)
        for value in series
        if not pd.isna(value)
    )
    return pd.Series(
        [max(values, default=None)],
        dtype="datetime64[ns, UTC]",
    )
