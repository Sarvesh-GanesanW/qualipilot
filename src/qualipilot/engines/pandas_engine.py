"""Pandas engine for ecosystem compatibility."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pandas as pd

from qualipilot.engines._file_formats import (
    require_unique_csv_columns,
    require_valid_json_lines,
)
from qualipilot.engines.base import (
    Engine,
    is_decimal_arrow_dtype,
    validate_pandas_columns,
)


class PandasEngine(Engine):
    """``Engine`` backed by a ``pandas.DataFrame``."""

    name = "pandas"

    def __init__(self, df: pd.DataFrame) -> None:
        validate_pandas_columns(df)
        self._df = df

    @classmethod
    def from_any(cls, data: Any) -> PandasEngine:
        if isinstance(data, pd.DataFrame):
            return cls(data)
        if type(data).__module__.startswith("polars"):
            return cls(data.to_pandas())
        if type(data).__module__.startswith("pyarrow"):
            return cls(data.to_pandas())
        if isinstance(data, str | Path):
            return cls(_read_path(Path(data)))
        raise TypeError(
            f"cannot build PandasEngine from {type(data).__name__}"
        )

    def row_count(self) -> int:
        return len(self._df)

    def columns(self) -> list[str]:
        return list(self._df.columns)

    def dtypes(self) -> dict[str, str]:
        return {c: str(dt) for c, dt in self._df.dtypes.items()}

    def numeric_columns(self) -> list[str]:
        columns = [
            column
            for column in self._df.select_dtypes(include="number").columns
            if not pd.api.types.is_timedelta64_dtype(self._df[column].dtype)
        ]
        columns.extend(
            column
            for column in self._df.columns
            if column not in columns and _is_decimal_series(self._df[column])
        )
        return columns

    def datetime_columns(self) -> list[str]:
        columns = list(
            self._df.select_dtypes(include=["datetime", "datetimetz"]).columns
        )
        columns.extend(
            column
            for column in self._df.columns
            if column not in columns and _is_date_series(self._df[column])
        )
        return columns

    def null_counts(self) -> dict[str, int]:
        return {c: int(v) for c, v in self._df.isna().sum().items()}

    def distinct_count(self, column: str) -> int:
        return self.distinct_counts([column])[column]

    def distinct_counts(self, columns: list[str]) -> dict[str, int]:
        if not columns:
            return {}
        counts = self._df[columns].nunique(dropna=True)
        return {column: int(counts[column]) for column in columns}

    def top_values(
        self,
        column: str,
        n: int = 10,
    ) -> list[tuple[str, int]]:
        counts = self._df[column].value_counts(dropna=True)
        values = [(str(idx), int(count)) for idx, count in counts.items()]
        return sorted(values, key=lambda item: (-item[1], item[0]))[:n]

    def quantiles(
        self,
        columns: list[str],
        qs: tuple[float, ...] = (0.25, 0.75),
    ) -> dict[str, dict[float, float]]:
        if not columns or not qs:
            return {}
        # pandas quantile accepts a list and returns a dataframe indexed
        # by q, which is exactly what we need
        frame = self._df[columns].copy()
        for column in columns:
            if _is_decimal_series(frame[column]):
                frame[column] = frame[column].map(
                    lambda value: (
                        float(value) if isinstance(value, Decimal) else value
                    )
                )
        q_df = frame.quantile(list(qs), interpolation="linear")
        return {
            c: {float(q): float(q_df.at[q, c]) for q in qs} for c in columns
        }

    def duplicate_count(self, subset: list[str] | None = None) -> int:
        return int(self._duplicate_mask(subset).sum())

    def sample_duplicates(
        self,
        n: int,
        subset: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        mask = self._duplicate_mask(subset)
        return cast(
            list[dict[str, Any]],
            self._df[mask].head(n).to_dict(orient="records"),
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
        return {
            column: int(
                ((self._df[column] < low) | (self._df[column] > high)).sum()
            )
            for column, (low, high) in ranges.items()
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
            self._df[mask].head(n).to_dict(orient="records"),
        )

    def max_datetime(self, column: str) -> Any:
        val = self._df[column].max()
        # pandas returns NaT for empty series, normalise to None
        return None if pd.isna(val) else val

    def _duplicate_mask(self, subset: list[str] | None) -> pd.Series:
        columns = subset or self.columns()
        keys = self._df[columns].copy()
        for column in columns:
            if pd.api.types.is_object_dtype(keys[column].dtype):
                keys[column] = keys[column].where(
                    keys[column].notna(),
                    None,
                )
        return keys.duplicated(keep=False)


def _read_path(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        require_unique_csv_columns(path)
        return pd.read_csv(
            path,
            keep_default_na=False,
            na_values=[""],
            skip_blank_lines=False,
        )
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".ndjson", ".jsonl"}:
        scalar_types = require_valid_json_lines(path)
        dtypes = {
            "string": "string",
            "integer": "Int64",
            "number": "Float64",
            "boolean": "boolean",
        }
        return pd.read_json(
            path,
            lines=True,
            dtype={
                column: dtypes[family]
                for column, family in scalar_types.items()
            },
            convert_dates=False,
        )
    raise ValueError(f"unsupported file type: {suffix}")


def _is_decimal_series(series: pd.Series) -> bool:
    if is_decimal_arrow_dtype(series.dtype):
        return True
    if not pd.api.types.is_object_dtype(series.dtype):
        return False
    values = series.dropna()
    return not values.empty and bool(
        values.map(lambda value: isinstance(value, Decimal)).all()
    )


def _is_date_series(series: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(
        series.dtype
    ) and pd.api.types.infer_dtype(series, skipna=True) in {
        "date",
        "datetime",
        "datetime64",
    }
