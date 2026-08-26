"""Default Polars dataframe engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from qualipilot.engines._file_formats import (
    require_unique_csv_columns,
    require_valid_json_lines,
)
from qualipilot.engines.base import (
    Engine,
    reject_nested_columns,
    validate_column_names,
    validate_pandas_columns,
)


class PolarsEngine(Engine):
    """``Engine`` backed by a ``polars.DataFrame``."""

    name = "polars"

    def __init__(self, df: pl.DataFrame) -> None:
        validate_column_names(df.columns)
        reject_nested_columns(
            [
                column
                for column, dtype in df.schema.items()
                if dtype.is_nested() or dtype == pl.Object
            ]
        )
        self._df = df

    # ---- constructors --------------------------------------------------

    @classmethod
    def from_any(cls, data: Any) -> PolarsEngine:
        """Build from a Polars/Pandas dataframe or a filesystem path."""
        if isinstance(data, pl.DataFrame):
            return cls(data)
        if isinstance(data, pl.LazyFrame):
            return cls(data.collect())
        # convert pandas lazily so we do not import when unused
        if type(data).__module__.startswith("pandas"):
            validate_pandas_columns(data)
            return cls(pl.from_pandas(data))
        if type(data).__module__.startswith("pyarrow"):
            frame = pl.from_arrow(data)
            if isinstance(frame, pl.Series):
                raise TypeError(
                    f"cannot build PolarsEngine from {type(data).__name__}"
                )
            return cls(frame)
        if isinstance(data, str | Path):
            return cls(_read_path(Path(data)))
        raise TypeError(
            f"cannot build PolarsEngine from {type(data).__name__}"
        )

    # ---- structural info ----------------------------------------------

    def row_count(self) -> int:
        return self._df.height

    def columns(self) -> list[str]:
        return list(self._df.columns)

    def dtypes(self) -> dict[str, str]:
        return {c: str(dt) for c, dt in self._df.schema.items()}

    def numeric_columns(self) -> list[str]:
        return [c for c, dt in self._df.schema.items() if dt.is_numeric()]

    def datetime_columns(self) -> list[str]:
        return [
            c
            for c, dt in self._df.schema.items()
            if dt in (pl.Datetime, pl.Date)
        ]

    # ---- per-column stats ---------------------------------------------

    def null_counts(self) -> dict[str, int]:
        # single pass, returns a 1-row frame of counts per column
        exprs: list[pl.Expr] = []
        for column, dtype in self._df.schema.items():
            missing = pl.col(column).is_null()
            if dtype.is_float():
                missing = missing | pl.col(column).is_nan()
            exprs.append(missing.sum().alias(column))
        if not exprs:
            return {}
        row = self._df.select(exprs).row(0)
        return dict(zip(self._df.columns, map(int, row), strict=False))

    def distinct_count(self, column: str) -> int:
        return self.distinct_counts([column])[column]

    def distinct_counts(self, columns: list[str]) -> dict[str, int]:
        if not columns:
            return {}
        expressions = []
        for column in columns:
            value = pl.col(column)
            valid = value.is_not_null()
            if self._df.schema[column].is_float():
                valid &= value.is_not_nan()
            expressions.append(value.filter(valid).n_unique().alias(column))
        row = self._df.select(expressions).row(0)
        return {
            column: int(value)
            for column, value in zip(columns, row, strict=True)
        }

    def top_values(
        self,
        column: str,
        n: int = 10,
    ) -> list[tuple[str, int]]:
        valid = pl.col(column).is_not_null()
        if self._df.schema[column].is_float():
            valid = valid & pl.col(column).is_not_nan()
        counts = self._df.filter(valid).group_by(column).len(name="count")
        values = [
            (str(row[column]), int(row["count"]))
            for row in counts.iter_rows(named=True)
        ]
        return sorted(values, key=lambda item: (-item[1], item[0]))[:n]

    def quantiles(
        self,
        columns: list[str],
        qs: tuple[float, ...] = (0.25, 0.75),
    ) -> dict[str, dict[float, float]]:
        if not columns or not qs:
            return {}
        # one aggregation expression per (col, q), folded into one pass
        exprs: list[pl.Expr] = []
        for column in columns:
            values = pl.col(column)
            if self._df.schema[column].is_float():
                values = values.filter(values.is_not_nan())
            exprs.extend(
                values.quantile(q, interpolation="linear").alias(
                    f"{column}__q{index}"
                )
                for index, q in enumerate(qs)
            )
        row = self._df.select(exprs).row(0)
        out: dict[str, dict[float, float]] = {c: {} for c in columns}
        idx = 0
        for c in columns:
            for q in qs:
                val = row[idx]
                out[c][q] = float(val) if val is not None else float("nan")
                idx += 1
        return out

    # ---- filters ------------------------------------------------------

    def duplicate_count(self, subset: list[str] | None = None) -> int:
        return int(self._duplicate_mask(subset).sum())

    def sample_duplicates(
        self,
        n: int,
        subset: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        dup_frame = self._df.filter(self._duplicate_mask(subset)).head(n)
        return dup_frame.to_dicts()

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
        row = self._df.select(
            self._outside_mask(column, *ranges[column]).sum().alias(column)
            for column in columns
        ).row(0)
        return {
            column: int(value)
            for column, value in zip(columns, row, strict=True)
        }

    def sample_outside(
        self,
        column: str,
        low: float,
        high: float,
        n: int,
    ) -> list[dict[str, Any]]:
        return (
            self._df.filter(self._outside_mask(column, low, high))
            .head(n)
            .to_dicts()
        )

    def max_datetime(self, column: str) -> Any:
        return self._df.select(pl.col(column).max()).item()

    def max_datetime_instant(
        self,
        column: str,
        naive_timezone: str = "UTC",
    ) -> Any:
        if self._df.schema[column] != pl.String:
            return super().max_datetime_instant(column, naive_timezone)

        text = pl.col(column).str.strip_chars()
        has_offset = text.str.contains(r"(?:[zZ]|[+-]\d{2}:?\d{2})$")
        aware = (
            pl.when(has_offset)
            .then(text)
            .otherwise(None)
            .str.to_datetime(time_zone="UTC", strict=False)
        )
        naive = (
            pl.when(~has_offset)
            .then(text)
            .otherwise(None)
            .str.to_datetime(strict=False)
            .dt.replace_time_zone(naive_timezone)
            .dt.convert_time_zone("UTC")
        )
        parsed = pl.coalesce(aware, naive)
        max_timestamp, invalid_count = self._df.select(
            parsed.max().alias("max_timestamp"),
            (pl.col(column).is_not_null() & parsed.is_null())
            .sum()
            .alias("invalid_count"),
        ).row(0)
        if invalid_count:
            raise ValueError(
                f"freshness column {column!r} contains invalid timestamps"
            )
        return max_timestamp

    def _outside_mask(self, column: str, low: float, high: float) -> pl.Expr:
        mask = (pl.col(column) < low) | (pl.col(column) > high)
        if self._df.schema[column].is_float():
            mask &= pl.col(column).is_not_nan()
        return mask

    def _duplicate_mask(self, subset: list[str] | None) -> pl.Series:
        columns = subset or self.columns()
        keys = self._df.select(
            pl.col(column).fill_nan(None)
            if self._df.schema[column].is_float()
            else pl.col(column)
            for column in columns
        )
        return keys.is_duplicated()


def _read_path(path: Path) -> pl.DataFrame:
    """Dispatch file readers based on extension."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        require_unique_csv_columns(path)
        return pl.read_csv(path, null_values="")
    if suffix in {".parquet", ".pq"}:
        return pl.read_parquet(path)
    if suffix in {".ndjson", ".jsonl"}:
        scalar_types = require_valid_json_lines(path)
        dtypes = {
            "string": pl.String,
            "integer": pl.Int64,
            "number": pl.Float64,
            "boolean": pl.Boolean,
        }
        return pl.read_ndjson(
            path,
            schema={
                column: dtypes[family]
                for column, family in scalar_types.items()
            },
            infer_schema_length=None,
        )
    raise ValueError(f"unsupported file type: {suffix}")
