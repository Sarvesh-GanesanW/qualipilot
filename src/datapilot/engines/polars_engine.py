"""Polars engine — the default backend.

We lean on LazyFrame so that multiple checks can be folded into one
query plan when the caller explicitly calls ``collect_all``. For the
public ``Engine`` API we materialise eagerly because each check asks a
different question; a future optimisation could batch them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from datapilot.engines.base import Engine


class PolarsEngine(Engine):
    """``Engine`` backed by a ``polars.DataFrame``."""

    name = "polars"

    def __init__(self, df: pl.DataFrame) -> None:
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
            return cls(pl.from_pandas(data))
        if isinstance(data, (str, Path)):
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
        return [
            c
            for c, dt in self._df.schema.items()
            if dt.is_numeric()
        ]

    def datetime_columns(self) -> list[str]:
        return [
            c
            for c, dt in self._df.schema.items()
            if dt in (pl.Datetime, pl.Date)
        ]

    # ---- per-column stats ---------------------------------------------

    def null_counts(self) -> dict[str, int]:
        # single pass, returns a 1-row frame of counts per column
        row = self._df.null_count().row(0)
        return dict(zip(self._df.columns, map(int, row), strict=False))

    def distinct_count(self, column: str) -> int:
        return int(self._df.select(pl.col(column).n_unique()).item())

    def top_values(
        self,
        column: str,
        n: int = 10,
    ) -> list[tuple[str, int]]:
        counts = (
            self._df.select(pl.col(column).value_counts(sort=True))
            .unnest(column)
            .head(n)
        )
        # second column holds counts regardless of naming differences
        value_col = counts.columns[0]
        count_col = counts.columns[1]
        out: list[tuple[str, int]] = []
        for row in counts.iter_rows():
            value, count = row
            out.append((str(value), int(count)))
        # silence unused assignment warnings when columns come back
        # with different names across polars versions
        _ = (value_col, count_col)
        return out

    def quantiles(
        self,
        columns: list[str],
        qs: tuple[float, ...] = (0.25, 0.75),
    ) -> dict[str, dict[float, float]]:
        if not columns or not qs:
            return {}
        # one aggregation expression per (col, q), folded into one pass
        exprs = [
            pl.col(c).quantile(q).alias(f"{c}__q{int(q * 1000)}")
            for c in columns
            for q in qs
        ]
        row = self._df.select(exprs).row(0)
        out: dict[str, dict[float, float]] = {c: {} for c in columns}
        idx = 0
        for c in columns:
            for q in qs:
                val = row[idx]
                out[c][q] = float(val) if val is not None else float("nan")
                idx += 1
        return out

    def describe(self) -> dict[str, dict[str, float]]:
        numeric = self.numeric_columns()
        if not numeric:
            return {}
        desc = self._df.select(numeric).describe()
        # polars returns a dataframe whose first column is the stat name
        stat_col = desc.columns[0]
        out: dict[str, dict[str, float]] = {c: {} for c in numeric}
        for row in desc.iter_rows(named=True):
            stat = str(row[stat_col])
            for c in numeric:
                val = row[c]
                if val is None:
                    continue
                try:
                    out[c][stat] = float(val)
                except (TypeError, ValueError):
                    # skip non-numeric stats like min on string columns
                    continue
        return out

    # ---- filters ------------------------------------------------------

    def duplicate_count(self, subset: list[str] | None = None) -> int:
        mask = self._df.is_duplicated()
        if subset:
            mask = self._df.select(subset).is_duplicated()
        return int(mask.sum())

    def sample_duplicates(
        self,
        n: int,
        subset: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        mask = (
            self._df.select(subset).is_duplicated()
            if subset
            else self._df.is_duplicated()
        )
        dup_frame = self._df.filter(mask).head(n)
        return dup_frame.to_dicts()

    def count_outside(
        self,
        column: str,
        low: float,
        high: float,
    ) -> int:
        return int(
            self._df.filter(
                (pl.col(column) < low) | (pl.col(column) > high)
            ).height
        )

    def sample_outside(
        self,
        column: str,
        low: float,
        high: float,
        n: int,
    ) -> list[dict[str, Any]]:
        return (
            self._df.filter(
                (pl.col(column) < low) | (pl.col(column) > high)
            )
            .head(n)
            .to_dicts()
        )

    def max_datetime(self, column: str) -> Any:
        val = self._df.select(pl.col(column).max()).item()
        return val


def _read_path(path: Path) -> pl.DataFrame:
    """Dispatch file readers based on extension."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pl.read_parquet(path)
    if suffix == ".json":
        return pl.read_json(path)
    if suffix in {".ndjson", ".jsonl"}:
        return pl.read_ndjson(path)
    raise ValueError(f"unsupported file type: {suffix}")
