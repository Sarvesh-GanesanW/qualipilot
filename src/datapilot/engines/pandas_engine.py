"""Pandas engine — kept for ecosystem compatibility.

Pandas is still the lingua franca for many downstream tools, so while
Polars is the default we keep a first-class adapter. Performance is
comparable up to ~1M rows, past that prefer Polars or Dask.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from datapilot.engines.base import Engine


class PandasEngine(Engine):
    """``Engine`` backed by a ``pandas.DataFrame``."""

    name = "pandas"

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    @classmethod
    def from_any(cls, data: Any) -> PandasEngine:
        if isinstance(data, pd.DataFrame):
            return cls(data)
        if type(data).__module__.startswith("polars"):
            return cls(data.to_pandas())
        if isinstance(data, (str, Path)):
            return cls(_read_path(Path(data)))
        raise TypeError(
            f"cannot build PandasEngine from {type(data).__name__}"
        )

    def row_count(self) -> int:
        return int(len(self._df))

    def columns(self) -> list[str]:
        return list(self._df.columns)

    def dtypes(self) -> dict[str, str]:
        return {c: str(dt) for c, dt in self._df.dtypes.items()}

    def numeric_columns(self) -> list[str]:
        return list(
            self._df.select_dtypes(include="number").columns
        )

    def datetime_columns(self) -> list[str]:
        return list(
            self._df.select_dtypes(
                include=["datetime", "datetimetz"]
            ).columns
        )

    def null_counts(self) -> dict[str, int]:
        return {
            c: int(v) for c, v in self._df.isna().sum().items()
        }

    def distinct_count(self, column: str) -> int:
        return int(self._df[column].nunique(dropna=True))

    def top_values(
        self,
        column: str,
        n: int = 10,
    ) -> list[tuple[str, int]]:
        counts = self._df[column].value_counts(dropna=True).head(n)
        return [(str(idx), int(cnt)) for idx, cnt in counts.items()]

    def quantiles(
        self,
        columns: list[str],
        qs: tuple[float, ...] = (0.25, 0.75),
    ) -> dict[str, dict[float, float]]:
        if not columns or not qs:
            return {}
        # pandas quantile accepts a list and returns a dataframe indexed
        # by q, which is exactly what we need
        q_df = self._df[columns].quantile(list(qs))
        return {
            c: {float(q): float(q_df.at[q, c]) for q in qs}
            for c in columns
        }

    def describe(self) -> dict[str, dict[str, float]]:
        numeric = self.numeric_columns()
        if not numeric:
            return {}
        desc = self._df[numeric].describe()
        return {
            c: {str(k): float(v) for k, v in desc[c].items()}
            for c in numeric
        }

    def duplicate_count(self, subset: list[str] | None = None) -> int:
        return int(self._df.duplicated(subset=subset, keep=False).sum())

    def sample_duplicates(
        self,
        n: int,
        subset: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        mask = self._df.duplicated(subset=subset, keep=False)
        return self._df[mask].head(n).to_dict(orient="records")

    def count_outside(
        self,
        column: str,
        low: float,
        high: float,
    ) -> int:
        series = self._df[column]
        return int(((series < low) | (series > high)).sum())

    def sample_outside(
        self,
        column: str,
        low: float,
        high: float,
        n: int,
    ) -> list[dict[str, Any]]:
        series = self._df[column]
        mask = (series < low) | (series > high)
        return self._df[mask].head(n).to_dict(orient="records")

    def max_datetime(self, column: str) -> Any:
        val = self._df[column].max()
        # pandas returns NaT for empty series, normalise to None
        return None if pd.isna(val) else val


def _read_path(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in {".ndjson", ".jsonl"}:
        return pd.read_json(path, lines=True)
    raise ValueError(f"unsupported file type: {suffix}")
