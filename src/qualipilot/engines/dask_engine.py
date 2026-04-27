"""Dask engine — used for larger-than-memory workloads.

All expensive operations are kept lazy until a ``.compute()`` boundary
so a single check does not trigger multiple full scans.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from qualipilot.engines.base import Engine

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
        self._df = df

    @classmethod
    def from_any(cls, data: Any, *, npartitions: int = 4) -> DaskEngine:
        if isinstance(data, dd.DataFrame):
            return cls(data)
        if isinstance(data, pd.DataFrame):
            return cls(dd.from_pandas(data, npartitions=npartitions))
        if type(data).__module__.startswith("polars"):
            return cls(
                dd.from_pandas(data.to_pandas(), npartitions=npartitions)
            )
        if isinstance(data, (str, Path)):
            path = Path(data)
            suffix = path.suffix.lower()
            if suffix == ".csv":
                return cls(dd.read_csv(str(path)))
            if suffix in {".parquet", ".pq"}:
                return cls(dd.read_parquet(str(path)))
            raise ValueError(f"unsupported file type: {suffix}")
        raise TypeError(f"cannot build DaskEngine from {type(data).__name__}")

    def row_count(self) -> int:
        return int(self._df.shape[0].compute())

    def columns(self) -> list[str]:
        return list(self._df.columns)

    def dtypes(self) -> dict[str, str]:
        return {c: str(dt) for c, dt in self._df.dtypes.items()}

    def numeric_columns(self) -> list[str]:
        return list(self._df.select_dtypes(include="number").columns)

    def datetime_columns(self) -> list[str]:
        return list(
            self._df.select_dtypes(include=["datetime", "datetimetz"]).columns
        )

    def null_counts(self) -> dict[str, int]:
        result = self._df.isna().sum().compute()
        return {c: int(v) for c, v in result.items()}

    def distinct_count(self, column: str) -> int:
        return int(self._df[column].nunique().compute())

    def top_values(
        self,
        column: str,
        n: int = 10,
    ) -> list[tuple[str, int]]:
        counts = self._df[column].value_counts().nlargest(n).compute()
        return [(str(idx), int(cnt)) for idx, cnt in counts.items()]

    def quantiles(
        self,
        columns: list[str],
        qs: tuple[float, ...] = (0.25, 0.75),
    ) -> dict[str, dict[float, float]]:
        if not columns or not qs:
            return {}
        # dask supports tdigest approximations for speed; we keep the
        # exact path because dataset shapes here are typically modest
        out: dict[str, dict[float, float]] = {c: {} for c in columns}
        futures = {
            (c, q): self._df[c].quantile(q) for c in columns for q in qs
        }
        # single compute call collapses the task graph for all pairs
        computed = dd.compute(*futures.values())
        for key, val in zip(futures.keys(), computed, strict=True):
            col, q = key
            out[col][float(q)] = (
                float(val) if val is not None else float("nan")
            )
        return out

    def describe(self) -> dict[str, dict[str, float]]:
        numeric = self.numeric_columns()
        if not numeric:
            return {}
        desc = self._df[numeric].describe().compute()
        return {
            c: {str(k): float(v) for k, v in desc[c].items()} for c in numeric
        }

    def duplicate_count(self, subset: list[str] | None = None) -> int:
        # global duplicate count requires a shuffle — unavoidable cost
        df = self._df if subset is None else self._df[subset]
        return int(df.duplicated(keep=False).sum().compute())

    def sample_duplicates(
        self,
        n: int,
        subset: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        base = self._df if subset is None else self._df[subset]
        mask = base.duplicated(keep=False)
        return self._df[mask].head(n, compute=True).to_dict(orient="records")

    def count_outside(
        self,
        column: str,
        low: float,
        high: float,
    ) -> int:
        series = self._df[column]
        return int(((series < low) | (series > high)).sum().compute())

    def sample_outside(
        self,
        column: str,
        low: float,
        high: float,
        n: int,
    ) -> list[dict[str, Any]]:
        series = self._df[column]
        mask = (series < low) | (series > high)
        return self._df[mask].head(n, compute=True).to_dict(orient="records")

    def max_datetime(self, column: str) -> Any:
        val = self._df[column].max().compute()
        return None if pd.isna(val) else val
