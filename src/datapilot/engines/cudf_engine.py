"""cuDF engine — GPU-accelerated path.

Only imported on demand; the base install does not depend on RAPIDS.
Most methods mirror the Pandas adapter because cuDF is a drop-in
replacement at the API level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datapilot.engines.base import Engine

try:
    import cudf  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "cudf is not installed; install RAPIDS "
        "(https://docs.rapids.ai/install) to enable the GPU engine"
    ) from exc


class CudfEngine(Engine):
    """``Engine`` backed by a ``cudf.DataFrame``."""

    name = "cudf"

    def __init__(self, df: cudf.DataFrame) -> None:
        self._df = df

    @classmethod
    def from_any(cls, data: Any) -> CudfEngine:
        if isinstance(data, cudf.DataFrame):
            return cls(data)
        if type(data).__module__.startswith("pandas"):
            return cls(cudf.DataFrame.from_pandas(data))
        if type(data).__module__.startswith("polars"):
            return cls(cudf.DataFrame.from_pandas(data.to_pandas()))
        if isinstance(data, (str, Path)):
            path = Path(data)
            suffix = path.suffix.lower()
            if suffix == ".csv":
                return cls(cudf.read_csv(str(path)))
            if suffix in {".parquet", ".pq"}:
                return cls(cudf.read_parquet(str(path)))
            raise ValueError(f"unsupported file type: {suffix}")
        raise TypeError(f"cannot build CudfEngine from {type(data).__name__}")

    def row_count(self) -> int:
        return len(self._df)

    def columns(self) -> list[str]:
        return list(self._df.columns)

    def dtypes(self) -> dict[str, str]:
        return {c: str(dt) for c, dt in self._df.dtypes.items()}

    def numeric_columns(self) -> list[str]:
        return list(self._df.select_dtypes(include="number").columns)

    def datetime_columns(self) -> list[str]:
        return list(self._df.select_dtypes(include=["datetime64[ns]"]).columns)

    def null_counts(self) -> dict[str, int]:
        return {
            c: int(v) for c, v in self._df.isna().sum().to_pandas().items()
        }

    def distinct_count(self, column: str) -> int:
        return int(self._df[column].nunique())

    def top_values(
        self,
        column: str,
        n: int = 10,
    ) -> list[tuple[str, int]]:
        counts = self._df[column].value_counts().head(n).to_pandas()
        return [(str(idx), int(cnt)) for idx, cnt in counts.items()]

    def quantiles(
        self,
        columns: list[str],
        qs: tuple[float, ...] = (0.25, 0.75),
    ) -> dict[str, dict[float, float]]:
        if not columns or not qs:
            return {}
        out: dict[str, dict[float, float]] = {c: {} for c in columns}
        for c in columns:
            vals = self._df[c].quantile(list(qs))
            series = vals.to_pandas()
            for q in qs:
                try:
                    out[c][float(q)] = float(series.loc[q])
                except KeyError:
                    out[c][float(q)] = float("nan")
        return out

    def describe(self) -> dict[str, dict[str, float]]:
        numeric = self.numeric_columns()
        if not numeric:
            return {}
        desc = self._df[numeric].describe().to_pandas()
        return {
            c: {str(k): float(v) for k, v in desc[c].items()} for c in numeric
        }

    def duplicate_count(self, subset: list[str] | None = None) -> int:
        df = self._df if subset is None else self._df[subset]
        return int(df.duplicated(keep=False).sum())

    def sample_duplicates(
        self,
        n: int,
        subset: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        base = self._df if subset is None else self._df[subset]
        mask = base.duplicated(keep=False)
        return self._df[mask].head(n).to_pandas().to_dict(orient="records")

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
        return self._df[mask].head(n).to_pandas().to_dict(orient="records")

    def max_datetime(self, column: str) -> Any:
        val = self._df[column].max()
        try:
            pd_val = val.to_pandas()
        except AttributeError:
            pd_val = val
        return pd_val
