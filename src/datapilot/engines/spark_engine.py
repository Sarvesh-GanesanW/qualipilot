"""PySpark engine.

Spark is the right fit when your data already lives in a lakehouse
(Delta, Iceberg, Hudi) and you have a cluster. On a laptop, DuckDB
or Polars will be faster and less fiddly — see ``DuckDBEngine``.

This file is import-light by design: pyspark is optional and loading
it triggers a JVM boot. We defer every touch of ``pyspark`` to the
first method call so the core install stays usable without Java.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from datapilot.engines.base import Engine

if TYPE_CHECKING:  # keep the type checker happy without imports
    from pyspark.sql import DataFrame as SparkDataFrame
    from pyspark.sql import SparkSession


def _require_spark() -> None:
    try:
        import pyspark  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pyspark is required for SparkEngine; "
            "install with `pip install data-pilot-checker[spark]` "
            "and make sure Java 17+ is on the PATH"
        ) from exc


class SparkEngine(Engine):
    """``Engine`` backed by a ``pyspark.sql.DataFrame``."""

    name = "spark"

    def __init__(self, df: SparkDataFrame) -> None:
        self._df = df

    @classmethod
    def from_any(
        cls,
        data: Any,
        *,
        spark: SparkSession | None = None,
    ) -> SparkEngine:
        """Build a Spark engine from a DataFrame, path, or Iceberg ref.

        Args:
            data: one of
                * a ``pyspark.sql.DataFrame`` (used directly)
                * ``"iceberg://<catalog>.<db>.<table>"`` — loaded via
                  the ``iceberg`` spark source
                * ``"delta://<path>"``
                * a plain path to Parquet / CSV / JSON
            spark: an existing SparkSession. If omitted we call
                ``SparkSession.getOrCreate``.
        """
        _require_spark()
        from pyspark.sql import DataFrame as SparkDataFrame
        from pyspark.sql import SparkSession

        session = spark or SparkSession.builder.getOrCreate()

        if isinstance(data, SparkDataFrame):
            return cls(data)
        if isinstance(data, (str, Path)):
            raw = str(data)
            if raw.startswith("iceberg://"):
                ref = raw[len("iceberg://") :]
                return cls(session.read.format("iceberg").load(ref))
            if raw.startswith("delta://"):
                ref = raw[len("delta://") :]
                return cls(session.read.format("delta").load(ref))
            suffix = Path(raw).suffix.lower()
            if suffix in {".parquet", ".pq"}:
                return cls(session.read.parquet(raw))
            if suffix == ".csv":
                return cls(
                    session.read.option("header", True)
                    .option("inferSchema", True)
                    .csv(raw)
                )
            if suffix in {".json", ".jsonl", ".ndjson"}:
                return cls(session.read.json(raw))
            raise ValueError(f"unsupported file type: {suffix}")
        if type(data).__module__.startswith("pandas"):
            return cls(session.createDataFrame(data))
        raise TypeError(f"cannot build SparkEngine from {type(data).__name__}")

    # ---- structural info ----------------------------------------------

    def row_count(self) -> int:
        return int(self._df.count())

    def columns(self) -> list[str]:
        return list(self._df.columns)

    def dtypes(self) -> dict[str, str]:
        return {name: dtype for name, dtype in self._df.dtypes}

    def numeric_columns(self) -> list[str]:
        numeric = {
            "tinyint",
            "smallint",
            "int",
            "bigint",
            "float",
            "double",
            "decimal",
        }
        return [
            c
            for c, t in self.dtypes().items()
            if any(t.startswith(x) for x in numeric)
        ]

    def datetime_columns(self) -> list[str]:
        return [
            c
            for c, t in self.dtypes().items()
            if t in {"date", "timestamp", "timestamp_ntz"}
        ]

    # ---- per-column stats ---------------------------------------------

    def null_counts(self) -> dict[str, int]:
        from pyspark.sql import functions as F

        exprs = [
            F.sum(F.col(c).isNull().cast("int")).alias(c)
            for c in self._df.columns
        ]
        row = self._df.agg(*exprs).collect()[0].asDict()
        return {c: int(row[c] or 0) for c in self._df.columns}

    def distinct_count(self, column: str) -> int:
        from pyspark.sql import functions as F

        return int(
            self._df.agg(F.countDistinct(F.col(column))).collect()[0][0]
        )

    def top_values(
        self,
        column: str,
        n: int = 10,
    ) -> list[tuple[str, int]]:
        from pyspark.sql import functions as F

        rows = (
            self._df.filter(F.col(column).isNotNull())
            .groupBy(column)
            .count()
            .orderBy(F.col("count").desc())
            .limit(n)
            .collect()
        )
        return [(str(r[0]), int(r[1])) for r in rows]

    def quantiles(
        self,
        columns: list[str],
        qs: tuple[float, ...] = (0.25, 0.75),
    ) -> dict[str, dict[float, float]]:
        if not columns or not qs:
            return {}
        # approxQuantile is O(N) with a tight error bound — exact
        # percentiles over a cluster would force a shuffle per col
        out: dict[str, dict[float, float]] = {c: {} for c in columns}
        for col in columns:
            vals = self._df.approxQuantile(col, list(qs), 0.001)
            for q, v in zip(qs, vals, strict=True):
                out[col][float(q)] = (
                    float(v) if v is not None else float("nan")
                )
        return out

    def describe(self) -> dict[str, dict[str, float]]:
        numeric = self.numeric_columns()
        if not numeric:
            return {}
        desc = self._df.select(*numeric).describe().collect()
        out: dict[str, dict[str, float]] = {c: {} for c in numeric}
        for row in desc:
            stat = row["summary"]
            for c in numeric:
                val = row[c]
                if val is None:
                    continue
                try:
                    out[c][stat] = float(val)
                except (TypeError, ValueError):
                    continue
        return out

    # ---- filters ------------------------------------------------------

    def duplicate_count(self, subset: list[str] | None = None) -> int:
        from pyspark.sql import functions as F

        cols = subset or self._df.columns
        grouped = self._df.groupBy(*cols).count().filter(F.col("count") > 1)
        return int(grouped.agg(F.sum("count")).collect()[0][0] or 0)

    def sample_duplicates(
        self,
        n: int,
        subset: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        from pyspark.sql import Window
        from pyspark.sql import functions as F

        cols = subset or self._df.columns
        window = Window.partitionBy(*cols)
        dupes = (
            self._df.withColumn("_n", F.count("*").over(window))
            .filter(F.col("_n") > 1)
            .drop("_n")
            .limit(n)
        )
        return [r.asDict() for r in dupes.collect()]

    def count_outside(self, column: str, low: float, high: float) -> int:
        from pyspark.sql import functions as F

        c = F.col(column)
        return int(self._df.filter((c < low) | (c > high)).count())

    def sample_outside(
        self, column: str, low: float, high: float, n: int
    ) -> list[dict[str, Any]]:
        from pyspark.sql import functions as F

        c = F.col(column)
        rows = self._df.filter((c < low) | (c > high)).limit(n)
        return [r.asDict() for r in rows.collect()]

    def max_datetime(self, column: str) -> Any:
        from pyspark.sql import functions as F

        return self._df.agg(F.max(column)).collect()[0][0]
