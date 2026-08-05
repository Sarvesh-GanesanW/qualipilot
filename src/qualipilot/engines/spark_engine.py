"""Optional PySpark engine."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from qualipilot.engines._file_formats import (
    require_safe_remote_url,
    require_unique_csv_columns,
    require_valid_json_lines,
)
from qualipilot.engines.base import (
    Engine,
    reject_nested_columns,
    validate_column_names,
)

if TYPE_CHECKING:  # keep the type checker happy without imports
    from pyspark.sql import DataFrame as SparkDataFrame
    from pyspark.sql import SparkSession


def _require_spark() -> None:
    try:
        import pyspark  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pyspark is required for SparkEngine; "
            "install with `pip install qualipilot[spark]` "
            "and make sure Java 17+ is on the PATH"
        ) from exc


class SparkEngine(Engine):
    """``Engine`` backed by a ``pyspark.sql.DataFrame``."""

    name = "spark"

    def __init__(self, df: SparkDataFrame) -> None:
        validate_column_names(list(df.columns))
        reject_nested_columns(
            [
                field.name
                for field in df.schema.fields
                if field.dataType.typeName()
                in {"array", "map", "struct", "variant"}
            ]
        )
        self._df = df

    @classmethod
    def from_any(
        cls,
        data: Any,
        *,
        spark_session: SparkSession | None = None,
        spark: SparkSession | None = None,
    ) -> SparkEngine:
        """Build a Spark engine from a DataFrame or file path.

        Args:
            data: one of
                * a ``pyspark.sql.DataFrame`` (used directly)
                * a plain path to Parquet / CSV / JSON
            spark_session: An existing SparkSession. If omitted we call
                ``SparkSession.getOrCreate``.
            spark: Deprecated alias for ``spark_session``.
        """
        if isinstance(data, str | Path):
            require_safe_remote_url(data)
        _require_spark()
        from pyspark.sql import DataFrame as SparkDataFrame
        from pyspark.sql import SparkSession

        if (
            spark_session is not None
            and spark is not None
            and spark_session is not spark
        ):
            raise ValueError("provide only one Spark session")
        requested_session = spark_session or spark
        session = requested_session or SparkSession.builder.getOrCreate()

        if isinstance(data, SparkDataFrame):
            if (
                requested_session is not None
                and data.sparkSession is not requested_session
            ):
                raise ValueError(
                    "Spark DataFrame belongs to a different Spark session"
                )
            return cls(data)
        if isinstance(data, str | Path):
            raw = str(data)
            suffix = Path(raw).suffix.lower()
            if suffix in {".parquet", ".pq"}:
                return cls(session.read.parquet(raw))
            if suffix == ".csv":
                _require_local_text_path(raw)
                require_unique_csv_columns(Path(raw))
                return cls(
                    session.read.option("header", True)
                    .option("inferSchema", True)
                    .option("mode", "FAILFAST")
                    .csv(raw)
                )
            if suffix in {".jsonl", ".ndjson"}:
                _require_local_text_path(raw)
                scalar_types = require_valid_json_lines(Path(raw))
                from pyspark.sql.types import (
                    BooleanType,
                    DoubleType,
                    LongType,
                    StringType,
                    StructField,
                    StructType,
                )

                dtypes = {
                    "string": StringType,
                    "integer": LongType,
                    "number": DoubleType,
                    "boolean": BooleanType,
                }
                schema = StructType(
                    [
                        StructField(column, dtypes[family](), True)
                        for column, family in scalar_types.items()
                    ]
                )
                return cls(
                    session.read.schema(schema)
                    .option("mode", "FAILFAST")
                    .json(raw)
                )
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
            if t.split("(", maxsplit=1)[0] in numeric
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

        dtypes = self.dtypes()
        exprs = []
        for column in self._df.columns:
            value = _spark_col(column)
            missing = value.isNull()
            if dtypes[column] in {"float", "double"}:
                missing = missing | F.isnan(value)
            exprs.append(F.sum(missing.cast("int")).alias(column))
        row = self._df.agg(*exprs).collect()[0].asDict()
        return {c: int(row[c] or 0) for c in self._df.columns}

    def distinct_count(self, column: str) -> int:
        return self.distinct_counts([column])[column]

    def distinct_counts(self, columns: list[str]) -> dict[str, int]:
        if not columns:
            return {}
        from pyspark.sql import functions as F

        dtypes = self.dtypes()
        expressions = []
        for column in columns:
            value = _spark_col(column)
            valid = value.isNotNull()
            if dtypes[column] in {"float", "double"}:
                valid &= ~F.isnan(value)
            expressions.append(
                F.countDistinct(F.when(valid, value)).alias(column)
            )
        row = self._df.agg(*expressions).collect()[0].asDict()
        return {column: int(row[column]) for column in columns}

    def top_values(
        self,
        column: str,
        n: int = 10,
    ) -> list[tuple[str, int]]:
        from pyspark.sql import functions as F

        value_column = "__qualipilot_value__"
        value = _spark_col(column)
        valid = value.isNotNull()
        if self.dtypes()[column] in {"float", "double"}:
            valid = valid & ~F.isnan(value)
        rows = (
            self._df.filter(valid)
            .select(value.alias(value_column))
            .groupBy(value_column)
            .count()
            .orderBy(
                F.col("count").desc(),
                F.col(value_column).cast("string").asc(),
            )
            .limit(n)
            .collect()
        )
        return [(str(row[value_column]), int(row["count"])) for row in rows]

    def quantiles(
        self,
        columns: list[str],
        qs: tuple[float, ...] = (0.25, 0.75),
    ) -> dict[str, dict[float, float]]:
        if not columns or not qs:
            return {}
        # Preserve Spark's distributed approximate-quantile behavior.
        out: dict[str, dict[float, float]] = {c: {} for c in columns}
        from pyspark.sql import functions as F

        dtypes = self.dtypes()
        aliases = [
            f"__qualipilot_quantile_{index}__" for index in range(len(columns))
        ]
        expressions = []
        for column, alias in zip(columns, aliases, strict=True):
            value = _spark_col(column)
            if dtypes[column] in {"float", "double"}:
                value = F.when(F.isnan(value), F.lit(None)).otherwise(value)
            expressions.append(value.alias(alias))
        values = self._df.select(*expressions).approxQuantile(
            aliases,
            list(qs),
            0.001,
        )
        for col, vals in zip(columns, values, strict=True):
            for index, q in enumerate(qs):
                value = vals[index] if index < len(vals) else None
                out[col][float(q)] = (
                    float(value) if value is not None else float("nan")
                )
        return out

    # ---- filters ------------------------------------------------------

    def duplicate_count(self, subset: list[str] | None = None) -> int:
        from pyspark.sql import functions as F

        cols = subset or self._df.columns
        count_column = "__qualipilot_count__"
        while count_column in self._df.columns:
            count_column += "_"
        grouped = (
            self._df.groupBy(*[self._duplicate_key(c) for c in cols])
            .agg(F.count("*").alias(count_column))
            .filter(F.col(count_column) > 1)
        )
        return int(grouped.agg(F.sum(count_column)).collect()[0][0] or 0)

    def sample_duplicates(
        self,
        n: int,
        subset: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        from pyspark.sql import Window
        from pyspark.sql import functions as F

        cols = subset or self._df.columns
        count_column = "__qualipilot_count__"
        while count_column in self._df.columns:
            count_column += "_"
        window = Window.partitionBy(*[self._duplicate_key(c) for c in cols])
        dupes = (
            self._df.withColumn(count_column, F.count("*").over(window))
            .filter(F.col(count_column) > 1)
            .drop(count_column)
            .limit(n)
        )
        return [r.asDict() for r in dupes.collect()]

    def count_outside(self, column: str, low: float, high: float) -> int:
        return self.counts_outside({column: (low, high)})[column]

    def counts_outside(
        self,
        ranges: dict[str, tuple[float, float]],
    ) -> dict[str, int]:
        if not ranges:
            return {}
        from pyspark.sql import functions as F

        row = (
            self._df.agg(
                *(
                    F.sum(
                        self._outside_mask(column, low, high).cast("int")
                    ).alias(column)
                    for column, (low, high) in ranges.items()
                )
            )
            .collect()[0]
            .asDict()
        )
        return {column: int(row[column] or 0) for column in ranges}

    def sample_outside(
        self, column: str, low: float, high: float, n: int
    ) -> list[dict[str, Any]]:
        rows = self._df.filter(self._outside_mask(column, low, high)).limit(n)
        return [r.asDict() for r in rows.collect()]

    def max_datetime(self, column: str) -> Any:
        from pyspark.sql import functions as F

        return self._df.agg(F.max(_spark_col(column))).collect()[0][0]

    def _outside_mask(self, column: str, low: float, high: float) -> Any:
        from pyspark.sql import functions as F

        value = _spark_col(column)
        mask = (value < low) | (value > high)
        if self.dtypes()[column] in {"float", "double"}:
            mask &= ~F.isnan(value)
        return mask

    def _duplicate_key(self, column: str) -> Any:
        from pyspark.sql import functions as F

        value = _spark_col(column)
        if self.dtypes()[column] in {"float", "double"}:
            return F.when(F.isnan(value), F.lit(None)).otherwise(value)
        return value


def _spark_col(name: str) -> Any:
    """Resolve a literal Spark column name, including dots/backticks."""
    from pyspark.sql import functions as F

    return F.col(_spark_identifier(name))


def _require_local_text_path(path: str) -> None:
    if "://" in path:
        raise ValueError(
            "Spark remote CSV and JSONL inputs are not supported because "
            "their raw records cannot be validated; use Parquet or a local "
            "text file"
        )


def _spark_identifier(name: str) -> str:
    return f"`{name.replace('`', '``')}`"
