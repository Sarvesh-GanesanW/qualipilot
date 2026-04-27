"""DuckDB engine.

DuckDB is an in-process columnar SQL engine. It is typically the
single fastest option on one machine: vectorised execution, parallel
SIMD ops, arrow-native zero-copy interop with Polars/Pandas.

We hold the input frame as a DuckDB relation and answer every
``Engine`` method with a single SQL query. The relation is keyed off
a reserved view name ``_t`` registered against a per-instance
connection, so two engines built over the same process stay
isolated.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, cast

from qualipilot.engines.base import Engine

try:
    import duckdb
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "duckdb is required for DuckDBEngine; "
        "install with `pip install qualipilot[duckdb]`"
    ) from exc

# generate unique view names so the engine can stack inside the same
# interpreter (e.g. a notebook that creates several checkers)
_name_counter = itertools.count()


class DuckDBEngine(Engine):
    """``Engine`` backed by a DuckDB relation."""

    name = "duckdb"

    def __init__(self, con: duckdb.DuckDBPyConnection, view: str) -> None:
        self._con = con
        self._view = view

    @classmethod
    def from_any(cls, data: Any) -> DuckDBEngine:
        con = duckdb.connect(database=":memory:")
        # duckdb prefers multithreaded scans on a dedicated conn
        con.execute("PRAGMA threads=8")
        view = f"_t_{next(_name_counter)}"

        if isinstance(data, (str, Path)):
            path = Path(data)
            suffix = path.suffix.lower()
            # duckdb's read_*_auto table functions reject ? parameters,
            # so we inline the path. Single quotes are escaped to keep
            # this SQL-injection-safe for filesystem paths.
            literal = str(path).replace("'", "''")
            if suffix == ".csv":
                con.execute(
                    f"CREATE VIEW {view} AS "
                    f"SELECT * FROM read_csv_auto('{literal}')"
                )
            elif suffix in {".parquet", ".pq"}:
                con.execute(
                    f"CREATE VIEW {view} AS "
                    f"SELECT * FROM read_parquet('{literal}')"
                )
            elif suffix in {".ndjson", ".jsonl", ".json"}:
                con.execute(
                    f"CREATE VIEW {view} AS "
                    f"SELECT * FROM read_json_auto('{literal}')"
                )
            else:
                raise ValueError(f"unsupported file type: {suffix}")
        elif type(data).__module__.startswith("pandas"):
            # register works zero-copy via arrow
            con.register(view, data)
        elif type(data).__module__.startswith("polars"):
            # polars -> arrow -> duckdb is also zero-copy
            con.register(view, data.to_arrow())
        elif type(data).__module__.startswith("pyarrow"):
            con.register(view, data)
        else:
            raise TypeError(
                f"cannot build DuckDBEngine from {type(data).__name__}"
            )
        return cls(con, view)

    # ---- structural info ----------------------------------------------

    def row_count(self) -> int:
        return int(self._scalar(f"SELECT COUNT(*) FROM {self._view}"))

    def columns(self) -> list[str]:
        return [
            row[0]
            for row in self._con.execute(
                f"DESCRIBE SELECT * FROM {self._view}"
            ).fetchall()
        ]

    def dtypes(self) -> dict[str, str]:
        rows = self._con.execute(
            f"DESCRIBE SELECT * FROM {self._view}"
        ).fetchall()
        return {name: str(dtype) for name, dtype, *_ in rows}

    def numeric_columns(self) -> list[str]:
        numeric_types = {
            "TINYINT",
            "SMALLINT",
            "INTEGER",
            "BIGINT",
            "HUGEINT",
            "FLOAT",
            "DOUBLE",
            "DECIMAL",
            "UTINYINT",
            "USMALLINT",
            "UINTEGER",
            "UBIGINT",
        }
        return [
            c
            for c, t in self.dtypes().items()
            if t.split("(")[0] in numeric_types
        ]

    def datetime_columns(self) -> list[str]:
        dt_types = {"DATE", "TIMESTAMP", "TIMESTAMP_S", "TIMESTAMP_MS"}
        return [
            c
            for c, t in self.dtypes().items()
            if t.split("(")[0].split(" ")[0] in dt_types
        ]

    # ---- per-column stats ---------------------------------------------

    def null_counts(self) -> dict[str, int]:
        cols = self.columns()
        if not cols:
            return {}
        # SUM(CASE WHEN x IS NULL THEN 1 ELSE 0 END) per column in one pass
        selects = ", ".join(
            f'SUM(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END) AS "{c}"'
            for c in cols
        )
        row = self._row(f"SELECT {selects} FROM {self._view}")
        return {c: int(v or 0) for c, v in zip(cols, row, strict=True)}

    def distinct_count(self, column: str) -> int:
        return int(
            self._scalar(
                f'SELECT COUNT(DISTINCT "{column}") FROM {self._view}'
            )
        )

    def top_values(
        self,
        column: str,
        n: int = 10,
    ) -> list[tuple[str, int]]:
        rows = self._con.execute(
            f'SELECT "{column}" AS v, COUNT(*) AS c FROM {self._view} '
            f'WHERE "{column}" IS NOT NULL '
            f"GROUP BY 1 ORDER BY c DESC LIMIT ?",
            [n],
        ).fetchall()
        return [(str(v), int(c)) for v, c in rows]

    def quantiles(
        self,
        columns: list[str],
        qs: tuple[float, ...] = (0.25, 0.75),
    ) -> dict[str, dict[float, float]]:
        if not columns or not qs:
            return {}
        # quantile_cont is exact in duckdb; quantile() is approximate
        # but faster — we use the exact one for tight IQR bounds
        selects = ", ".join(
            f'quantile_cont("{c}", {q}) AS "{c}__{int(q * 1000)}"'
            for c in columns
            for q in qs
        )
        row = self._row(f"SELECT {selects} FROM {self._view}")
        out: dict[str, dict[float, float]] = {c: {} for c in columns}
        idx = 0
        for c in columns:
            for q in qs:
                val = row[idx]
                out[c][float(q)] = (
                    float(val) if val is not None else float("nan")
                )
                idx += 1
        return out

    def describe(self) -> dict[str, dict[str, float]]:
        numeric = self.numeric_columns()
        if not numeric:
            return {}
        rel = self._con.sql(f"SELECT * FROM {self._view}")
        desc = rel.describe().fetchdf()
        stat_col = desc.columns[0]
        out: dict[str, dict[str, float]] = {c: {} for c in numeric}
        for _, r in desc.iterrows():
            stat = str(r[stat_col])
            for c in numeric:
                val = r.get(c)
                if val is None:
                    continue
                try:
                    out[c][stat] = float(val)
                except (TypeError, ValueError):
                    continue
        return out

    # ---- filters ------------------------------------------------------

    def duplicate_count(self, subset: list[str] | None = None) -> int:
        cols_sql = _quoted_list(subset or self.columns())
        return int(
            self._scalar(
                f"""
                WITH counts AS (
                    SELECT {cols_sql}, COUNT(*) AS n
                    FROM {self._view}
                    GROUP BY {cols_sql}
                    HAVING COUNT(*) > 1
                )
                SELECT COALESCE(SUM(n), 0) FROM counts
                """
            )
        )

    def sample_duplicates(
        self,
        n: int,
        subset: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        cols = subset or self.columns()
        cols_sql = _quoted_list(cols)
        rows = self._con.execute(
            f"""
            WITH dupes AS (
                SELECT *, COUNT(*) OVER (PARTITION BY {cols_sql}) AS _n
                FROM {self._view}
            )
            SELECT * EXCLUDE (_n) FROM dupes WHERE _n > 1 LIMIT ?
            """,
            [n],
        ).fetchdf()
        return cast(list[dict[str, Any]], rows.to_dict(orient="records"))

    def count_outside(self, column: str, low: float, high: float) -> int:
        return int(
            self._scalar(
                f"SELECT COUNT(*) FROM {self._view} "
                f'WHERE "{column}" < ? OR "{column}" > ?',
                [low, high],
            )
        )

    def sample_outside(
        self, column: str, low: float, high: float, n: int
    ) -> list[dict[str, Any]]:
        rows = self._con.execute(
            f"SELECT * FROM {self._view} "
            f'WHERE "{column}" < ? OR "{column}" > ? LIMIT ?',
            [low, high, n],
        ).fetchdf()
        return cast(list[dict[str, Any]], rows.to_dict(orient="records"))

    def max_datetime(self, column: str) -> Any:
        return self._scalar(f'SELECT MAX("{column}") FROM {self._view}')

    # ---- internals -----------------------------------------------------

    def _row(
        self,
        sql: str,
        params: list[Any] | None = None,
    ) -> tuple[Any, ...]:
        """Run a query expected to return exactly one row, non-None."""
        cursor = (
            self._con.execute(sql, params)
            if params is not None
            else self._con.execute(sql)
        )
        result = cursor.fetchone()
        if not isinstance(result, tuple):
            raise RuntimeError(f"duckdb returned no row for: {sql!r}")
        return result

    def _scalar(self, sql: str, params: list[Any] | None = None) -> Any:
        return self._row(sql, params)[0]


def _quoted_list(cols: list[str]) -> str:
    return ", ".join(f'"{c}"' for c in cols)
