"""DuckDB-backed dataframe engine."""

from __future__ import annotations

import math
from contextlib import suppress
from pathlib import Path
from typing import Any

from qualipilot.engines._duckdb_sql import quote_identifier, quote_literal
from qualipilot.engines._file_formats import (
    require_unique_csv_columns,
    require_valid_json_lines,
)
from qualipilot.engines.base import (
    Engine,
    as_utc_datetime,
    reject_nested_columns,
    validate_column_names,
    validate_pandas_columns,
)

try:
    import duckdb
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "duckdb is required for DuckDBEngine; "
        "install with `pip install qualipilot[duckdb]`"
    ) from exc


class DuckDBEngine(Engine):
    """``Engine`` backed by a DuckDB relation."""

    name = "duckdb"

    def __init__(
        self,
        con: duckdb.DuckDBPyConnection | None,
        view: str,
        *,
        relation: duckdb.DuckDBPyRelation | None = None,
    ) -> None:
        if (con is None) == (relation is None):
            raise ValueError("provide exactly one DuckDB source")
        self._con = con
        self._relation = relation
        self._view = view
        self._dtypes_cache: dict[str, str] | None = None
        self._closed = False
        validate_column_names(list(self.dtypes()))
        reject_nested_columns(
            [
                column
                for column, dtype in self.dtypes().items()
                if dtype.startswith(("STRUCT(", "MAP(", "UNION("))
                or dtype.endswith("]")
            ]
        )

    @classmethod
    def from_any(
        cls,
        data: Any,
        *,
        threads: int | None = None,
        duckdb_connection: duckdb.DuckDBPyConnection | None = None,
    ) -> DuckDBEngine:
        _validate_source_columns(data)
        if duckdb_connection is not None and not isinstance(
            duckdb_connection, duckdb.DuckDBPyConnection
        ):
            raise TypeError("duckdb_connection must be a DuckDB connection")
        if threads is not None and threads < 1:
            raise ValueError("threads must be >= 1")
        if type(data).__module__.startswith("polars"):
            unsupported = [
                column
                for column, dtype in data.schema.items()
                if str(dtype) in {"Int128", "UInt128"}
            ]
            if unsupported:
                raise ValueError(
                    "DuckDB engine does not support 128-bit Polars "
                    f"columns: {unsupported}"
                )

        if duckdb_connection is not None and threads is not None:
            duckdb_connection.execute("SET threads = ?", [threads])
        if (
            isinstance(data, duckdb.DuckDBPyRelation)
            and duckdb_connection is None
        ):
            return cls(None, "_t", relation=data)
        if duckdb_connection is not None:
            return cls(
                None,
                "_t",
                relation=_relation_from_connection(
                    duckdb_connection,
                    data,
                ),
            )

        con = duckdb.connect(database=":memory:")
        view = "_t"
        try:
            if threads is not None:
                con.execute("SET threads = ?", [threads])

            _register_private_source(con, view, data)
            return cls(con, view)
        except Exception:
            con.close()
            raise

    def close(self) -> None:
        """Release the private DuckDB connection and registered inputs."""
        if self._con is not None and not self._closed:
            self._con.close()
            self._closed = True

    def __enter__(self) -> DuckDBEngine:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()

    # ---- structural info ----------------------------------------------

    def row_count(self) -> int:
        view = quote_identifier(self._view)
        return int(self._scalar(f"SELECT COUNT(*) FROM {view}"))

    def columns(self) -> list[str]:
        return list(self.dtypes())

    def dtypes(self) -> dict[str, str]:
        if self._dtypes_cache is None:
            if self._relation is not None:
                self._dtypes_cache = {
                    name: str(dtype)
                    for name, dtype in zip(
                        self._relation.columns,
                        self._relation.types,
                        strict=True,
                    )
                }
            else:
                rows = self._execute(
                    f"DESCRIBE SELECT * FROM {quote_identifier(self._view)}"
                ).fetchall()
                self._dtypes_cache = {
                    name: str(dtype) for name, dtype, *_ in rows
                }
        return dict(self._dtypes_cache)

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
        dt_types = {
            "DATE",
            "TIMESTAMP",
            "TIMESTAMP_S",
            "TIMESTAMP_MS",
            "TIMESTAMP_NS",
        }
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
            f"SUM(CASE WHEN {self._missing_sql(c)} THEN 1 ELSE 0 END) "
            f"AS {quote_identifier(c)}"
            for c in cols
        )
        row = self._row(
            f"SELECT {selects} FROM {quote_identifier(self._view)}"
        )
        return {c: int(v or 0) for c, v in zip(cols, row, strict=True)}

    def distinct_count(self, column: str) -> int:
        return self.distinct_counts([column])[column]

    def distinct_counts(self, columns: list[str]) -> dict[str, int]:
        if not columns:
            return {}
        selects = ", ".join(
            f"COUNT(DISTINCT CASE WHEN {self._missing_sql(column)} "
            f"THEN NULL ELSE {quote_identifier(column)} END) "
            f"AS {quote_identifier(column)}"
            for column in columns
        )
        row = self._row(
            f"SELECT {selects} FROM {quote_identifier(self._view)}"
        )
        return {
            column: int(value)
            for column, value in zip(columns, row, strict=True)
        }

    def top_values(
        self,
        column: str,
        n: int = 10,
    ) -> list[tuple[str, int]]:
        identifier = quote_identifier(column)
        rows = self._execute(
            f"SELECT {identifier} AS v, COUNT(*) AS c "
            f"FROM {quote_identifier(self._view)} "
            f"WHERE NOT ({self._missing_sql(column)}) "
            f"GROUP BY 1 ORDER BY c DESC, CAST(v AS VARCHAR) ASC LIMIT ?",
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
            f"quantile_cont({quote_identifier(c)}, {float(q)!r}) "
            f"FILTER (WHERE NOT ({self._missing_sql(c)})) "
            f"AS {quote_identifier(f'{c}__q{index}')}"
            for c in columns
            for index, q in enumerate(qs)
        )
        row = self._row(
            f"SELECT {selects} FROM {quote_identifier(self._view)}"
        )
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

    # ---- filters ------------------------------------------------------

    def duplicate_count(self, subset: list[str] | None = None) -> int:
        columns = subset or self.columns()
        if not columns:
            row_count = self.row_count()
            return row_count if row_count > 1 else 0
        cols_sql = self._duplicate_keys_sql(columns)
        return int(
            self._scalar(
                f"""
                WITH counts AS (
                    SELECT COUNT(*) AS duplicate_count
                    FROM {quote_identifier(self._view)}
                    GROUP BY {cols_sql}
                    HAVING COUNT(*) > 1
                )
                SELECT COALESCE(SUM(duplicate_count), 0) FROM counts
                """
            )
        )

    def sample_duplicates(
        self,
        n: int,
        subset: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        cols = subset or self.columns()
        if not cols:
            return [{}] * min(n, self.duplicate_count())
        cols_sql = self._duplicate_keys_sql(cols)
        return self._records(
            f"""
            SELECT *
            FROM {quote_identifier(self._view)}
            QUALIFY COUNT(*) OVER (PARTITION BY {cols_sql}) > 1
            LIMIT ?
            """,
            [n],
        )

    def count_outside(self, column: str, low: float, high: float) -> int:
        return self.counts_outside({column: (low, high)})[column]

    def counts_outside(
        self,
        ranges: dict[str, tuple[float, float]],
    ) -> dict[str, int]:
        if not ranges:
            return {}
        columns = list(ranges)
        selects = ", ".join(
            "COUNT(*) FILTER (WHERE NOT "
            f"({self._missing_sql(column)}) AND "
            f"({quote_identifier(column)} < ? OR "
            f"{quote_identifier(column)} > ?)) "
            f"AS {quote_identifier(column)}"
            for column in columns
        )
        params = [bound for column in columns for bound in ranges[column]]
        row = self._row(
            f"SELECT {selects} FROM {quote_identifier(self._view)}",
            params,
        )
        return {
            column: int(value)
            for column, value in zip(columns, row, strict=True)
        }

    def sample_outside(
        self, column: str, low: float, high: float, n: int
    ) -> list[dict[str, Any]]:
        identifier = quote_identifier(column)
        return self._records(
            f"SELECT * FROM {quote_identifier(self._view)} "
            f"WHERE NOT ({self._missing_sql(column)}) "
            f"AND ({identifier} < ? OR {identifier} > ?) LIMIT ?",
            [low, high, n],
        )

    def max_datetime(self, column: str) -> Any:
        return self._scalar(
            f"SELECT MAX({quote_identifier(column)}) "
            f"FROM {quote_identifier(self._view)}"
        )

    def max_datetime_instant(
        self,
        column: str,
        naive_timezone: str = "UTC",
    ) -> Any:
        if self.dtypes()[column].split("(")[0] != "VARCHAR":
            return super().max_datetime_instant(column, naive_timezone)

        identifier = quote_identifier(column)
        text = f"trim(CAST({identifier} AS VARCHAR))"
        offset_pattern = quote_literal(r"(?:[zZ]|[+-]\d{2}:?\d{2})$")
        has_offset = f"regexp_matches({text}, {offset_pattern})"
        parsed = (
            f"CASE WHEN {has_offset} "
            f"THEN TRY_CAST({text} AS TIMESTAMPTZ) "
            f"ELSE timezone({quote_literal(naive_timezone)}, "
            f"TRY_CAST({text} AS TIMESTAMP)) END"
        )
        max_timestamp, invalid_count = self._row(
            f"SELECT MAX({parsed}), COUNT(*) FILTER ("
            f"WHERE {identifier} IS NOT NULL AND {parsed} IS NULL) "
            f"FROM {quote_identifier(self._view)}"
        )
        if invalid_count:
            raise ValueError(
                f"freshness column {column!r} contains invalid timestamps"
            )
        return (
            None
            if max_timestamp is None
            else as_utc_datetime(max_timestamp, "UTC")
        )

    # ---- internals -----------------------------------------------------

    def _row(
        self,
        sql: str,
        params: list[Any] | None = None,
    ) -> tuple[Any, ...]:
        """Run a query expected to return exactly one row, non-None."""
        cursor = self._execute(sql, params)
        result = cursor.fetchone()
        if not isinstance(result, tuple):
            raise RuntimeError(f"duckdb returned no row for: {sql!r}")
        return result

    def _scalar(self, sql: str, params: list[Any] | None = None) -> Any:
        return self._row(sql, params)[0]

    def _records(
        self, sql: str, params: list[Any] | None = None
    ) -> list[dict[str, Any]]:
        cursor = self._execute(sql, params)
        columns = (
            list(cursor.columns)
            if self._relation is not None
            else [description[0] for description in cursor.description]
        )
        return [
            dict(zip(columns, row, strict=True)) for row in cursor.fetchall()
        ]

    def _execute(
        self,
        sql: str,
        params: list[Any] | None = None,
    ) -> Any:
        if self._relation is not None:
            return self._relation.query(
                self._view,
                _bind_relation_parameters(sql, params or []),
            )
        if self._con is None:  # pragma: no cover - constructor invariant
            raise RuntimeError("DuckDB engine has no source")
        return (
            self._con.execute(sql, params)
            if params is not None
            else self._con.execute(sql)
        )

    def _missing_sql(self, column: str) -> str:
        identifier = quote_identifier(column)
        dtype = self.dtypes()[column].split("(")[0]
        if dtype in {"FLOAT", "DOUBLE"}:
            return f"{identifier} IS NULL OR isnan({identifier})"
        return f"{identifier} IS NULL"

    def _duplicate_keys_sql(self, columns: list[str]) -> str:
        dtypes = self.dtypes()
        keys = []
        for column in columns:
            identifier = quote_identifier(column)
            if dtypes[column].split("(")[0] in {"FLOAT", "DOUBLE"}:
                keys.append(
                    f"CASE WHEN isnan({identifier}) THEN NULL "
                    f"ELSE {identifier} END"
                )
            else:
                keys.append(identifier)
        return ", ".join(keys)


def _file_source(path: Path) -> str:
    """Return the DuckDB reader expression for a local file."""
    suffix = path.suffix.lower()
    literal = quote_literal(str(path))
    if suffix == ".csv":
        require_unique_csv_columns(path)
        return f"read_csv_auto({literal}, delim = ',')"
    if suffix in {".parquet", ".pq"}:
        return f"read_parquet({literal})"
    if suffix in {".ndjson", ".jsonl"}:
        scalar_types = require_valid_json_lines(path)
        dtypes = {
            "string": "VARCHAR",
            "integer": "BIGINT",
            "number": "DOUBLE",
            "boolean": "BOOLEAN",
        }
        columns = ", ".join(
            f"{quote_identifier(column)}: {quote_literal(dtypes[family])}"
            for column, family in scalar_types.items()
        )
        return (
            f"read_json_auto({literal}, format = 'newline_delimited', "
            f"columns = {{{columns}}})"
        )
    raise ValueError(f"unsupported file type: {suffix}")


def _relation_from_connection(
    connection: duckdb.DuckDBPyConnection,
    data: Any,
) -> duckdb.DuckDBPyRelation:
    """Bind a supported input to a caller-owned DuckDB connection."""
    if isinstance(data, duckdb.DuckDBPyRelation):
        return connection.sql("SELECT * FROM data")
    if isinstance(data, str | Path):
        return connection.sql(f"SELECT * FROM {_file_source(Path(data))}")
    if type(data).__module__.startswith("pandas"):
        return connection.from_df(data)
    if type(data).__module__.startswith("polars"):
        return connection.from_arrow(data.to_arrow())
    if type(data).__module__.startswith("pyarrow"):
        return connection.from_arrow(data)
    raise TypeError(f"cannot build DuckDBEngine from {type(data).__name__}")


def _register_private_source(
    connection: duckdb.DuckDBPyConnection,
    view: str,
    data: Any,
) -> None:
    """Register a supported input on an engine-owned connection."""
    if isinstance(data, str | Path):
        source = _file_source(Path(data))
        connection.execute(
            f"CREATE VIEW {quote_identifier(view)} AS SELECT * FROM {source}"
        )
    elif type(data).__module__.startswith("pandas"):
        connection.register(view, data)
    elif type(data).__module__.startswith("polars"):
        connection.register(view, data.to_arrow())
    elif type(data).__module__.startswith("pyarrow"):
        connection.register(view, data)
    else:
        raise TypeError(
            f"cannot build DuckDBEngine from {type(data).__name__}"
        )


def _validate_source_columns(data: Any) -> None:
    module = type(data).__module__
    if module.startswith("pandas"):
        validate_pandas_columns(data)
    elif module.startswith("polars"):
        validate_column_names(data.columns)
    elif module.startswith("pyarrow"):
        validate_column_names(data.column_names)
    elif isinstance(data, duckdb.DuckDBPyRelation):
        validate_column_names(data.columns)
    elif isinstance(data, str | Path):
        path = Path(data)
        if path.suffix.lower() in {".parquet", ".pq"}:
            import pyarrow.parquet as pq

            validate_column_names(pq.read_schema(path).names)


def _bind_relation_parameters(
    sql: str,
    params: list[Any],
) -> str:
    parameter = iter(_relation_parameter_literals(params))
    result: list[str] = []
    quote: str | None = None
    index = 0
    while index < len(sql):
        character = sql[index]
        result.append(character)
        if quote is not None:
            if character == quote:
                if index + 1 < len(sql) and sql[index + 1] == quote:
                    result.append(sql[index + 1])
                    index += 1
                else:
                    quote = None
        elif character in {'"', "'"}:
            quote = character
        elif character == "?":
            try:
                result[-1] = next(parameter)
            except StopIteration as exc:
                raise ValueError(
                    "missing DuckDB relation query parameter"
                ) from exc
        index += 1

    try:
        next(parameter)
    except StopIteration:
        return "".join(result)
    raise ValueError("too many DuckDB relation query parameters")


def _relation_parameter_literals(params: list[Any]) -> list[str]:
    literals: list[str] = []
    for value in params:
        if isinstance(value, bool | int):
            literals.append(str(value).upper())
        elif isinstance(value, float) and math.isfinite(value):
            literals.append(repr(value))
        else:
            raise TypeError(
                "DuckDB relation parameters must be finite numbers"
            )
    return literals
