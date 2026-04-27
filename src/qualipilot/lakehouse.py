"""Iceberg / Delta loaders that turn lakehouse tables into qualipilot inputs.

Three routes to an Iceberg table from Python, in decreasing order of
headache:

1. **Spark** via ``spark.read.format("iceberg")`` — needs JVM,
   Java 17+, and the Iceberg Spark runtime JAR on the classpath.
2. **DuckDB** via its ``iceberg`` extension — no JVM, reads directly
   from S3 / filesystem. Fast for medium-scale dedup + checks.
3. **PyIceberg** via ``pyiceberg.catalog`` — pure Python, returns
   an arrow table. Good for small/medium scans or when you do not
   want an extra query engine in-process.

The helpers below prefer (2) and (3) because they avoid the JVM.
They each return a polars DataFrame, which any qualipilot engine can
consume, or can be handed to ``DuckDBEngine.from_any`` directly.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def load_iceberg_duckdb(
    table: str,
    *,
    s3_region: str | None = None,
    s3_access_key: str | None = None,
    s3_secret_key: str | None = None,
    s3_endpoint: str | None = None,
) -> Any:
    """Load an Iceberg table via DuckDB's iceberg extension.

    Args:
        table: Iceberg metadata path or fully-qualified table id. For
            metadata-path mode pass something like
            ``"s3://bucket/warehouse/db/table"``; DuckDB discovers the
            latest snapshot under that path.
        s3_region / s3_access_key / s3_secret_key / s3_endpoint: S3
            credentials. Leave None to pick up the default AWS chain.

    Returns:
        ``polars.DataFrame`` with the table contents.

    Requires:
        ``pip install qualipilot[duckdb,iceberg]``.
    """
    import duckdb

    con = duckdb.connect(":memory:")
    con.execute("INSTALL iceberg; LOAD iceberg;")
    con.execute("INSTALL httpfs; LOAD httpfs;")

    if s3_region:
        con.execute(f"SET s3_region='{s3_region}'")
    if s3_access_key and s3_secret_key:
        con.execute(f"SET s3_access_key_id='{s3_access_key}'")
        con.execute(f"SET s3_secret_access_key='{s3_secret_key}'")
    if s3_endpoint:
        con.execute(f"SET s3_endpoint='{s3_endpoint}'")

    arrow_tbl = con.execute("SELECT * FROM iceberg_scan(?)", [table]).arrow()
    import polars as pl

    return pl.from_arrow(arrow_tbl)


def load_iceberg_pyiceberg(
    catalog_name: str,
    table_identifier: str,
    *,
    catalog_config: dict[str, str] | None = None,
    row_filter: str | None = None,
) -> Any:
    """Load an Iceberg table via pyiceberg (pure Python, no JVM).

    Args:
        catalog_name: e.g. ``"glue"``, ``"hive"``, ``"rest"``.
        table_identifier: ``"database.table"`` or ``"ns.sub.table"``.
        catalog_config: kwargs forwarded to ``load_catalog``. Example
            for AWS Glue::

                {
                    "type": "glue",
                    "s3.region": "us-east-1",
                }
        row_filter: optional SQL WHERE-style filter passed through
            pyiceberg for predicate pushdown.

    Returns:
        ``polars.DataFrame``.
    """
    from pyiceberg.catalog import load_catalog

    catalog = load_catalog(catalog_name, **(catalog_config or {}))
    table = catalog.load_table(table_identifier)
    scan = table.scan(row_filter=row_filter) if row_filter else table.scan()
    arrow_tbl = scan.to_arrow()
    import polars as pl

    return pl.from_arrow(arrow_tbl)


def load_delta(path: str) -> Any:
    """Load a Delta Lake table via ``deltalake`` + polars.

    Requires ``pip install deltalake``.
    """
    import deltalake  # type: ignore[import-not-found]
    import polars as pl

    dt = deltalake.DeltaTable(path)
    return pl.from_arrow(dt.to_pyarrow_table())
