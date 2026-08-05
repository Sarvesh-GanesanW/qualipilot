"""Dataframe engine adapters.

The checker orchestrator talks to an ``Engine`` interface rather than a
specific dataframe library.
"""

from __future__ import annotations

from typing import Any

from qualipilot.engines._file_formats import require_safe_remote_url
from qualipilot.engines.base import Engine
from qualipilot.engines.pandas_engine import PandasEngine
from qualipilot.engines.polars_engine import PolarsEngine

__all__ = [
    "Engine",
    "PandasEngine",
    "PolarsEngine",
    "build_engine",
]


def build_engine(
    data: Any,
    kind: str = "auto",
    *,
    npartitions: int = 4,
    duckdb_threads: int | None = None,
) -> Engine:
    """Pick the right engine for the given input + requested backend.

    Args:
        data: A supported dataframe or file path.
        kind: Backend name. ``auto`` chooses from the input type.
        npartitions: Partition count passed to Dask when that engine is
            picked.
        duckdb_threads: Optional per-connection DuckDB thread limit.

    Returns:
        A concrete ``Engine`` bound to the supplied dataframe.

    Raises:
        ValueError: If ``kind`` is unknown or incompatible with
            ``data``.
        ImportError: If the optional backend package is missing.
    """
    if isinstance(data, str):
        require_safe_remote_url(data)
    resolved = _resolve_kind(data, kind)

    if resolved == "polars":
        return PolarsEngine.from_any(data)
    if resolved == "pandas":
        return PandasEngine.from_any(data)
    if resolved == "duckdb":
        from qualipilot.engines.duckdb_engine import DuckDBEngine

        return DuckDBEngine.from_any(data, threads=duckdb_threads)
    if resolved == "dask":
        from qualipilot.engines.dask_engine import DaskEngine

        return DaskEngine.from_any(data, npartitions=npartitions)
    if resolved == "spark":
        from qualipilot.engines.spark_engine import SparkEngine

        return SparkEngine.from_any(data)

    raise ValueError(f"unknown engine kind: {kind!r}")


def _resolve_kind(data: Any, kind: str) -> str:
    """Decide which engine to instantiate when ``kind='auto'``."""
    if kind != "auto":
        return kind

    # inspect object type without forcing imports of optional deps
    module = type(data).__module__
    if module.startswith("polars"):
        return "polars"
    if module.startswith("dask"):
        return "dask"
    if module.startswith("pyspark"):
        return "spark"
    if module == "_duckdb" and type(data).__name__ == "DuckDBPyRelation":
        return "duckdb"
    if module.startswith("pandas"):
        # Keep the automatic single-node path on the default backend.
        return "polars"
    if module.startswith("pyarrow"):
        return "polars"
    # strings/paths go through polars reader by default
    return "polars"
