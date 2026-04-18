"""Dataframe engine adapters.

The checker orchestrator talks to an ``Engine`` protocol rather than a
specific dataframe library. This keeps Polars (default), Pandas, Dask
and cuDF swappable without touching check code.
"""

from __future__ import annotations

from typing import Any

from datapilot.engines.base import Engine
from datapilot.engines.pandas_engine import PandasEngine
from datapilot.engines.polars_engine import PolarsEngine

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
) -> Engine:
    """Pick the right engine for the given input + requested backend.

    Args:
        data: A pandas/polars/dask/cudf dataframe, or a file path.
        kind: One of ``auto``, ``polars``, ``pandas``, ``dask``,
            ``cudf``. ``auto`` inspects ``data`` and chooses the
            lowest-overhead backend.
        npartitions: Partition count passed to Dask when that engine is
            picked.

    Returns:
        A concrete ``Engine`` bound to the supplied dataframe.

    Raises:
        ValueError: If ``kind`` is unknown or incompatible with
            ``data``.
        ImportError: If the optional backend package is missing.
    """
    resolved = _resolve_kind(data, kind)

    if resolved == "polars":
        return PolarsEngine.from_any(data)
    if resolved == "pandas":
        return PandasEngine.from_any(data)
    if resolved == "dask":
        from datapilot.engines.dask_engine import DaskEngine

        return DaskEngine.from_any(data, npartitions=npartitions)
    if resolved == "cudf":
        from datapilot.engines.cudf_engine import CudfEngine

        return CudfEngine.from_any(data)

    raise ValueError(f"unknown engine kind: {kind!r}")


def _resolve_kind(data: Any, kind: str) -> str:
    """Decide which engine to instantiate when ``kind='auto'``."""
    if kind != "auto":
        return kind

    # inspect object type without forcing imports of optional deps
    module = type(data).__module__
    if module.startswith("polars"):
        return "polars"
    if module.startswith("cudf"):
        return "cudf"
    if module.startswith("dask"):
        return "dask"
    if module.startswith("pandas"):
        # default upgrade path, polars is faster for single-node
        return "polars"
    # strings/paths go through polars reader by default
    return "polars"
