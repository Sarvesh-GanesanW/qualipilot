"""Optional probabilistic duplicate check.

Unlike ``DuplicatesCheck`` (exact row match), this runs the
Fellegi-Sunter linker on a user-supplied comparison spec and reports
the number of probable duplicate clusters.
"""

from __future__ import annotations

from typing import Any

from datapilot.checks.base import Check, CheckContext


class LinkageCheck(Check):
    """Report probabilistic duplicate clusters.

    Active only when ``CheckConfig.linkage`` is populated.
    """

    name = "linkage"

    def _execute(self, ctx: CheckContext) -> tuple[str, dict[str, Any]]:
        link_cfg = ctx.config.linkage
        if link_cfg is None:
            return "ok", {"skipped": True}

        # deferred import so the core install stays slim
        from datapilot.linking import RecordLinker

        # engine-agnostic: always pull a polars frame for the linker.
        # the per-engine cost is tiny compared with the linkage run.
        pl_df = _engine_to_polars(ctx.engine)
        result = RecordLinker(pl_df, link_cfg).run()
        summary = result.summary()

        # clusters of size > 1 are the "probable duplicate groups"
        counts: dict[int, int] = {}
        for cid in result.clusters.values():
            counts[cid] = counts.get(cid, 0) + 1
        multi = sum(1 for sz in counts.values() if sz > 1)
        total_records_in_dupe_group = sum(
            sz for sz in counts.values() if sz > 1
        )

        severity = "warn" if multi > 0 else "ok"
        return severity, {
            **summary,
            "duplicate_clusters": multi,
            "records_in_duplicate_groups": total_records_in_dupe_group,
        }


def _engine_to_polars(engine: Any) -> Any:
    """Return the underlying frame as polars regardless of engine."""
    # PolarsEngine stores a DataFrame directly
    name = getattr(engine, "name", "")
    if name == "polars":
        return engine._df
    if name == "pandas":
        import polars as pl

        return pl.from_pandas(engine._df)
    # dask / cudf — round-trip via pandas
    import polars as pl

    underlying = engine._df
    materialised = (
        underlying.compute()
        if hasattr(underlying, "compute")
        else underlying.to_pandas()
    )
    return pl.from_pandas(materialised)
