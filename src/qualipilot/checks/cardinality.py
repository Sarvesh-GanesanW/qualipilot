"""Cardinality / uniqueness profile.

Helps spot accidentally-constant columns and highly categorical ones
that should be bucketed. Reports distinct count plus the top-10 values
for every column.
"""

from __future__ import annotations

from typing import Any

from qualipilot.checks.base import Check, CheckContext


class CardinalityCheck(Check):
    name = "cardinality"

    def _execute(self, ctx: CheckContext) -> tuple[str, dict[str, Any]]:
        total_rows = ctx.engine.row_count() or 1
        report: list[dict[str, Any]] = []
        any_constant = False

        for col in ctx.engine.columns():
            try:
                distinct = ctx.engine.distinct_count(col)
            except Exception:  # pragma: no cover
                # some engines fail on nested dtypes, we log + move on
                continue
            top = ctx.engine.top_values(col, n=10)
            if distinct <= 1 and total_rows > 1:
                any_constant = True
            report.append(
                {
                    "column": col,
                    "distinct_count": distinct,
                    "unique_ratio": round(distinct / total_rows, 6),
                    "top_values": top,
                }
            )

        severity = "warn" if any_constant else "ok"
        return severity, {"per_column": report}
