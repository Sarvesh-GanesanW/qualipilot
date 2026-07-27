"""Cardinality and optional value-frequency checks."""

from __future__ import annotations

from typing import Any

from qualipilot.checks.base import Check, CheckContext
from qualipilot.models.results import Severity


class CardinalityCheck(Check):
    name = "cardinality"

    def _execute(self, ctx: CheckContext) -> tuple[Severity, dict[str, Any]]:
        row_count = (
            ctx.row_count
            if ctx.row_count is not None
            else ctx.engine.row_count()
        )
        total_rows = row_count or 1
        report: list[dict[str, Any]] = []
        any_constant = False

        columns = (
            ctx.columns if ctx.columns is not None else ctx.engine.columns()
        )
        distinct_counts = ctx.engine.distinct_counts(columns)
        for col in columns:
            distinct = distinct_counts[col]
            top = (
                ctx.engine.top_values(col, n=10)
                if ctx.config.include_top_values
                else []
            )
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

        severity: Severity = "warn" if any_constant else "ok"
        return severity, {"per_column": report}
