"""Missing value check.

Reports per-column null counts and percentages. Severity rules:
    * any nulls -> warn
    * columns where >50% are null -> error
"""

from __future__ import annotations

from typing import Any

from datapilot.checks.base import Check, CheckContext


class MissingValuesCheck(Check):
    name = "missing_values"

    def _execute(self, ctx: CheckContext) -> tuple[str, dict[str, Any]]:
        nulls = ctx.engine.null_counts()
        total_rows = ctx.engine.row_count() or 1
        stats = [
            {
                "column": col,
                "null_count": int(count),
                "null_percentage": round((count / total_rows) * 100, 4),
            }
            for col, count in nulls.items()
        ]

        total_nulls = sum(s["null_count"] for s in stats)
        worst = max((s["null_percentage"] for s in stats), default=0.0)

        severity = "ok"
        if worst > 50:
            severity = "error"
        elif total_nulls > 0:
            severity = "warn"

        return severity, {
            "total_null_count": total_nulls,
            "worst_column_pct": worst,
            "per_column": stats,
        }
