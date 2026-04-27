"""Duplicate row check.

Uses a global duplicate count so distributed backends do not
under-count the way ``map_partitions(.duplicated())`` would.
"""

from __future__ import annotations

from typing import Any

from qualipilot.checks.base import Check, CheckContext


class DuplicatesCheck(Check):
    name = "duplicates"

    def _execute(self, ctx: CheckContext) -> tuple[str, dict[str, Any]]:
        subset = ctx.config.duplicate_subset
        total = ctx.engine.duplicate_count(subset=subset)
        sample = ctx.engine.sample_duplicates(
            n=ctx.config.sample_size, subset=subset
        )
        severity = "warn" if total > 0 else "ok"
        return severity, {
            "total_duplicate_rows": total,
            "subset": subset,
            "sample": sample,
        }
