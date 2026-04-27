"""Range validation check.

Given ``column_ranges`` in the config, counts and samples rows whose
values fall outside the allowed ``[min, max]`` envelope.
"""

from __future__ import annotations

from typing import Any

from qualipilot.checks.base import Check, CheckContext


class RangesCheck(Check):
    name = "ranges"

    def _execute(self, ctx: CheckContext) -> tuple[str, dict[str, Any]]:
        ranges = ctx.config.column_ranges
        if not ranges:
            return "ok", {"per_column": []}

        existing = set(ctx.engine.columns())
        report: list[dict[str, Any]] = []
        any_violations = False

        for col, spec in ranges.items():
            if col not in existing:
                # skip columns that were dropped upstream; surfaces as
                # a warning via missing data rather than hard fail
                report.append(
                    {
                        "column": col,
                        "min_allowed": spec.min,
                        "max_allowed": spec.max,
                        "violation_count": 0,
                        "sample": [],
                        "note": "column not present in dataset",
                    }
                )
                continue

            count = ctx.engine.count_outside(col, spec.min, spec.max)
            sample = (
                ctx.engine.sample_outside(
                    col, spec.min, spec.max, ctx.config.sample_size
                )
                if count
                else []
            )
            if count:
                any_violations = True
            report.append(
                {
                    "column": col,
                    "min_allowed": spec.min,
                    "max_allowed": spec.max,
                    "violation_count": count,
                    "sample": sample,
                }
            )

        severity = "error" if any_violations else "ok"
        return severity, {"per_column": report}
