"""Range validation check.

Given ``column_ranges`` in the config, counts and samples rows whose
values fall outside the allowed ``[min, max]`` envelope.
"""

from __future__ import annotations

from typing import Any

from qualipilot.checks.base import Check, CheckContext
from qualipilot.models.results import Severity


class RangesCheck(Check):
    name = "ranges"

    def _execute(self, ctx: CheckContext) -> tuple[Severity, dict[str, Any]]:
        ranges = ctx.config.column_ranges
        if not ranges:
            return "ok", {"per_column": []}

        columns = (
            ctx.columns if ctx.columns is not None else ctx.engine.columns()
        )
        existing = set(columns)
        numeric = set(ctx.engine.numeric_columns())
        report: list[dict[str, Any]] = []
        any_violations = False
        any_misapplied = False

        for col, spec in ranges.items():
            if col not in existing:
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
                any_misapplied = True
                continue

            if col not in numeric:
                # range constraint on a non-numeric column is almost
                # always a config typo. flag explicitly so it does not
                # hide as a silent ok.
                report.append(
                    {
                        "column": col,
                        "min_allowed": spec.min,
                        "max_allowed": spec.max,
                        "violation_count": 0,
                        "sample": [],
                        "note": (
                            "column dtype is non-numeric; range constraint "
                            "cannot be applied"
                        ),
                    }
                )
                any_misapplied = True
                continue

            count = ctx.engine.count_outside(col, spec.min, spec.max)
            sample = (
                ctx.engine.sample_outside(
                    col, spec.min, spec.max, ctx.config.sample_size
                )
                if count and ctx.config.sample_size
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

        severity: Severity = (
            "error" if any_violations or any_misapplied else "ok"
        )
        return severity, {"per_column": report}
