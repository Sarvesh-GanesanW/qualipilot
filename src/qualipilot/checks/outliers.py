"""Outlier check using the IQR rule.

Computes Q1 and Q3 for every numeric column in a single pass via the
engine's batched ``quantiles`` API, then counts/samples values outside
``[Q1 - k*IQR, Q3 + k*IQR]``.
"""

from __future__ import annotations

import math
from typing import Any

from qualipilot.checks.base import Check, CheckContext
from qualipilot.models.results import Severity


class OutliersCheck(Check):
    name = "outliers"

    def _execute(self, ctx: CheckContext) -> tuple[Severity, dict[str, Any]]:
        numeric = ctx.engine.numeric_columns()
        if not numeric:
            return "ok", {
                "per_column": [],
                "quantile_provenance": ctx.engine.quantile_provenance,
            }

        qmap = ctx.engine.quantiles(numeric, qs=(0.25, 0.75))
        k = ctx.config.outlier_iqr_multiplier

        bounds: dict[str, tuple[float, float]] = {}
        skipped: set[str] = set()
        for col in numeric:
            q1 = qmap[col][0.25]
            q3 = qmap[col][0.75]
            if _is_nan(q1) or _is_nan(q3):
                continue
            iqr = q3 - q1
            low = q1 - k * iqr
            high = q3 + k * iqr
            if not all(map(math.isfinite, (q1, q3, low, high))):
                skipped.add(col)
                continue
            bounds[col] = (low, high)
        counts = ctx.engine.counts_outside(bounds)

        report: list[dict[str, Any]] = []
        any_outliers = False
        for col in numeric:
            if col in skipped:
                any_outliers = True
                report.append(
                    {
                        "column": col,
                        "skipped": "non-finite quantile bounds",
                    }
                )
                continue
            if col not in bounds:
                continue
            low, high = bounds[col]
            count = counts[col]
            sample = (
                ctx.engine.sample_outside(
                    col, low, high, ctx.config.sample_size
                )
                if count and ctx.config.sample_size
                else []
            )
            if count:
                any_outliers = True
            report.append(
                {
                    "column": col,
                    "lower_bound": low,
                    "upper_bound": high,
                    "outlier_count": count,
                    "sample": sample,
                }
            )

        severity: Severity = "warn" if any_outliers else "ok"
        return severity, {
            "per_column": report,
            "quantile_provenance": ctx.engine.quantile_provenance,
        }


def _is_nan(value: float) -> bool:
    # protects against None-like sentinels alongside real nan floats
    try:
        return math.isnan(value)
    except TypeError:
        return False
