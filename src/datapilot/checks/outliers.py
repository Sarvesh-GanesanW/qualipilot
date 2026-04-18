"""Outlier check using the IQR rule.

Computes Q1 and Q3 for every numeric column in a single pass via the
engine's batched ``quantiles`` API, then counts/samples values outside
``[Q1 - k*IQR, Q3 + k*IQR]``.
"""

from __future__ import annotations

import math
from typing import Any

from datapilot.checks.base import Check, CheckContext


class OutliersCheck(Check):
    name = "outliers"

    def _execute(self, ctx: CheckContext) -> tuple[str, dict[str, Any]]:
        numeric = ctx.engine.numeric_columns()
        if not numeric:
            return "ok", {"per_column": []}

        qmap = ctx.engine.quantiles(numeric, qs=(0.25, 0.75))
        k = ctx.config.outlier_iqr_multiplier

        report: list[dict[str, Any]] = []
        any_outliers = False
        for col in numeric:
            q1 = qmap[col][0.25]
            q3 = qmap[col][0.75]
            if _is_nan(q1) or _is_nan(q3):
                # constant or empty column, nothing to flag
                continue
            iqr = q3 - q1
            low = q1 - k * iqr
            high = q3 + k * iqr
            count = ctx.engine.count_outside(col, low, high)
            sample = (
                ctx.engine.sample_outside(
                    col, low, high, ctx.config.sample_size
                )
                if count
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

        severity = "warn" if any_outliers else "ok"
        return severity, {"per_column": report}


def _is_nan(value: float) -> bool:
    # protects against None-like sentinels alongside real nan floats
    try:
        return math.isnan(value)
    except TypeError:
        return False
