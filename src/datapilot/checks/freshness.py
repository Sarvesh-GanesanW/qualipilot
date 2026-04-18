"""Data freshness check.

Flags datasets where the latest timestamp in configured columns is
older than ``freshness_max_age_hours``. Useful for scheduled batch
jobs that should never publish stale snapshots.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from datapilot.checks.base import Check, CheckContext


class FreshnessCheck(Check):
    name = "freshness"

    def _execute(
        self, ctx: CheckContext
    ) -> tuple[str, dict[str, Any]]:
        cols = ctx.config.freshness_columns or ctx.engine.datetime_columns()
        if not cols:
            return "ok", {"per_column": []}

        max_age = timedelta(hours=ctx.config.freshness_max_age_hours)
        now = datetime.now(UTC)
        report: list[dict[str, Any]] = []
        any_stale = False

        for col in cols:
            max_ts = ctx.engine.max_datetime(col)
            if max_ts is None:
                report.append(
                    {
                        "column": col,
                        "max_timestamp": None,
                        "is_stale": True,
                        "note": "no non-null values",
                    }
                )
                any_stale = True
                continue

            # normalise naive timestamps so subtraction is safe
            ts = _as_aware(max_ts)
            age = now - ts
            stale = age > max_age
            if stale:
                any_stale = True
            report.append(
                {
                    "column": col,
                    "max_timestamp": ts.isoformat(),
                    "age_hours": round(
                        age.total_seconds() / 3600, 3
                    ),
                    "is_stale": stale,
                }
            )

        severity = "error" if any_stale else "ok"
        return severity, {"per_column": report}


def _as_aware(value: Any) -> datetime:
    """Coerce engine-returned timestamps into timezone-aware datetime."""
    # pandas.Timestamp, polars Datetime and stdlib datetime all satisfy
    # the duck-typing we need; pandas ts has .to_pydatetime()
    dt: datetime
    if hasattr(value, "to_pydatetime"):
        dt = value.to_pydatetime()
    elif isinstance(value, datetime):
        dt = value
    else:
        # last-resort coerce via iso parsing for exotic types
        dt = datetime.fromisoformat(str(value))

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt
