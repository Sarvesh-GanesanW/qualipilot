"""Data freshness check.

Flags datasets where the latest timestamp in configured columns is
older than ``freshness_max_age_hours``. Useful for scheduled batch
jobs that should never publish stale snapshots.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from qualipilot.checks.base import Check, CheckContext
from qualipilot.models.results import Severity


class FreshnessCheck(Check):
    name = "freshness"

    def _execute(self, ctx: CheckContext) -> tuple[Severity, dict[str, Any]]:
        cols = ctx.config.freshness_columns or ctx.engine.datetime_columns()
        if not cols:
            return "ok", {"per_column": []}

        max_age = timedelta(hours=ctx.config.freshness_max_age_hours)
        future_tolerance = timedelta(
            hours=ctx.config.freshness_future_tolerance_hours
        )
        now = datetime.now(UTC)
        report: list[dict[str, Any]] = []
        has_error = False

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
                has_error = True
                continue

            ts = _as_aware(max_ts, ctx.config.freshness_timezone)
            age = now - ts
            stale = age > max_age
            is_future = age < -future_tolerance
            if stale or is_future:
                has_error = True
            report.append(
                {
                    "column": col,
                    "max_timestamp": ts.isoformat(),
                    "age_hours": round(age.total_seconds() / 3600, 3),
                    "is_stale": stale,
                    "is_future": is_future,
                }
            )

        severity: Severity = "error" if has_error else "ok"
        return severity, {"per_column": report}


def _as_aware(value: Any, naive_timezone: str = "UTC") -> datetime:
    """Coerce engine-returned timestamps into timezone-aware datetime."""
    # pandas.Timestamp, polars Datetime and stdlib datetime all satisfy
    # the duck-typing we need; pandas ts has .to_pydatetime()
    dt: datetime
    if hasattr(value, "to_pydatetime"):
        dt = value.to_pydatetime()
    elif isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value))

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(naive_timezone))
    return dt
