"""Data freshness check.

Flags datasets where the latest timestamp in configured columns is
older than ``freshness_max_age_hours``. Useful for scheduled batch
jobs that should never publish stale snapshots.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from qualipilot.checks.base import Check, CheckContext
from qualipilot.engines.base import as_utc_datetime
from qualipilot.models.results import Severity


class FreshnessCheck(Check):
    name = "freshness"

    def _execute(self, ctx: CheckContext) -> tuple[Severity, dict[str, Any]]:
        cols = ctx.config.freshness_columns or ctx.engine.datetime_columns()
        if not cols:
            return "error", {
                "per_column": [],
                "note": (
                    "freshness is enabled but no datetime columns were "
                    "configured or detected"
                ),
            }

        max_age = timedelta(hours=ctx.config.freshness_max_age_hours)
        future_tolerance = timedelta(
            hours=ctx.config.freshness_future_tolerance_hours
        )
        now = datetime.now(UTC)
        report: list[dict[str, Any]] = []
        has_error = False
        columns = set(
            ctx.columns if ctx.columns is not None else ctx.engine.columns()
        )

        for col in cols:
            if col not in columns:
                report.append(
                    {
                        "column": col,
                        "max_timestamp": None,
                        "is_stale": True,
                        "note": "column not present in dataset",
                    }
                )
                has_error = True
                continue
            max_ts = ctx.engine.max_datetime_instant(
                col,
                ctx.config.freshness_timezone,
            )
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
    return as_utc_datetime(value, naive_timezone)
