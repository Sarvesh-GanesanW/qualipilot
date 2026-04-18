"""Typed result models returned from every check.

These models are the public contract: serialisable to JSON, safe to ship
across process boundaries (Lambda, message queues), and stable enough
for downstream dashboards to rely on.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

Severity = Literal["ok", "warn", "error"]


class ColumnNullStat(BaseModel):
    column: str
    null_count: int
    null_percentage: float


class DuplicateInfo(BaseModel):
    total_duplicate_rows: int
    subset: list[str] | None = None
    sample: list[dict[str, Any]] = Field(default_factory=list)


class OutlierInfo(BaseModel):
    column: str
    lower_bound: float
    upper_bound: float
    outlier_count: int
    sample: list[dict[str, Any]] = Field(default_factory=list)


class RangeViolationInfo(BaseModel):
    column: str
    min_allowed: float
    max_allowed: float
    violation_count: int
    sample: list[dict[str, Any]] = Field(default_factory=list)


class CardinalityInfo(BaseModel):
    column: str
    distinct_count: int
    top_values: list[tuple[str, int]] = Field(default_factory=list)


class FreshnessInfo(BaseModel):
    column: str
    max_timestamp: datetime | None
    max_age_hours: float
    is_stale: bool


class DatasetStats(BaseModel):
    row_count: int
    column_count: int
    columns: list[str]
    dtypes: dict[str, str]
    engine: str


class CheckResult(BaseModel):
    """Outcome of a single check.

    ``payload`` carries the typed per-check info (one of the *Info
    models above). It is kept as ``dict`` here so adding new checks
    does not require expanding this class.
    """

    name: str
    severity: Severity
    duration_seconds: float
    payload: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class QualityReport(BaseModel):
    """Aggregate result of a full ``DataQualityChecker.run()`` call."""

    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    dataset: DatasetStats
    results: list[CheckResult]
    llm_report: str | None = None
    config_hash: str | None = None

    def to_json(self, *, indent: int = 2) -> str:
        """Render the report as a JSON string."""
        return self.model_dump_json(indent=indent)

    def failed_checks(self) -> list[CheckResult]:
        """Checks that hit the ``error`` severity."""
        return [r for r in self.results if r.severity == "error"]

    def warning_checks(self) -> list[CheckResult]:
        """Checks that surfaced warnings but did not fail."""
        return [r for r in self.results if r.severity == "warn"]
