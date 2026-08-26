"""Serializable report models returned by the checker."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

Severity = Literal["ok", "warn", "error"]
CheckStatus = Literal["completed", "failed"]
LLMStatus = Literal["disabled", "completed", "failed"]


def _package_version() -> str:
    try:
        return version("qualipilot")
    except PackageNotFoundError:  # pragma: no cover - source-only imports
        return "unknown"


class _ResultModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        ser_json_bytes="base64",
    )


class DatasetStats(_ResultModel):
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    columns: list[str]
    dtypes: dict[str, str]
    engine: str
    source: str | None = None
    source_version: str | None = None


class CheckResult(_ResultModel):
    """Outcome of a single check.

    ``payload`` is JSON-normalized at construction so extension checks
    cannot break report serialization.
    """

    name: str
    severity: Severity
    status: CheckStatus = "completed"
    duration_seconds: float = Field(ge=0)
    payload: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    @field_validator("payload", mode="before")
    @classmethod
    def _make_payload_json_safe(cls, value: Any) -> dict[str, Any]:
        normalized = json.loads(
            json.dumps(_json_safe(value), default=str, allow_nan=False)
        )
        if not isinstance(normalized, dict):
            raise ValueError("payload must be a JSON object")
        return cast(dict[str, Any], normalized)

    @model_validator(mode="after")
    def _validate_status(self) -> CheckResult:
        if self.status == "failed" and not self.error:
            raise ValueError("failed checks require an error")
        if self.status == "completed" and self.error:
            raise ValueError("completed checks cannot carry an error")
        return self


class QualityReport(_ResultModel):
    """Aggregate result of a full ``DataQualityChecker.run()`` call."""

    schema_version: Literal["1.0"] = "1.0"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    package_version: str = Field(default_factory=_package_version)
    dataset: DatasetStats
    results: list[CheckResult]
    llm_report: str | None = None
    llm_status: LLMStatus = "disabled"
    llm_error: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    config_hash: str | None = None

    def to_json(self, *, indent: int = 2) -> str:
        """Render the report as a JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> QualityReport:
        """Load a compatible report while ignoring newer optional fields."""
        data = json.loads(payload)
        if isinstance(data, dict):
            data = {
                key: value
                for key, value in data.items()
                if key in cls.model_fields
            }
            dataset = data.get("dataset")
            if isinstance(dataset, dict):
                data["dataset"] = {
                    key: value
                    for key, value in dataset.items()
                    if key in DatasetStats.model_fields
                }
            results = data.get("results")
            if isinstance(results, list):
                data["results"] = [
                    {
                        key: value
                        for key, value in result.items()
                        if key in CheckResult.model_fields
                    }
                    if isinstance(result, dict)
                    else result
                    for result in results
                ]
        return cls.model_validate(data)

    def failed_checks(self) -> list[CheckResult]:
        """Checks that hit the ``error`` severity."""
        return [r for r in self.results if r.severity == "error"]

    def warning_checks(self) -> list[CheckResult]:
        """Checks that surfaced warnings but did not fail."""
        return [r for r in self.results if r.severity == "warn"]


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value
