"""Serializable report models returned by the checker."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal, TypeAlias, TypeVar, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

Severity = Literal["ok", "warn", "error"]
CheckStatus = Literal["completed", "failed"]
LLMStatus = Literal["disabled", "completed", "failed"]
JSONValue: TypeAlias = JsonValue


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
    nested_normalized_columns: list[str] = Field(default_factory=list)


class BuiltInPayload(_ResultModel):
    """Typed envelope shared by built-in payloads; extensions remain legal."""

    model_config = ConfigDict(extra="allow", allow_inf_nan=False)


class QualityContractPayload(BuiltInPayload):
    schema_version: Literal["1.0"]
    rule_packs: list[str]
    findings: list[
        SchemaContractFinding
        | ComparisonContractFinding
        | ForeignKeyContractFinding
    ]
    violation_count: int = Field(ge=0)


class IcebergDtypeMismatch(_ResultModel):
    column: str
    expected: str
    actual: str | None


class IcebergTypeChange(_ResultModel):
    column: str
    expected: str
    actual: str


class IcebergSchemaPolicy(_ResultModel):
    violation_count: int = Field(ge=0)
    additions: list[str]
    removals: list[str]
    type_changes: list[IcebergTypeChange]


class IcebergFileProbe(_ResultModel):
    performed: bool
    checked: int = Field(ge=0)
    missing: int = Field(ge=0)
    size_mismatches: int = Field(ge=0)
    errors: int = Field(ge=0)
    complete: bool


class IcebergPayload(BuiltInPayload):
    identifier: str
    metadata_complete: bool
    captured_snapshot_id: int | None = None
    current_snapshot_id: int | None = None
    history_count: int | None = Field(default=None, ge=0)
    snapshot_count: int | None = Field(default=None, ge=0)
    snapshot_age_hours: float | None = None
    data_files: int | None = Field(default=None, ge=0)
    delete_files: int | None = Field(default=None, ge=0)
    total_files: int | None = Field(default=None, ge=0)
    active_partition_spec_count: int | None = Field(default=None, ge=0)
    active_spec_ids: list[int] = Field(default_factory=list)
    manifest_count: int | None = Field(default=None, ge=0)
    captured_schema_id: int | None = Field(default=None, ge=0)
    current_spec_id: int | None = Field(default=None, ge=0)
    schema_: dict[str, str] = Field(default_factory=dict, alias="schema")
    dtype_mismatches: list[IcebergDtypeMismatch] = Field(default_factory=list)
    schema_policy: IcebergSchemaPolicy | None = None
    ancestor_ok: bool | None = None
    file_probe: IcebergFileProbe | None = None
    file_probe_note: str | None = None


class DatasetDtypeMismatch(_ResultModel):
    column: str
    expected: str
    actual: str


class DatasetContractPayload(BuiltInPayload):
    row_count: int
    min_rows: int
    missing_required_columns: list[str]
    dtype_mismatches: list[DatasetDtypeMismatch]


class DuplicatesPayload(BuiltInPayload):
    total_duplicate_rows: int
    subset: list[str] | None = None
    sample: list[dict[str, JSONValue]] = Field(default_factory=list)


class ContractTypeChange(_ResultModel):
    column: str
    expected: str
    actual: str


class SchemaContractFinding(_ResultModel):
    pack: str
    rule: Literal["schema"]
    kind: Literal["schema"]
    violation_count: int = Field(ge=0)
    additions: list[str]
    removals: list[str]
    type_changes: list[ContractTypeChange]


class ComparisonContractFinding(_ResultModel):
    pack: str
    rule: str
    kind: Literal["comparison"]
    violation_count: int = Field(ge=0)
    missing_columns: list[str]
    nulls_pass: bool


class ForeignKeyContractFinding(_ResultModel):
    pack: str
    rule: str
    kind: Literal["foreign_key"]
    violation_count: int = Field(ge=0)
    reference: str
    missing_reference: bool
    nulls_pass: bool


class MissingColumnFinding(BuiltInPayload):
    column: str
    null_count: int = Field(ge=0)
    null_percentage: float


class MissingValuesPayload(BuiltInPayload):
    total_null_count: int = Field(ge=0)
    worst_column_pct: float
    per_column: list[MissingColumnFinding]


class DataTypesPayload(BuiltInPayload):
    per_column: dict[str, str]
    rollup: dict[str, int] = Field(default_factory=dict)


class CardinalityFinding(BuiltInPayload):
    column: str
    distinct_count: int = Field(ge=0)
    unique_ratio: float
    top_values: list[tuple[str, int]] = Field(default_factory=list)


class CardinalityPayload(BuiltInPayload):
    per_column: list[CardinalityFinding]


class RangeFinding(BuiltInPayload):
    column: str
    min_allowed: float
    max_allowed: float
    violation_count: int = Field(ge=0)
    sample: list[dict[str, JSONValue]]
    note: str | None = None


class RangesPayload(BuiltInPayload):
    per_column: list[RangeFinding]


class OutlierFinding(BuiltInPayload):
    column: str
    lower_bound: float
    upper_bound: float
    outlier_count: int = Field(ge=0)
    sample: list[dict[str, JSONValue]]


class OutlierSkipped(BuiltInPayload):
    column: str
    skipped: str


class QuantileProvenance(BuiltInPayload):
    method: Literal["exact", "approximate"]
    relative_error: float | None = None


class OutliersPayload(BuiltInPayload):
    per_column: list[OutlierFinding | OutlierSkipped]
    quantile_provenance: QuantileProvenance


class FreshnessObserved(BuiltInPayload):
    column: str
    max_timestamp: str
    age_hours: float
    is_stale: bool
    is_future: bool


class FreshnessUnavailable(BuiltInPayload):
    column: str
    max_timestamp: None = None
    is_stale: Literal[True]
    note: str


class FreshnessPayload(BuiltInPayload):
    per_column: list[FreshnessObserved | FreshnessUnavailable]
    note: str | None = None


class LinkageSkippedPayload(BuiltInPayload):
    """Stable payload when linkage is intentionally disabled."""

    skipped: Literal[True]


class LinkageResultPayload(BuiltInPayload):
    """Concrete result from a fitted linkage run."""

    candidate_pairs: int
    matched_pairs: int
    match_threshold_probability: float
    clusters: int
    timings_ms: dict[str, float]
    lambda_: float = Field(alias="lambda")
    fit_status: str
    fit_reason: str | None
    fit_warnings: list[str]
    duplicate_clusters: int
    records_in_duplicate_groups: int


class FailedPayload(_ResultModel):
    """The stable empty payload emitted for a failed check."""


class GenericPayload(BuiltInPayload):
    """Forward-compatible typed view for application extension payloads."""


TypedPayload: TypeAlias = (
    DatasetContractPayload
    | MissingValuesPayload
    | DuplicatesPayload
    | DataTypesPayload
    | OutliersPayload
    | RangesPayload
    | CardinalityPayload
    | FreshnessPayload
    | LinkageSkippedPayload
    | LinkageResultPayload
    | QualityContractPayload
    | IcebergPayload
    | FailedPayload
    | GenericPayload
)


PayloadModel = TypeVar("PayloadModel", bound=BuiltInPayload)


class Metric(_ResultModel):
    """One stable, source-redacted observability sample."""

    name: str
    value: float
    labels: dict[str, str] = Field(default_factory=dict)


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

    def typed_payload(self) -> TypedPayload:
        """Return a stable typed view for built-in payloads and extensions."""
        if self.status == "failed":
            return FailedPayload.model_validate(self.payload)
        if self.name == "linkage":
            model = (
                LinkageSkippedPayload
                if self.payload.get("skipped") is True
                else LinkageResultPayload
            )
            return model.model_validate(self.payload)
        fallback_model = _PAYLOAD_MODELS.get(self.name)
        if fallback_model is None:
            return GenericPayload.model_validate(self.payload)
        return cast(TypedPayload, fallback_model.model_validate(self.payload))

    def payload_as(self, model: type[PayloadModel]) -> PayloadModel:
        """Validate this wire-compatible payload as a caller-selected model."""
        return model.model_validate(self.payload)

    @model_validator(mode="after")
    def _validate_status(self) -> CheckResult:
        if self.status == "failed" and not self.error:
            raise ValueError("failed checks require an error")
        if self.status == "completed" and self.error:
            raise ValueError("completed checks cannot carry an error")
        return self


_PAYLOAD_MODELS: dict[str, type[BuiltInPayload]] = {
    "dataset_contract": DatasetContractPayload,
    "missing_values": MissingValuesPayload,
    "duplicates": DuplicatesPayload,
    "data_types": DataTypesPayload,
    "outliers": OutliersPayload,
    "ranges": RangesPayload,
    "cardinality": CardinalityPayload,
    "freshness": FreshnessPayload,
    "quality_contract": QualityContractPayload,
    "iceberg_table": IcebergPayload,
}


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

    def iter_metrics(self) -> Iterator[Metric]:
        """Yield finite metrics without source names, paths, or values."""
        levels = {"ok": 0.0, "warn": 1.0, "error": 2.0}
        for result in self.results:
            labels = {"check": result.name, "status": result.status}
            yield Metric(
                name="qualipilot_check_severity",
                value=levels[result.severity],
                labels=labels,
            )
            yield Metric(
                name="qualipilot_check_duration_seconds",
                value=result.duration_seconds,
                labels=labels,
            )

    def prometheus_text(self) -> str:
        """Render the stable metrics stream in Prometheus text format."""
        lines = [
            "# TYPE qualipilot_check_severity gauge",
            "# TYPE qualipilot_check_duration_seconds gauge",
        ]
        for metric in self.iter_metrics():
            labels = ",".join(
                f'{key}="{_prometheus_label(value)}"'
                for key, value in sorted(metric.labels.items())
            )
            lines.append(f"{metric.name}{{{labels}}} {metric.value}")
        return "\n".join(lines) + "\n"

    def emit_metrics(self, callback: Callable[[Metric], None]) -> None:
        """Send metrics to a caller-owned logging or telemetry adapter."""
        for metric in self.iter_metrics():
            callback(metric)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value


def _prometheus_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
