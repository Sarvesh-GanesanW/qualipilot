"""User-facing configuration models.

Two layers:
    * ``CheckConfig`` / ``LLMConfig`` — declarative, serialisable from
      YAML/JSON so CI pipelines can version-control their checks.
    * ``QualipilotConfig`` — the full runtime config, usable via
      ``pydantic_settings`` so env vars override YAML.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    TypeAdapter,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    EnvSettingsSource,
    SettingsConfigDict,
)

from qualipilot.linking.config import LinkConfig
from qualipilot.models.results import Severity

EngineName = Literal["auto", "polars", "pandas", "duckdb", "dask", "spark"]
LLMProvider = Literal["none", "bedrock", "ollama", "openai", "gz"]
ReportFormat = Literal["json", "html", "markdown"]
BuiltInCheckName = Literal[
    "dataset_contract",
    "missing_values",
    "duplicates",
    "data_types",
    "outliers",
    "ranges",
    "cardinality",
    "freshness",
    "linkage",
]


class _DuplicateConfigKeyError(ValueError):
    pass


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        hide_input_in_errors=True,
    )


class ColumnRange(_StrictModel):
    """Declarative min/max constraint for a numeric column."""

    min: float = Field(allow_inf_nan=False)
    max: float = Field(allow_inf_nan=False)

    @field_validator("max")
    @classmethod
    def _check_bounds(cls, v: float, info: ValidationInfo) -> float:
        min_val = info.data.get("min")
        if min_val is not None and v < min_val:
            raise ValueError("max must be >= min")
        return v


class CheckConfig(_StrictModel):
    """What checks to run and how strict to be.

    Each boolean toggles a check; numeric fields tune behaviour.
    """

    missing_values: bool = True
    duplicates: bool = True
    data_types: bool = True
    outliers: bool = True
    ranges: bool = True
    cardinality: bool = True
    freshness: bool = False
    severity_overrides: dict[BuiltInCheckName, Severity] = Field(
        default_factory=dict
    )

    min_rows: int = Field(default=1, ge=0)
    required_columns: list[str] = Field(default_factory=list)
    expected_dtypes: dict[str, str] = Field(default_factory=dict)
    outlier_iqr_multiplier: float = Field(
        default=1.5, gt=0, allow_inf_nan=False
    )
    duplicate_subset: list[str] | None = None
    column_ranges: dict[str, ColumnRange] = Field(default_factory=dict)
    freshness_columns: list[str] = Field(default_factory=list)
    freshness_max_age_hours: float = Field(
        default=24.0, gt=0, allow_inf_nan=False
    )
    freshness_timezone: str = "UTC"
    freshness_future_tolerance_hours: float = Field(
        default=0.0, ge=0, allow_inf_nan=False
    )
    sample_size: int = Field(default=0, ge=0, le=1000)
    include_top_values: bool = False

    # probabilistic dedup — None disables the LinkageCheck
    linkage: LinkConfig | None = None

    @field_validator(
        "required_columns", "freshness_columns", "duplicate_subset"
    )
    @classmethod
    def _validate_column_names(
        cls, value: list[str] | None
    ) -> list[str] | None:
        if value is None:
            return None
        normalized = [column.strip() for column in value]
        if any(not column for column in normalized):
            raise ValueError("column names must not be empty")
        if len(normalized) != len(
            {column.casefold() for column in normalized}
        ):
            raise ValueError("column names must not contain duplicates")
        return normalized

    @field_validator("expected_dtypes")
    @classmethod
    def _validate_expected_dtypes(
        cls, value: dict[str, str]
    ) -> dict[str, str]:
        normalized: dict[str, str] = {}
        seen: set[str] = set()
        for raw_column, raw_dtype in value.items():
            column = raw_column.strip()
            dtype = raw_dtype.strip()
            if not column or not dtype:
                raise ValueError("expected dtype names must not be empty")
            key = column.casefold()
            if key in seen:
                raise ValueError(
                    "expected dtype columns must not contain duplicates"
                )
            seen.add(key)
            normalized[column] = dtype
        return normalized

    @field_validator("column_ranges")
    @classmethod
    def _validate_range_columns(
        cls,
        value: dict[str, ColumnRange],
    ) -> dict[str, ColumnRange]:
        normalized: dict[str, ColumnRange] = {}
        seen: set[str] = set()
        for raw_column, bounds in value.items():
            column = raw_column.strip()
            if not column:
                raise ValueError("range column names must not be empty")
            key = column.casefold()
            if key in seen:
                raise ValueError("range columns must not contain duplicates")
            seen.add(key)
            normalized[column] = bounds
        return normalized

    @field_validator("freshness_timezone")
    @classmethod
    def _validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown timezone: {value}") from exc
        return value

    @model_validator(mode="after")
    def _validate_embedded_linkage(self) -> CheckConfig:
        if self.linkage is not None and self.linkage.mode != "dedupe":
            raise ValueError("checks.linkage supports dedupe mode only")
        return self


class LLMConfig(_StrictModel):
    """LLM provider settings; provider=none disables summarisation."""

    provider: LLMProvider = "none"
    connection_name: str | None = None
    model: str = ""
    # bedrock
    region: str = "us-east-1"
    aws_profile: str | None = None
    # ollama / openai-compatible
    base_url: str = "http://localhost:11434/v1"
    api_key: SecretStr | None = Field(default=None, repr=False, exclude=True)
    allow_insecure_http: bool = False
    # shared
    max_tokens: int = Field(default=1500, gt=0, le=64_000)
    temperature: float = Field(default=0.2, ge=0, le=2.0, allow_inf_nan=False)
    timeout_seconds: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    retries: int = Field(default=3, ge=0, le=10)
    system_prompt: str = (
        "You are a senior data engineer. Given a data quality summary, "
        "produce a concise markdown report with findings, impact, and "
        "recommended cleanup steps. Be specific; avoid filler."
    )

    @model_validator(mode="before")
    @classmethod
    def _select_connection_provider(cls, value: Any) -> Any:
        if not isinstance(value, Mapping) or "connection_name" not in value:
            return value

        data = dict(value)
        if data.get("connection_name") is None:
            return data
        provider = data.get("provider")
        if provider is None:
            data["provider"] = "gz"
        elif provider != "gz":
            raise ValueError(
                "connection_name cannot be combined with a direct provider"
            )

        mixed_fields = {
            field
            for field, is_mixed in {
                "model": bool(str(data.get("model") or "").strip()),
                "api_key": data.get("api_key") is not None,
                "aws_profile": bool(
                    str(data.get("aws_profile") or "").strip()
                ),
                "base_url": data.get("base_url", "http://localhost:11434/v1")
                != "http://localhost:11434/v1",
                "region": data.get("region", "us-east-1") != "us-east-1",
            }.items()
            if is_mixed
        }
        if mixed_fields:
            fields = ", ".join(sorted(mixed_fields))
            raise ValueError(
                f"connection_name cannot be combined with: {fields}"
            )
        return data

    @field_validator("connection_name")
    @classmethod
    def _validate_connection_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("connection_name must not be blank")
        return value

    @model_validator(mode="after")
    def _validate_provider_settings(self) -> LLMConfig:
        if self.provider == "gz" and self.connection_name is None:
            raise ValueError("gz requires connection_name")
        if self.provider != "gz" and self.connection_name is not None:
            raise ValueError(
                "connection_name cannot be combined with a direct provider"
            )
        if self.provider == "bedrock" and self.temperature > 1.0:
            raise ValueError(
                f"bedrock temperature must be <= 1.0; got {self.temperature}"
            )
        if self.provider not in {"none", "gz"} and not self.model.strip():
            raise ValueError(f"{self.provider} requires an explicit model")
        if self.provider == "bedrock" and not self.region.strip():
            raise ValueError("bedrock requires a region")
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("base_url must be an absolute http(s) URL")
        if (
            parsed.username is not None
            or parsed.password is not None
            or parsed.params
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "base_url must not contain userinfo, parameters, "
                "a query, or a fragment"
            )
        if self.provider in {"ollama", "openai"}:
            is_loopback = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            if (
                parsed.scheme != "https"
                and not is_loopback
                and not self.allow_insecure_http
            ):
                raise ValueError(
                    f"{self.provider} base_url must use https outside "
                    "localhost; "
                    "set allow_insecure_http=true only for a trusted network"
                )
        return self


class QualipilotConfig(BaseSettings):
    """Top-level runtime configuration.

    Values merge in this priority (lowest to highest):
        1. defaults on this model
        2. YAML/JSON file loaded via ``from_file``
        3. environment variables prefixed ``QUALIPILOT_``
        4. kwargs passed explicitly at construction
    """

    model_config = SettingsConfigDict(
        env_prefix="QUALIPILOT_",
        env_nested_delimiter="__",
        extra="forbid",
        allow_inf_nan=False,
        hide_input_in_errors=True,
    )

    engine: EngineName = "auto"
    checks: CheckConfig = Field(default_factory=CheckConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    output_path: Path | None = None
    report_format: ReportFormat = "json"

    @model_validator(mode="after")
    def _validate_engine_features(self) -> QualipilotConfig:
        if self.checks.linkage is not None and self.engine not in {
            "auto",
            "polars",
            "pandas",
        }:
            raise ValueError(
                "checks.linkage requires the polars or pandas engine"
            )
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> QualipilotConfig:
        """Build config from a YAML or JSON file.

        Args:
            path: Filesystem path to the config document.

        Returns:
            A fully-initialised ``QualipilotConfig``.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the file extension is unsupported.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)

        raw = p.read_text(encoding="utf-8")
        try:
            if p.suffix.lower() in {".yml", ".yaml"}:
                data = yaml.load(raw, Loader=_UniqueKeyLoader) or {}
            elif p.suffix.lower() == ".json":
                data = json.loads(
                    raw,
                    object_pairs_hook=_unique_config_mapping,
                )
            else:
                raise ValueError(f"unsupported config extension: {p.suffix}")
        except (
            _DuplicateConfigKeyError,
            json.JSONDecodeError,
            yaml.YAMLError,
        ) as exc:
            raise ValueError(f"invalid config syntax: {exc}") from exc
        if not isinstance(data, Mapping):
            raise ValueError("config root must be a mapping")

        merged = dict(data)
        _deep_update(merged, EnvSettingsSource(cls)())
        return TypeAdapter(cls).validate_python(merged)


def _unique_config_mapping(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateConfigKeyError(f"duplicate config key: {key!r}")
        result[key] = value
    return result


def _construct_unique_yaml_mapping(
    loader: Any,
    node: Any,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    pairs = loader.construct_pairs(node, deep=deep)
    result: dict[object, object] = {}
    for key, value in pairs:
        try:
            duplicate = key in result
        except TypeError as exc:
            raise _DuplicateConfigKeyError(
                "config mapping keys must be scalar"
            ) from exc
        if duplicate:
            raise _DuplicateConfigKeyError(f"duplicate config key: {key!r}")
        result[key] = value
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_yaml_mapping,
)


def _deep_update(
    target: dict[str, object], updates: dict[str, object]
) -> None:
    for key, value in updates.items():
        current = target.get(key)
        if isinstance(value, dict) and isinstance(current, dict):
            _deep_update(current, value)
        else:
            target[key] = value
