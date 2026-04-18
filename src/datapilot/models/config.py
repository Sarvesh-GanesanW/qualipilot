"""User-facing configuration models.

Two layers:
    * ``CheckConfig`` / ``LLMConfig`` — declarative, serialisable from
      YAML/JSON so CI pipelines can version-control their checks.
    * ``DatapilotConfig`` — the full runtime config, usable via
      ``pydantic_settings`` so env vars override YAML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

EngineName = Literal["auto", "polars", "pandas", "dask", "cudf"]
LLMProvider = Literal["none", "bedrock", "ollama", "openai"]
ReportFormat = Literal["json", "html", "markdown"]


class ColumnRange(BaseModel):
    """Declarative min/max constraint for a numeric column."""

    min: float
    max: float

    @field_validator("max")
    @classmethod
    def _check_bounds(cls, v: float, info) -> float:  # type: ignore[no-untyped-def]
        # keeps yaml typos from silently accepting min > max
        min_val = info.data.get("min")
        if min_val is not None and v < min_val:
            raise ValueError("max must be >= min")
        return v


class CheckConfig(BaseModel):
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

    outlier_iqr_multiplier: float = Field(default=1.5, gt=0)
    duplicate_subset: list[str] | None = None
    column_ranges: dict[str, ColumnRange] = Field(default_factory=dict)
    freshness_columns: list[str] = Field(default_factory=list)
    freshness_max_age_hours: float = Field(default=24.0, gt=0)
    sample_size: int = Field(default=10, ge=0, le=1000)


class LLMConfig(BaseModel):
    """LLM provider settings; provider=none disables summarisation."""

    provider: LLMProvider = "none"
    model: str = ""
    # bedrock
    region: str = "us-east-1"
    aws_profile: str | None = None
    # ollama / openai-compatible
    base_url: str = "http://localhost:11434/v1"
    api_key: str | None = None
    # shared
    max_tokens: int = Field(default=1500, gt=0, le=64_000)
    temperature: float = Field(default=0.2, ge=0, le=2.0)
    timeout_seconds: float = Field(default=60.0, gt=0)
    retries: int = Field(default=3, ge=0, le=10)
    system_prompt: str = (
        "You are a senior data engineer. Given a data quality summary, "
        "produce a concise markdown report with findings, impact, and "
        "recommended cleanup steps. Be specific; avoid filler."
    )


class DatapilotConfig(BaseSettings):
    """Top-level runtime configuration.

    Values merge in this priority (lowest to highest):
        1. defaults on this model
        2. YAML/JSON file loaded via ``from_file``
        3. environment variables prefixed ``DATAPILOT_``
        4. kwargs passed explicitly at construction
    """

    model_config = SettingsConfigDict(
        env_prefix="DATAPILOT_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    engine: EngineName = "auto"
    checks: CheckConfig = Field(default_factory=CheckConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    output_path: Path | None = None
    report_format: ReportFormat = "json"
    log_level: str = "INFO"
    json_logs: bool = False

    @classmethod
    def from_file(cls, path: str | Path) -> DatapilotConfig:
        """Build config from a YAML or JSON file.

        Args:
            path: Filesystem path to the config document.

        Returns:
            A fully-initialised ``DatapilotConfig``.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the file extension is unsupported.
        """
        import json

        import yaml

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)

        raw = p.read_text(encoding="utf-8")
        # we accept yaml OR json because ops teams mix the two
        if p.suffix.lower() in {".yml", ".yaml"}:
            data = yaml.safe_load(raw) or {}
        elif p.suffix.lower() == ".json":
            data = json.loads(raw)
        else:
            raise ValueError(f"unsupported config extension: {p.suffix}")
        return cls(**data)
