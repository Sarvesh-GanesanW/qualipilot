"""Configuration for ``RecordLinker``.

Designed to serialise cleanly to YAML so the whole model spec lives
in version control next to the data pipeline.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from qualipilot.linking.comparisons import ComparisonSpec

Mode = Literal["dedupe", "link"]
Backend = Literal["polars", "duckdb"]
UnicodeForm = Literal["NFC", "NFKC", "NFD", "NFKD"]


class StringNormalization(BaseModel):
    """String cleanup applied before matching and consolidation."""

    model_config = ConfigDict(validate_default=True, extra="forbid")

    unicode_form: UnicodeForm | None = "NFKC"
    trim: bool = True
    collapse_whitespace: bool = True
    lowercase: bool = True
    null_tokens: tuple[str, ...] = ("",)
    regex_replacements: tuple[tuple[str, str], ...] = ()

    @field_validator("null_tokens")
    @classmethod
    def _require_unique_null_tokens(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        if len(set(values)) != len(values):
            raise ValueError("null_tokens must be unique")
        return values

    @field_validator("regex_replacements")
    @classmethod
    def _require_regex_patterns(
        cls,
        replacements: tuple[tuple[str, str], ...],
    ) -> tuple[tuple[str, str], ...]:
        if any(not pattern for pattern, _ in replacements):
            raise ValueError("normalization regex patterns must not be empty")
        return replacements


class LinkConfig(BaseModel):
    """Record-linkage model specification."""

    model_config = ConfigDict(validate_default=True, extra="forbid")

    backend: Backend = "polars"

    mode: Mode = "dedupe"
    unique_id_column: str = Field(min_length=1)

    comparisons: list[ComparisonSpec] = Field(default_factory=list)
    # each rule is a list of columns whose values must all agree; the
    # candidate set is the union across rules. cartesian = empty list.
    blocking_rules: list[list[str]] = Field(default_factory=list)
    normalization: dict[str, StringNormalization] = Field(default_factory=dict)

    prior_match_probability: float = Field(
        default=0.001,
        gt=0,
        lt=1,
        description="lambda seed; small because most pairs are non-matches",
    )
    match_threshold_probability: float = Field(default=0.9, ge=0.5, lt=1.0)

    em_max_iter: int = Field(default=100, ge=1, le=200)
    em_tolerance: float = Field(default=1e-3, gt=0)
    allow_warning_fit: bool = False

    # when the blocking output is huge, learning m/u on every pair is
    # wasteful — we fit EM on a random sample then score all pairs
    em_sample_size: int = Field(default=500_000, ge=10_000)

    # unblocked linkage stops at this budget; blocked linkage logs when
    # its estimated upper bound exceeds it
    max_pairs_warning: int = Field(default=5_000_000, gt=0)
    # hard cap — linker raises rather than oomkill the process when
    # blocking produces more pairs than this
    max_pairs_hard_cap: int = Field(default=50_000_000, gt=0)

    em_random_seed: int = Field(
        default=0,
        ge=0,
        description=(
            "seed for the RNG that subsamples pairs into EM. 0 keeps the "
            "historical default; bump it when running comparative trials "
            "where deterministic-but-different sampling matters."
        ),
    )
    duckdb_threads: int | None = Field(default=None, ge=1)

    @field_validator("unique_id_column")
    @classmethod
    def _strip_id_column(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("unique_id_column must not be blank")
        return value

    @field_validator("comparisons")
    @classmethod
    def _require_comparisons(
        cls, v: list[ComparisonSpec]
    ) -> list[ComparisonSpec]:
        if not v:
            raise ValueError(
                "at least one comparison is required to score pairs"
            )
        columns = [comparison.column for comparison in v]
        if len(set(columns)) != len(columns):
            raise ValueError("comparison columns must be unique")
        return v

    @field_validator("blocking_rules")
    @classmethod
    def _validate_blocking_rules(
        cls,
        rules: list[list[str]],
    ) -> list[list[str]]:
        normalised: list[tuple[str, ...]] = []
        seen: set[frozenset[str]] = set()
        for rule in rules:
            columns = [column.strip() for column in rule]
            if not columns or any(not column for column in columns):
                raise ValueError("blocking rules must not be empty")
            if len(set(columns)) != len(columns):
                raise ValueError(
                    "columns within a blocking rule must be unique"
                )
            key = frozenset(columns)
            if key in seen:
                raise ValueError(
                    "blocking rules must be unique regardless of column order"
                )
            if any(key < other or other < key for other in seen):
                raise ValueError(
                    "blocking rules must not contain redundant supersets"
                )
            seen.add(key)
            normalised.append(tuple(sorted(columns)))
        return [list(rule) for rule in sorted(normalised)]

    @field_validator("normalization")
    @classmethod
    def _validate_normalization_columns(
        cls,
        value: dict[str, StringNormalization],
    ) -> dict[str, StringNormalization]:
        normalized: dict[str, StringNormalization] = {}
        for raw_column, options in value.items():
            column = raw_column.strip()
            if not column:
                raise ValueError(
                    "normalization column names must not be blank"
                )
            if column in normalized:
                raise ValueError(
                    "normalization columns must not contain duplicates"
                )
            normalized[column] = options
        return normalized

    @field_validator("em_tolerance")
    @classmethod
    def _finite_tolerance(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("em_tolerance must be finite")
        return value

    @model_validator(mode="after")
    def _validate_model(self) -> LinkConfig:
        comparison_columns = {
            comparison.column for comparison in self.comparisons
        }
        if self.unique_id_column in comparison_columns:
            raise ValueError("unique_id_column cannot also be a comparison")
        if any(self.unique_id_column in rule for rule in self.blocking_rules):
            raise ValueError("unique_id_column cannot be used for blocking")
        normalization_columns = set(self.normalization)
        if self.unique_id_column in normalization_columns:
            raise ValueError(
                "unique_id_column cannot be normalized; identifiers must "
                "remain unchanged"
            )
        if self.max_pairs_warning > self.max_pairs_hard_cap:
            raise ValueError("max_pairs_warning must be <= max_pairs_hard_cap")
        return self
