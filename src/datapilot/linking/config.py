"""Configuration for ``RecordLinker``.

Designed to serialise cleanly to YAML so the whole model spec lives
in version control next to the data pipeline.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from datapilot.linking.comparisons import ComparisonSpec

Mode = Literal["dedupe", "link"]
Backend = Literal["polars", "duckdb"]


class LinkConfig(BaseModel):
    """Record-linkage model specification."""

    model_config = ConfigDict(validate_default=True)

    backend: Backend = "polars"

    mode: Mode = "dedupe"
    unique_id_column: str

    comparisons: list[ComparisonSpec] = Field(default_factory=list)
    # each rule is a list of columns whose values must all agree; the
    # candidate set is the union across rules. cartesian = empty list.
    blocking_rules: list[list[str]] = Field(default_factory=list)

    prior_match_probability: float = Field(
        default=0.001,
        gt=0,
        lt=1,
        description="lambda seed; small because most pairs are non-matches",
    )
    match_threshold_probability: float = Field(default=0.9, ge=0.5, lt=1.0)

    em_max_iter: int = Field(default=15, ge=1, le=200)
    em_tolerance: float = Field(default=1e-3, gt=0)

    # when the blocking output is huge, learning m/u on every pair is
    # wasteful — we fit EM on a random sample then score all pairs
    em_sample_size: int = Field(default=500_000, ge=10_000)

    max_pairs_warning: int = Field(default=5_000_000, gt=0)
    # hard cap — linker raises rather than oomkill the process when
    # blocking produces more pairs than this
    max_pairs_hard_cap: int = Field(default=50_000_000, gt=0)

    @field_validator("comparisons")
    @classmethod
    def _require_comparisons(
        cls, v: list[ComparisonSpec]
    ) -> list[ComparisonSpec]:
        if not v:
            raise ValueError(
                "at least one comparison is required to score pairs"
            )
        return v
