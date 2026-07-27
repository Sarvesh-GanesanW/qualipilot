"""Probabilistic record linkage and entity resolution.

Candidate blocking uses Polars or DuckDB, parameter estimation uses
NumPy, and matched pairs are clustered with union-find.
"""

from qualipilot.linking.comparisons import (
    ComparisonSpec,
    ExactMatch,
    FuzzyString,
    NumericDiff,
)
from qualipilot.linking.config import LinkConfig, StringNormalization
from qualipilot.linking.consolidate import (
    ConsolidationAudit,
    ConsolidationConfig,
    ConsolidationResult,
    MergeRule,
    SurvivorSortKey,
    consolidate_records,
)
from qualipilot.linking.linker import (
    DeduplicationResult,
    LinkageResult,
    RecordLinker,
    normalize_records,
)

__all__ = [
    "ComparisonSpec",
    "ConsolidationAudit",
    "ConsolidationConfig",
    "ConsolidationResult",
    "DeduplicationResult",
    "ExactMatch",
    "FuzzyString",
    "LinkConfig",
    "LinkageResult",
    "MergeRule",
    "NumericDiff",
    "RecordLinker",
    "StringNormalization",
    "SurvivorSortKey",
    "consolidate_records",
    "normalize_records",
]
