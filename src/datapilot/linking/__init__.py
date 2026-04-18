"""Probabilistic record linkage / entity resolution.

In-house Fellegi-Sunter implementation, tuned for speed:
    * blocking via polars hash joins (never materialise N^2)
    * comparison levels assigned with numpy vector ops
    * EM estimated in pure numpy, no per-pair Python loops
    * connected-component clustering via numpy union-find
"""

from datapilot.linking.comparisons import (
    ComparisonSpec,
    ExactMatch,
    FuzzyString,
    NumericDiff,
)
from datapilot.linking.config import LinkConfig
from datapilot.linking.linker import LinkageResult, RecordLinker

__all__ = [
    "ComparisonSpec",
    "ExactMatch",
    "FuzzyString",
    "LinkConfig",
    "LinkageResult",
    "NumericDiff",
    "RecordLinker",
]
