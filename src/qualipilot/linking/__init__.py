"""Probabilistic record linkage / entity resolution.

In-house Fellegi-Sunter implementation, tuned for speed:
    * blocking via polars hash joins (never materialise N^2)
    * comparison levels assigned with numpy vector ops
    * EM estimated in pure numpy, no per-pair Python loops
    * connected-component clustering via numpy union-find
"""

from qualipilot.linking.comparisons import (
    ComparisonSpec,
    ExactMatch,
    FuzzyString,
    NumericDiff,
)
from qualipilot.linking.config import LinkConfig
from qualipilot.linking.linker import LinkageResult, RecordLinker

__all__ = [
    "ComparisonSpec",
    "ExactMatch",
    "FuzzyString",
    "LinkConfig",
    "LinkageResult",
    "NumericDiff",
    "RecordLinker",
]
