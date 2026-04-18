"""Comparison primitives.

A *comparison* inspects one column of a candidate pair and returns an
integer **level**. Level 0 is always the "null / no signal" case.
Higher levels mean stronger agreement; the top level is exact match.

Levels are kept small (typically 2-4) so the Fellegi-Sunter EM has
enough data to estimate each level's m/u probability reliably.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl
from pydantic import BaseModel, Field


class _BaseComparison(BaseModel):
    column: str

    # each concrete subclass sets this
    kind: str
    levels: int

    def left_col(self) -> str:
        return f"{self.column}_l"

    def right_col(self) -> str:
        return f"{self.column}_r"


class ExactMatch(_BaseComparison):
    """Three levels: null (0), different (1), exact match (2)."""

    kind: Literal["exact"] = "exact"
    levels: int = 3

    def assign_levels(self, pairs: pl.DataFrame) -> np.ndarray:
        left = pairs[self.left_col()]
        right = pairs[self.right_col()]
        # null on either side -> level 0
        null_mask = (left.is_null() | right.is_null()).to_numpy()
        equal_mask = (left == right).to_numpy()
        # start everyone at 1 (different), promote exact matches to 2,
        # demote null rows to 0
        levels = np.ones(len(pairs), dtype=np.uint8)
        levels[equal_mask] = 2
        levels[null_mask] = 0
        return levels


class NumericDiff(_BaseComparison):
    """Bucket |a-b| into thresholds.

    Example: thresholds=(1.0, 5.0) -> levels are
        0 null
        1 diff > 5.0
        2 diff <= 5.0
        3 diff <= 1.0
    """

    kind: Literal["numeric"] = "numeric"
    thresholds: tuple[float, ...] = Field(default=(1.0, 5.0))
    levels: int = 0  # filled in validator below

    def model_post_init(self, __context: object) -> None:
        # levels = null + "far" + one per threshold
        object.__setattr__(self, "levels", 2 + len(self.thresholds))

    def assign_levels(self, pairs: pl.DataFrame) -> np.ndarray:
        left = pairs[self.left_col()].cast(pl.Float64).to_numpy()
        right = pairs[self.right_col()].cast(pl.Float64).to_numpy()
        diff = np.abs(left - right)
        null_mask = np.isnan(left) | np.isnan(right)

        # base level 1 = "far" (diff exceeds the largest threshold)
        levels = np.ones(len(pairs), dtype=np.uint8)
        # iterate largest-to-smallest so tighter buckets win
        for rank, t in enumerate(
            sorted(self.thresholds, reverse=True), start=2
        ):
            levels = np.where(diff <= t, rank, levels).astype(np.uint8)
        levels[null_mask] = 0
        return levels


class FuzzyString(_BaseComparison):
    """Bucket jaro-winkler similarity into discrete levels.

    Example: thresholds=(0.92, 0.80) -> levels are
        0 null
        1 sim < 0.80
        2 sim in [0.80, 0.92)
        3 sim >= 0.92
    """

    kind: Literal["fuzzy"] = "fuzzy"
    thresholds: tuple[float, ...] = Field(default=(0.92, 0.80))
    levels: int = 0

    def model_post_init(self, __context: object) -> None:
        object.__setattr__(self, "levels", 2 + len(self.thresholds))

    def assign_levels(self, pairs: pl.DataFrame) -> np.ndarray:
        # tight C loop via rapidfuzz; no python-per-char work
        from rapidfuzz.distance import JaroWinkler

        left = pairs[self.left_col()].to_list()
        right = pairs[self.right_col()].to_list()
        n = len(left)
        out = np.empty(n, dtype=np.float32)
        for i in range(n):
            a = left[i]
            b = right[i]
            if a is None or b is None:
                out[i] = np.nan
                continue
            # normalized_similarity returns 0..1
            out[i] = JaroWinkler.normalized_similarity(str(a), str(b))

        null_mask = np.isnan(out)
        levels = np.ones(n, dtype=np.uint8)
        # smallest threshold first -> larger levels for higher sim
        for rank, t in enumerate(sorted(self.thresholds), start=2):
            levels = np.where(out >= t, rank, levels).astype(np.uint8)
        levels[null_mask] = 0
        return levels


# pydantic discriminated union so YAML configs are type-safe
ComparisonSpec = ExactMatch | FuzzyString | NumericDiff
