"""Comparison primitives.

A *comparison* inspects one column of a candidate pair and returns an
integer **level**. Level 0 is always the "null / no signal" case.
Higher levels mean stronger agreement; the top level is exact match.

Levels are kept small (typically 2-4) so the Fellegi-Sunter EM has
enough data to estimate each level's m/u probability reliably.
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Literal

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator


class _BaseComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)

    column: str = Field(min_length=1)

    # each concrete subclass sets this
    kind: str

    def left_col(self) -> str:
        return f"{self.column}_l"

    def right_col(self) -> str:
        return f"{self.column}_r"

    @property
    def levels(self) -> int:
        raise NotImplementedError

    @field_validator("column")
    @classmethod
    def _strip_column(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("column must not be blank")
        return value


class ExactMatch(_BaseComparison):
    """Three levels: null (0), different (1), exact match (2)."""

    kind: Literal["exact"] = "exact"

    @property
    def levels(self) -> int:
        return 3

    def assign_levels(self, pairs: pl.DataFrame) -> np.ndarray:
        left = pairs[self.left_col()]
        right = pairs[self.right_col()]
        # null on either side -> level 0
        null_mask = _missing_mask(left) | _missing_mask(right)
        equal_mask = (left == right).fill_null(False).to_numpy().astype(bool)
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
    thresholds: tuple[float, ...] = Field(
        default=(1.0, 5.0),
        min_length=1,
        max_length=253,
    )

    @property
    def levels(self) -> int:
        return 2 + len(self.thresholds)

    @field_validator("thresholds")
    @classmethod
    def _validate_thresholds(
        cls,
        values: tuple[float, ...],
    ) -> tuple[float, ...]:
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("numeric thresholds must be finite and >= 0")
        if len(set(values)) != len(values):
            raise ValueError("numeric thresholds must be unique")
        return values

    def assign_levels(self, pairs: pl.DataFrame) -> np.ndarray:
        left_name = self.left_col()
        right_name = self.right_col()
        left = pairs[left_name]
        right = pairs[right_name]
        if left.dtype.is_integer() and right.dtype.is_integer():
            diff_expr = (
                pl.col(left_name).cast(pl.Decimal(38, 0))
                - pl.col(right_name).cast(pl.Decimal(38, 0))
            ).abs()
            level_expr = pl.lit(1, dtype=pl.UInt8)
            for rank, threshold in enumerate(
                sorted(self.thresholds, reverse=True),
                start=2,
            ):
                # NumericDiff rejects 128-bit inputs, so every possible
                # cross-signed integer difference is below 2**65.
                within_threshold = (
                    pl.lit(True)
                    if threshold >= 2**65
                    else diff_expr <= pl.lit(Decimal(str(threshold)))
                )
                level_expr = (
                    pl.when(within_threshold)
                    .then(pl.lit(rank, dtype=pl.UInt8))
                    .otherwise(level_expr)
                )
            return (
                pairs.select(
                    pl.when(
                        pl.col(left_name).is_null()
                        | pl.col(right_name).is_null()
                    )
                    .then(pl.lit(0, dtype=pl.UInt8))
                    .otherwise(level_expr)
                    .alias("__level__")
                )
                .to_series()
                .to_numpy()
                .astype(np.uint8)
            )
        if left.dtype.is_decimal() and right.dtype.is_decimal():
            left_expr = pl.col(left_name)
            right_expr = pl.col(right_name)
            same_sign = ((left_expr >= 0) & (right_expr >= 0)) | (
                (left_expr < 0) & (right_expr < 0)
            )
            zero = pl.lit(0).cast(left.dtype)
            decimal_diff = (
                pl.when(same_sign).then(left_expr).otherwise(zero)
                - pl.when(same_sign).then(right_expr).otherwise(zero)
            ).abs()
            float_diff = (
                left_expr.cast(pl.Float64) - right_expr.cast(pl.Float64)
            ).abs()
            diff_expr = (
                pl.when(same_sign).then(decimal_diff).otherwise(float_diff)
            )
        else:
            diff_expr = (
                pl.col(left_name).cast(pl.Float64)
                - pl.col(right_name).cast(pl.Float64)
            ).abs()
        diff = (
            pairs.select(diff_expr.cast(pl.Float64).alias("__difference__"))
            .to_series()
            .to_numpy()
        )
        null_mask = np.isnan(diff)

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
    thresholds: tuple[float, ...] = Field(
        default=(0.92, 0.80),
        min_length=1,
        max_length=253,
    )

    @property
    def levels(self) -> int:
        return 2 + len(self.thresholds)

    @field_validator("thresholds")
    @classmethod
    def _validate_thresholds(
        cls,
        values: tuple[float, ...],
    ) -> tuple[float, ...]:
        if any(
            not math.isfinite(value) or not 0 <= value <= 1 for value in values
        ):
            raise ValueError("fuzzy thresholds must be finite and in [0, 1]")
        if len(set(values)) != len(values):
            raise ValueError("fuzzy thresholds must be unique")
        return values

    def assign_levels(self, pairs: pl.DataFrame) -> np.ndarray:
        # tight C loop via rapidfuzz; no python-per-char work
        try:
            from rapidfuzz.distance import JaroWinkler
        except ImportError as exc:
            raise ImportError(
                "rapidfuzz is required for fuzzy linking; "
                "install with `pip install qualipilot[linking]`"
            ) from exc

        left = pairs[self.left_col()].to_list()
        right = pairs[self.right_col()].to_list()
        n = len(left)
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            a = left[i]
            b = right[i]
            if _scalar_is_missing(a) or _scalar_is_missing(b):
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


def _missing_mask(series: pl.Series) -> np.ndarray:
    missing = series.is_null()
    if series.dtype.is_float():
        missing = missing | series.is_nan()
    return missing.to_numpy().astype(bool)


def _scalar_is_missing(value: object) -> bool:
    return value is None or (
        isinstance(value, float | np.floating) and math.isnan(float(value))
    )
