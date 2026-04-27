"""Engine protocol — the contract every dataframe backend must meet.

Keeping this thin deliberately: anything that can be answered by a
``SELECT`` over a single logical table. Joins, stats beyond quantiles,
and sampling policies live in individual checks, not here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Engine(ABC):
    """Abstract base for dataframe backends."""

    name: str

    # ---- structural info ------------------------------------------------

    @abstractmethod
    def row_count(self) -> int:
        """Return the number of rows in the underlying dataframe."""

    @abstractmethod
    def columns(self) -> list[str]:
        """Return column names preserving original order."""

    @abstractmethod
    def dtypes(self) -> dict[str, str]:
        """Return a mapping of column name to dtype string."""

    @abstractmethod
    def numeric_columns(self) -> list[str]:
        """Return names of columns with numeric dtype."""

    @abstractmethod
    def datetime_columns(self) -> list[str]:
        """Return names of columns with datetime dtype."""

    # ---- per-column stats ----------------------------------------------

    @abstractmethod
    def null_counts(self) -> dict[str, int]:
        """Return null count per column."""

    @abstractmethod
    def distinct_count(self, column: str) -> int:
        """Return distinct value count for ``column``."""

    @abstractmethod
    def top_values(
        self,
        column: str,
        n: int = 10,
    ) -> list[tuple[str, int]]:
        """Return ``(value, count)`` pairs for the most common values."""

    @abstractmethod
    def quantiles(
        self,
        columns: list[str],
        qs: tuple[float, ...] = (0.25, 0.75),
    ) -> dict[str, dict[float, float]]:
        """Return quantile values, computed in a single pass where possible.

        Args:
            columns: Numeric columns to profile.
            qs: Quantile fractions in ``[0, 1]``.

        Returns:
            ``{column: {q: value}}``.
        """

    @abstractmethod
    def describe(self) -> dict[str, dict[str, float]]:
        """Return pandas-style describe() per numeric column."""

    # ---- filters -------------------------------------------------------

    @abstractmethod
    def duplicate_count(self, subset: list[str] | None = None) -> int:
        """Return the total number of duplicate rows.

        Unlike partition-local duplicated counts, this must see every
        row globally to avoid under-counting on distributed engines.
        """

    @abstractmethod
    def sample_duplicates(
        self,
        n: int,
        subset: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return up to ``n`` duplicate rows as dicts."""

    @abstractmethod
    def count_outside(
        self,
        column: str,
        low: float,
        high: float,
    ) -> int:
        """Return row count where ``column`` falls outside ``[low, high]``."""

    @abstractmethod
    def sample_outside(
        self,
        column: str,
        low: float,
        high: float,
        n: int,
    ) -> list[dict[str, Any]]:
        """Return up to ``n`` sample rows violating the range."""

    @abstractmethod
    def max_datetime(self, column: str) -> Any:
        """Return the max value of a datetime column, or None if empty."""
