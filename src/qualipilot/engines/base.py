"""Interface implemented by dataframe backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import UTC, date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow as pa


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

    def dtype_family(self, column: str) -> str:
        """Return a portable logical dtype family for ``column``."""
        return dtype_family_from_name(self.dtypes()[column])

    @abstractmethod
    def numeric_columns(self) -> list[str]:
        """Return names of columns with numeric dtype."""

    @abstractmethod
    def datetime_columns(self) -> list[str]:
        """Return names of columns with datetime dtype."""

    # ---- per-column stats ----------------------------------------------

    @abstractmethod
    def null_counts(self) -> dict[str, int]:
        """Return missing-value count per column (nulls and floating NaNs)."""

    @abstractmethod
    def distinct_count(self, column: str) -> int:
        """Return distinct non-missing value count for ``column``."""

    def distinct_counts(self, columns: list[str]) -> dict[str, int]:
        """Return distinct non-missing value counts for ``columns``."""
        return {column: self.distinct_count(column) for column in columns}

    @abstractmethod
    def top_values(
        self,
        column: str,
        n: int = 10,
    ) -> list[tuple[str, int]]:
        """Return common non-missing values with deterministic tie ordering."""

    @abstractmethod
    def quantiles(
        self,
        columns: list[str],
        qs: tuple[float, ...] = (0.25, 0.75),
    ) -> dict[str, dict[float, float]]:
        """Return linear quantiles where the backend supports them.

        Distributed engines may return deterministic approximations rather
        than collecting the full column into driver memory.

        Args:
            columns: Numeric columns to profile.
            qs: Quantile fractions in ``[0, 1]``.

        Returns:
            ``{column: {q: value}}``.
        """

    @property
    def quantile_provenance(self) -> dict[str, str | float]:
        """Describe the quantiles exposed by this engine."""
        return {"method": "exact"}

    # ---- filters -------------------------------------------------------

    @abstractmethod
    def duplicate_count(self, subset: list[str] | None = None) -> int:
        """Return the total number of duplicate rows.

        Unlike partition-local duplicated counts, this must see every
        row globally to avoid under-counting on distributed engines.
        Null and NaN are the same missing-value key.
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

    def counts_outside(
        self,
        ranges: dict[str, tuple[float, float]],
    ) -> dict[str, int]:
        """Return outside-range row counts for multiple columns."""
        return {
            column: self.count_outside(column, low, high)
            for column, (low, high) in ranges.items()
        }

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

    def max_datetime_instant(
        self,
        column: str,
        naive_timezone: str = "UTC",
    ) -> datetime | None:
        """Return the latest value as a timezone-aware UTC instant."""
        value = self.max_datetime(column)
        return (
            None if value is None else as_utc_datetime(value, naive_timezone)
        )


def reject_nested_columns(columns: list[str]) -> None:
    """Enforce the scalar-column contract shared by every engine."""
    if columns:
        names = ", ".join(sorted(columns))
        raise TypeError(
            "nested list, object, map, and struct columns are not supported; "
            f"flatten these columns first: {names}"
        )


def validate_column_names(columns: Sequence[object]) -> None:
    """Reject names that backends interpret inconsistently."""
    if any(not isinstance(column, str) for column in columns):
        raise ValueError("dataset column names must be strings")
    names = [str(column) for column in columns]
    try:
        for name in names:
            name.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(
            "dataset column names must contain valid Unicode"
        ) from exc
    if any("\0" in name for name in names):
        raise ValueError("dataset column names must not contain NUL bytes")
    if any(not name.strip() for name in names):
        raise ValueError("dataset column names must not be blank")
    if any(name != name.strip() for name in names):
        raise ValueError(
            "dataset column names must not have surrounding whitespace"
        )
    if len(names) != len({name.casefold() for name in names}):
        raise ValueError("dataset column names must be unique ignoring case")


def validate_pandas_columns(frame: pd.DataFrame) -> None:
    """Reject Pandas values that portable engines interpret differently."""
    validate_column_names(list(frame.columns))
    reject_nested_columns(
        [
            column
            for column in frame.columns
            if is_nested_arrow_dtype(frame[column].dtype)
            or (
                pd.api.types.is_object_dtype(frame[column].dtype)
                and not frame[column].map(pd.api.types.is_scalar).all()
            )
        ]
    )
    portable_object_families = {
        "boolean",
        "bytes",
        "date",
        "datetime",
        "datetime64",
        "decimal",
        "empty",
        "floating",
        "integer",
        "mixed-integer-float",
        "string",
        "time",
        "timedelta",
        "timedelta64",
    }
    unsupported: list[str] = []
    for column, dtype in frame.dtypes.items():
        if is_unsupported_pandas_dtype(dtype):
            unsupported.append(f"{column} ({dtype})")
        elif pd.api.types.is_object_dtype(dtype) or isinstance(
            dtype, pd.CategoricalDtype
        ):
            values = (
                frame[column].astype(object)
                if isinstance(dtype, pd.CategoricalDtype)
                else frame[column]
            )
            family = pd.api.types.infer_dtype(values, skipna=True)
            if family not in portable_object_families:
                unsupported.append(f"{column} ({dtype}/{family})")
    if unsupported:
        raise TypeError(
            "unsupported pandas column types: "
            + ", ".join(unsupported)
            + "; cast them to a portable string, numeric, date, or "
            "binary dtype"
        )


def is_unsupported_pandas_dtype(dtype: Any) -> bool:
    """Return whether a Pandas dtype loses meaning in another engine."""
    return pd.api.types.is_complex_dtype(dtype) or isinstance(
        dtype,
        (pd.PeriodDtype, pd.IntervalDtype, pd.SparseDtype),
    )


def is_nested_arrow_dtype(dtype: Any) -> bool:
    """Return whether a pandas-compatible dtype wraps a nested Arrow type."""
    arrow_dtype = getattr(dtype, "pyarrow_dtype", None)
    return isinstance(arrow_dtype, pa.DataType) and pa.types.is_nested(
        arrow_dtype
    )


def is_decimal_arrow_dtype(dtype: Any) -> bool:
    """Return whether a pandas-compatible dtype wraps Arrow Decimal."""
    arrow_dtype = getattr(dtype, "pyarrow_dtype", None)
    return isinstance(arrow_dtype, pa.DataType) and pa.types.is_decimal(
        arrow_dtype
    )


def is_date_arrow_dtype(dtype: Any) -> bool:
    """Return whether a pandas-compatible dtype wraps an Arrow date."""
    arrow_dtype = getattr(dtype, "pyarrow_dtype", None)
    return isinstance(arrow_dtype, pa.DataType) and pa.types.is_date(
        arrow_dtype
    )


def as_utc_datetime(value: Any, naive_timezone: str = "UTC") -> datetime:
    """Parse one ISO-like temporal value into a comparable UTC instant."""
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, time.min)
    else:
        parsed = datetime.fromisoformat(str(value).strip())
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(naive_timezone))
    return parsed.astimezone(UTC)


def dtype_family_from_name(dtype: str) -> str:
    """Map native engine dtype spellings to portable logical families."""
    value = dtype.strip().casefold()
    if (
        value.startswith("uint")
        or (value.startswith("int") and value[3:4].isdigit())
        or value
        in {
            "byte",
            "int",
            "short",
            "long",
            "tinyint",
            "smallint",
            "integer",
            "bigint",
            "hugeint",
            "utinyint",
            "usmallint",
            "uinteger",
            "ubigint",
        }
    ):
        return "integer"
    for prefixes, family in (
        (("float", "double", "real"), "float"),
        (("decimal",), "decimal"),
        (("bool", "boolean"), "boolean"),
        (("datetime", "timestamp"), "datetime"),
        (("date",), "date"),
        (("duration", "timedelta"), "duration"),
        (("time",), "time"),
        (("binary", "bytes"), "binary"),
        (("str", "string", "varchar", "char"), "string"),
        (("categorical", "category", "enum"), "categorical"),
    ):
        if value.startswith(prefixes):
            return family
    return {"blob": "binary", "text": "string", "utf8": "string"}.get(
        value, value
    )


def object_dtype_family(families: set[str]) -> str:
    """Return the portable family represented by inferred object values."""
    if families == {"integer"}:
        return "integer"
    if families in ({"floating"}, {"integer", "floating"}):
        return "float"
    if families == {"decimal"}:
        return "decimal"
    if families == {"boolean"}:
        return "boolean"
    if families <= {"date", "datetime", "datetime64"} and families:
        return "datetime" if families != {"date"} else "date"
    if families <= {"timedelta", "timedelta64"} and families:
        return "duration"
    if families == {"time"}:
        return "time"
    if families == {"bytes"}:
        return "binary"
    if families == {"string"}:
        return "string"
    return "object"
