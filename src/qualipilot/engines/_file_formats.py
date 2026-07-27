"""Shared file-format validation."""

import csv
import glob
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import urlsplit, urlunsplit

from qualipilot.engines.base import validate_column_names


class _DuplicateJsonKeyError(ValueError):
    pass


class _InvalidJsonConstantError(ValueError):
    pass


def safe_source_name(value: str | Path) -> str:
    """Remove URL credentials and tokens from reportable source names."""
    raw = str(value)
    if "://" not in raw:
        return raw
    parsed = urlsplit(raw)
    netloc = parsed.netloc.rsplit("@", 1)[-1]
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def require_safe_remote_url(value: str | Path) -> None:
    """Reject credentials that downstream reader errors could disclose."""
    raw = str(value)
    if "://" not in raw:
        return
    parsed = urlsplit(raw)
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "remote source URLs must not contain userinfo, a query, or "
            "a fragment; configure credentials outside the URL"
        )


def require_unique_csv_columns(path: str | Path) -> list[str] | None:
    """Reject duplicate or inconsistent headers across local CSV inputs."""
    expected: list[str] | None = None
    for name, source in _text_sources(path, encoding="utf-8-sig"):
        reader = csv.reader(source, strict=True)
        try:
            header = next(reader, None)
            if header is None:
                raise ValueError(f"CSV input {name} must contain a header row")
            validate_column_names(header)
            if expected is not None and header != expected:
                raise ValueError("CSV files must use identical columns")
            for row_number, row in enumerate(reader, 2):
                if not row:
                    raise ValueError(
                        f"CSV row {row_number} in {name} must not be blank"
                    )
                if any("\0" in field for field in row):
                    raise ValueError(
                        f"CSV row {row_number} in {name} "
                        "must not contain NUL bytes"
                    )
                if len(row) != len(header):
                    raise ValueError(
                        f"CSV row {row_number} in {name} has {len(row)} "
                        f"fields; expected {len(header)}"
                    )
        except csv.Error as exc:
            raise ValueError(f"invalid CSV syntax in {name}: {exc}") from exc
        expected = header
    if expected is None:
        raise ValueError("CSV input must contain a header row")
    return expected


def require_valid_json_lines(path: str | Path) -> dict[str, str]:
    """Validate newline-delimited, flat JSON objects without losing keys."""
    columns: dict[str, str] = {}
    expected_columns: set[str] | None = None
    for name, source in _text_sources(path, encoding="utf-8"):
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = _load_json_record(line, name, line_number)
            record_columns = set(record)
            if (
                expected_columns is not None
                and record_columns != expected_columns
            ):
                raise ValueError(
                    "JSONL records must use identical columns "
                    f"(line {line_number} in {name})"
                )
            expected_columns = record_columns
            for column, value in record.items():
                family = _json_scalar_family(value)
                columns[column] = _merge_json_family(
                    columns.get(column, "null"),
                    family,
                    column=column,
                    name=name,
                    line_number=line_number,
                )
    if expected_columns is None:
        raise ValueError("JSONL input must contain at least one object")
    return {
        column: "string" if family == "null" else family
        for column, family in columns.items()
    }


def _load_json_record(
    line: str,
    name: str,
    line_number: int,
) -> dict[str, Any]:
    try:
        record = json.loads(
            line,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except _DuplicateJsonKeyError as exc:
        raise ValueError(
            f"JSONL record on line {line_number} in {name} "
            "contains duplicate object keys"
        ) from exc
    except (json.JSONDecodeError, _InvalidJsonConstantError) as exc:
        raise ValueError(
            f"invalid JSONL record on line {line_number} in {name}"
        ) from exc
    if not isinstance(record, dict):
        raise ValueError(
            f"JSONL record on line {line_number} in {name} must be an object"
        )
    if not record:
        raise ValueError(
            f"JSONL record on line {line_number} in {name} "
            "must contain at least one column"
        )
    if any(isinstance(value, (dict, list)) for value in record.values()):
        raise ValueError("JSONL input must contain only scalar values")
    validate_column_names(list(record))
    return record


def _merge_json_family(
    previous: str,
    family: str,
    *,
    column: str,
    name: str,
    line_number: int,
) -> str:
    if family == "null":
        return previous
    if previous == "null":
        return family
    if {previous, family} <= {"integer", "number"}:
        return "number" if "number" in {previous, family} else "integer"
    if previous != family:
        raise ValueError(
            "JSONL columns must use one scalar type "
            f"(column {column!r}, line {line_number} in {name})"
        )
    return previous


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise _InvalidJsonConstantError(value)


def _json_scalar_family(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        if not -(2**63) <= value < 2**63:
            raise ValueError("JSONL integers must fit signed 64-bit values")
        return "integer"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSONL numbers must be finite")
        return "number"
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError(
                "JSONL strings must contain valid Unicode"
            ) from exc
        return "string"
    raise ValueError("JSONL input must contain only scalar values")


def _text_sources(
    path: str | Path,
    *,
    encoding: str,
) -> Iterator[tuple[str, TextIO]]:
    raw = str(path)
    if "://" in raw:
        require_safe_remote_url(raw)
        try:
            import fsspec
        except ImportError as exc:
            raise ImportError(
                "remote text validation requires fsspec"
            ) from exc
        for remote_file in fsspec.open_files(
            raw,
            mode="rt",
            encoding=encoding,
            newline="",
        ):
            with remote_file as source:
                yield safe_source_name(raw), source
        return

    local = Path(path)
    if local.is_dir():
        paths = sorted(local.iterdir())
    elif glob.has_magic(raw):
        paths = [Path(match) for match in sorted(glob.glob(raw))]
    else:
        paths = [local]
    for candidate in paths:
        with candidate.open(encoding=encoding, newline="") as source:
            yield str(candidate), source
