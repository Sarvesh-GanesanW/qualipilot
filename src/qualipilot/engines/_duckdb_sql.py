"""Small SQL-safety helpers shared by the DuckDB adapters."""

from __future__ import annotations


def quote_identifier(value: str) -> str:
    """Return ``value`` as a safely quoted DuckDB identifier."""
    return '"' + value.replace('"', '""') + '"'


def quote_literal(value: str) -> str:
    """Return ``value`` as a safely quoted DuckDB string literal."""
    return "'" + value.replace("'", "''") + "'"
