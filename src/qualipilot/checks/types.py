"""Dataset contract and dtype inventory checks."""

from __future__ import annotations

from collections import Counter
from typing import Any

from qualipilot.checks.base import Check, CheckContext
from qualipilot.models.results import Severity


class DatasetContractCheck(Check):
    name = "dataset_contract"

    def _execute(self, ctx: CheckContext) -> tuple[Severity, dict[str, Any]]:
        row_count = (
            ctx.row_count
            if ctx.row_count is not None
            else ctx.engine.row_count()
        )
        columns = (
            ctx.columns if ctx.columns is not None else ctx.engine.columns()
        )
        required = [
            *ctx.config.required_columns,
            *(
                column
                for column in ctx.config.expected_dtypes
                if column not in ctx.config.required_columns
            ),
        ]
        missing = [column for column in required if column not in columns]
        dtypes = ctx.dtypes if ctx.dtypes is not None else ctx.engine.dtypes()
        dtype_mismatches = [
            {
                "column": column,
                "expected": expected,
                "actual": dtypes[column],
            }
            for column, expected in ctx.config.expected_dtypes.items()
            if column in dtypes and dtypes[column] != expected
        ]
        is_too_small = row_count < ctx.config.min_rows
        severity: Severity = (
            "error" if missing or dtype_mismatches or is_too_small else "ok"
        )
        return severity, {
            "row_count": row_count,
            "min_rows": ctx.config.min_rows,
            "missing_required_columns": missing,
            "dtype_mismatches": dtype_mismatches,
        }


class DataTypesCheck(Check):
    name = "data_types"

    def _execute(self, ctx: CheckContext) -> tuple[Severity, dict[str, Any]]:
        dtypes = ctx.dtypes if ctx.dtypes is not None else ctx.engine.dtypes()
        rollup = Counter(dtypes.values())
        return "ok", {
            "per_column": dtypes,
            "rollup": dict(rollup),
        }
