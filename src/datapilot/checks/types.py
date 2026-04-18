"""Data types check.

Reports dtype per column plus a rollup by type. Severity is always
informational (``ok``); downstream schema validation is what flags
real problems.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from datapilot.checks.base import Check, CheckContext


class DataTypesCheck(Check):
    name = "data_types"

    def _execute(self, ctx: CheckContext) -> tuple[str, dict[str, Any]]:
        dtypes = ctx.engine.dtypes()
        rollup = Counter(dtypes.values())
        return "ok", {
            "per_column": dtypes,
            "rollup": dict(rollup),
        }
