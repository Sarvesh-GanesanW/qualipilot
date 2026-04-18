"""Base types shared by every check."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from datapilot.engines.base import Engine
from datapilot.models.config import CheckConfig
from datapilot.models.results import CheckResult


@dataclass(slots=True)
class CheckContext:
    """Immutable inputs handed to every check."""

    engine: Engine
    config: CheckConfig


class Check(ABC):
    """Base class for data quality checks.

    Subclasses implement ``_execute`` which returns the serialisable
    payload. The base class times execution, captures exceptions, and
    assigns the severity.
    """

    name: str

    def run(self, ctx: CheckContext) -> CheckResult:
        """Execute the check, wrapping timing and error handling."""
        start = time.perf_counter()
        try:
            severity, payload = self._execute(ctx)
        except Exception as exc:
            return CheckResult(
                name=self.name,
                severity="error",
                duration_seconds=time.perf_counter() - start,
                payload={},
                error=f"{type(exc).__name__}: {exc}",
            )
        return CheckResult(
            name=self.name,
            severity=severity,
            duration_seconds=time.perf_counter() - start,
            payload=payload,
        )

    @abstractmethod
    def _execute(self, ctx: CheckContext) -> tuple[str, dict[str, Any]]:
        """Run the check logic.

        Returns:
            Tuple of ``(severity, payload)``. Severity is ``"ok"``,
            ``"warn"`` or ``"error"``. Payload is a JSON-serialisable
            dict stored on the resulting ``CheckResult``.
        """
