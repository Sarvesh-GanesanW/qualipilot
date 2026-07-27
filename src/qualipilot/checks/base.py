"""Base types shared by every check."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from qualipilot.engines.base import Engine
from qualipilot.models.config import CheckConfig
from qualipilot.models.results import CheckResult, Severity

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CheckContext:
    """Immutable inputs handed to every check."""

    engine: Engine
    config: CheckConfig
    row_count: int | None = None
    columns: list[str] | None = None
    dtypes: dict[str, str] | None = None


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
            return CheckResult(
                name=self.name,
                severity=severity,
                duration_seconds=time.perf_counter() - start,
                payload=payload,
            )
        except Exception as exc:
            logger.exception("check %s failed", self.name)
            return CheckResult(
                name=self.name,
                severity="error",
                status="failed",
                duration_seconds=time.perf_counter() - start,
                payload={},
                error=f"{type(exc).__name__}: {exc}",
            )

    @abstractmethod
    def _execute(self, ctx: CheckContext) -> tuple[Severity, dict[str, Any]]:
        """Run the check logic.

        Returns:
            Tuple of ``(severity, payload)``. Severity is ``"ok"``,
            ``"warn"`` or ``"error"``. Payload is a JSON-serialisable
            dict stored on the resulting ``CheckResult``.
        """
