"""Small explicit registry for application-owned checks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, get_args

from qualipilot.checks.base import Check, CheckContext
from qualipilot.models.config import BuiltInCheckName
from qualipilot.models.results import Severity

RegisteredCheck = Callable[[CheckContext], tuple[Severity, dict[str, Any]]]
_CHECKS: dict[str, RegisteredCheck] = {}


def register_check(name: str, check: RegisteredCheck) -> None:
    """Register an application callable under a config-safe explicit name."""
    if not name or name.strip() != name:
        raise ValueError("registered check names must not be blank or padded")
    if name in get_args(BuiltInCheckName):
        raise ValueError(f"registered check cannot shadow built-in: {name}")
    if name in _CHECKS:
        raise ValueError(f"check already registered: {name}")
    _CHECKS[name] = check


class RegisteredCheckAdapter(Check):
    def __init__(self, name: str) -> None:
        self.name = name

    def _execute(self, ctx: CheckContext) -> tuple[Severity, dict[str, Any]]:
        try:
            registered = _CHECKS[self.name]
        except KeyError as exc:
            raise ValueError(f"no check registered as {self.name!r}") from exc
        return registered(ctx)
