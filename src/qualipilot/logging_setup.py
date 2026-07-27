"""Logging configuration used by CLI entrypoints and library users.

Rich is preferred for developer-facing logs; plain JSON formatter is
used when running in cloud/Lambda where CloudWatch handles parsing.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

_STRUCTURED_FIELDS = (
    "check",
    "duration_seconds",
    "engine",
    "output_path",
    "report_format",
    "severity",
    "status",
)


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log event."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for field_name in _STRUCTURED_FIELDS:
            if hasattr(record, field_name):
                payload[field_name] = getattr(record, field_name)
        return json.dumps(payload, default=str)


def configure_logging(
    level: str = "INFO",
    *,
    json_logs: bool | None = None,
) -> None:
    """Configure the ``qualipilot`` package logger.

    Args:
        level: Standard python log level name.
        json_logs: If None, auto-detect based on ``QUALIPILOT_JSON_LOGS``
            env var. Set explicitly when calling from tests.
    """
    if json_logs is None:
        json_logs = os.environ.get("QUALIPILOT_JSON_LOGS", "").lower() in {
            "1",
            "true",
            "yes",
        }

    level_name = level.upper()
    numeric_level = logging.getLevelNamesMapping().get(level_name)
    if numeric_level is None:
        raise ValueError(f"unknown log level: {level}")

    package_logger = logging.getLogger("qualipilot")
    for existing in list(package_logger.handlers):
        if existing.get_name() == "qualipilot":
            package_logger.removeHandler(existing)
    package_logger.setLevel(numeric_level)
    package_logger.propagate = False

    handler: logging.Handler
    if json_logs:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
    else:
        try:
            from rich.logging import RichHandler

            handler = RichHandler(
                rich_tracebacks=True,
                show_time=True,
                show_path=False,
                markup=False,
            )
        except ImportError:
            # rich is a core dep but keep a fallback for slim images
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)-7s %(name)s - %(message)s"
                )
            )

    handler.set_name("qualipilot")
    package_logger.addHandler(handler)
