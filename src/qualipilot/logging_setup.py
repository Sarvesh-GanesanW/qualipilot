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


class _JsonFormatter(logging.Formatter):
    """Emit a single-line JSON record per log event.

    keeps CloudWatch + Datadog happy without pulling in a heavy
    logging dependency.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(
    level: str = "INFO",
    *,
    json_logs: bool | None = None,
) -> None:
    """Wire up the root logger once at process startup.

    Args:
        level: Standard python log level name.
        json_logs: If None, auto-detect based on ``QUALIPILOT_JSON_LOGS``
            env var. Set explicitly when calling from tests.
    """
    # auto-detect cloud envs so we do not have to pass flags around
    if json_logs is None:
        json_logs = os.environ.get("QUALIPILOT_JSON_LOGS", "").lower() in {
            "1",
            "true",
            "yes",
        }

    root = logging.getLogger()
    # wipe existing handlers so repeated calls in notebooks stay clean
    for handler in list(root.handlers):
        root.removeHandler(handler)

    root.setLevel(level.upper())

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

    root.addHandler(handler)
    # silence noisy third-party loggers that pollute our output
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
