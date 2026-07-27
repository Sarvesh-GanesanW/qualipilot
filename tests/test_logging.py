"""Logging setup regression tests."""

from __future__ import annotations

import json
import logging

import pytest

from qualipilot.logging_setup import _JsonFormatter, configure_logging


def test_configure_logging_preserves_root_handlers() -> None:
    root = logging.getLogger()
    sentinel = logging.NullHandler()
    root.addHandler(sentinel)
    try:
        configure_logging(json_logs=True)
        assert sentinel in root.handlers
    finally:
        root.removeHandler(sentinel)


def test_json_formatter_keeps_structured_check_fields() -> None:
    record = logging.LogRecord(
        name="qualipilot.checker",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="finished",
        args=(),
        exc_info=None,
    )
    record.check = "ranges"
    record.severity = "error"

    payload = json.loads(_JsonFormatter().format(record))

    assert payload["check"] == "ranges"
    assert payload["severity"] == "error"


def test_configure_logging_rejects_unknown_level() -> None:
    with pytest.raises(ValueError, match="unknown log level"):
        configure_logging("loud")
