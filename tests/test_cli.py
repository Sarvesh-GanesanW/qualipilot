"""CLI smoke tests using Typer's CliRunner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from qualipilot.cli import app
from qualipilot.models.config import LLMConfig

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "qualipilot" in result.stdout


def test_version_flag() -> None:
    """Both `qualipilot --version` and `qualipilot version` should work."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "qualipilot" in result.stdout


def test_invalid_engine_rejected() -> None:
    """Typer's Choice validation should reject typos before any work runs."""
    result = runner.invoke(
        app,
        ["check", "tests/conftest.py", "--engine", "polrs"],
    )
    assert result.exit_code != 0
    combined = result.stdout + (result.stderr or "")
    assert "polrs" in combined or "Invalid value" in combined


def test_bedrock_rejects_high_temperature() -> None:
    """Bedrock cannot accept temperature > 1.0; config should fail fast."""
    pattern = r"bedrock temperature must be <= 1\.0"
    with pytest.raises(ValueError, match=pattern):
        LLMConfig(provider="bedrock", temperature=1.5)


def test_non_bedrock_accepts_high_temperature() -> None:
    """openai-compatible endpoints accept up to 2.0; should not raise."""
    cfg = LLMConfig(provider="openai", temperature=1.5)
    assert cfg.temperature == 1.5


def test_check_writes_json(tmp_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "check",
            str(tmp_csv),
            "--engine",
            "polars",
            "--output",
            str(out),
            "--fail-on",
            "warn",
        ],
    )
    # warn severity triggers exit code 1
    assert result.exit_code in {0, 1}
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "results" in payload


def test_check_range_parsing(tmp_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "check",
            str(tmp_csv),
            "--output",
            str(out),
            "--range",
            "amount=0,100",
            "--fail-on",
            "error",
        ],
    )
    # expected: amount range violated -> exit 1
    assert result.exit_code == 1


def test_check_markdown_format(tmp_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    runner.invoke(
        app,
        ["check", str(tmp_csv), "--output", str(out)],
    )
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Data Quality Report")
