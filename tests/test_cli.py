"""CLI smoke tests using Typer's CliRunner."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from qualipilot.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "qualipilot" in result.stdout


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
