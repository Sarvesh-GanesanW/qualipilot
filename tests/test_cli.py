"""CLI smoke tests using Typer's CliRunner."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
import typer
from typer.testing import CliRunner

from qualipilot.cli import (
    EngineChoice,
    LLMChoice,
    SeverityChoice,
    _build_config,
    _build_consolidation_config,
    _compute_exit_code,
    _parse_merge_rule,
    _parse_survivor_sort,
    _stage_frame,
    app,
)
from qualipilot.models.config import LLMConfig
from qualipilot.models.results import CheckResult, DatasetStats, QualityReport

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "qualipilot" in result.stdout


def test_module_cli_renders_data_errors_without_traceback(
    tmp_path: Path,
) -> None:
    path = tmp_path / "duplicate-columns.csv"
    path.write_text("id,id\n1,2\n", encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, "-m", "qualipilot", "check", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode == 2
    assert "column names must be unique" in output
    assert "Traceback" not in output


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
    combined = result.output
    assert "polrs" in combined or "Invalid value" in combined


def test_bedrock_rejects_high_temperature() -> None:
    """Bedrock cannot accept temperature > 1.0; config should fail fast."""
    pattern = r"bedrock temperature must be <= 1\.0"
    with pytest.raises(ValueError, match=pattern):
        LLMConfig(provider="bedrock", temperature=1.5)


def test_bedrock_cli_requires_model(tmp_csv: Path) -> None:
    result = runner.invoke(app, ["check", str(tmp_csv), "--llm", "bedrock"])

    assert result.exit_code != 0
    combined = result.output
    assert "requires an explicit model" in combined


def test_non_bedrock_accepts_high_temperature() -> None:
    """openai-compatible endpoints accept up to 2.0; should not raise."""
    cfg = LLMConfig(
        provider="openai",
        model="test-model",
        temperature=1.5,
    )
    assert cfg.temperature == 1.5


def test_quiet_and_verbose_mutually_exclusive(tmp_csv: Path) -> None:
    result = runner.invoke(
        app,
        ["--quiet", "--verbose", "check", str(tmp_csv)],
    )
    assert result.exit_code != 0
    combined = result.output
    assert "mutually exclusive" in combined


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
    assert result.exit_code == 1
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


def test_invalid_range_is_a_cli_error(tmp_csv: Path) -> None:
    result = runner.invoke(
        app,
        ["check", str(tmp_csv), "--range", "amount=nan,100"],
    )

    assert result.exit_code != 0
    combined = result.output
    assert "invalid --range" in combined


def test_repeated_range_columns_are_rejected(tmp_csv: Path) -> None:
    result = runner.invoke(
        app,
        [
            "check",
            str(tmp_csv),
            "--range",
            "amount=0,10",
            "--range",
            " Amount =1,9",
        ],
    )

    assert result.exit_code == 2
    assert "repeats column" in result.output


def test_check_markdown_format(tmp_csv: Path, tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    runner.invoke(
        app,
        ["check", str(tmp_csv), "--output", str(out)],
    )
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Data Quality Report")


def test_file_report_format_survives_cli_defaults(
    tmp_csv: Path, tmp_path: Path
) -> None:
    config = tmp_path / "qualipilot.yaml"
    config.write_text("report_format: markdown\n", encoding="utf-8")
    output = tmp_path / "report.txt"

    result = runner.invoke(
        app,
        [
            "check",
            str(tmp_csv),
            "--config",
            str(config),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert output.read_text(encoding="utf-8").startswith(
        "# Data Quality Report"
    )


def test_cli_can_disable_file_configured_llm(
    tmp_csv: Path, tmp_path: Path
) -> None:
    config = tmp_path / "qualipilot.yaml"
    config.write_text(
        "llm:\n  provider: ollama\n  model: local-test\n",
        encoding="utf-8",
    )
    output = tmp_path / "report.json"

    result = runner.invoke(
        app,
        [
            "check",
            str(tmp_csv),
            "--config",
            str(config),
            "--llm",
            "none",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["llm_status"] == "disabled"


def test_provider_override_preserves_configured_llm_tuning(
    tmp_path: Path,
) -> None:
    config = tmp_path / "qualipilot.yaml"
    config.write_text(
        "engine: pandas\n"
        "llm:\n"
        "  provider: ollama\n"
        "  model: local-test\n"
        "  max_tokens: 777\n"
        "  temperature: 0.7\n"
        "  timeout_seconds: 12\n"
        "  retries: 7\n"
        "  system_prompt: custom\n",
        encoding="utf-8",
    )

    merged = _build_config(
        config=config,
        engine=EngineChoice.auto,
        report_format=None,
        llm_provider=LLMChoice.openai,
        llm_model="remote-test",
        bedrock_region=None,
        aws_profile=None,
        base_url="https://api.example.com/v1",
        allow_insecure_http=None,
        range_spec=None,
    )

    assert merged.llm.provider == "openai"
    assert merged.engine == "auto"
    assert merged.llm.max_tokens == 777
    assert merged.llm.temperature == 0.7
    assert merged.llm.timeout_seconds == 12
    assert merged.llm.retries == 7
    assert merged.llm.system_prompt == "custom"


def test_invalid_config_is_a_cli_error(tmp_csv: Path, tmp_path: Path) -> None:
    config = tmp_path / "qualipilot.yaml"
    config.write_text("checks: [\n", encoding="utf-8")

    result = runner.invoke(
        app,
        ["check", str(tmp_csv), "--config", str(config)],
    )

    assert result.exit_code == 2
    assert "invalid configuration" in result.output
    assert "Traceback" not in result.output


def test_api_key_is_not_a_cli_option(tmp_csv: Path) -> None:
    result = runner.invoke(
        app,
        ["check", str(tmp_csv), "--api-key", "secret"],
    )

    assert result.exit_code == 2
    assert "No such option" in result.output


def test_link_output_cannot_overwrite_input(tmp_path: Path) -> None:
    input_path = tmp_path / "records.csv"
    original = "id,name\n1,Alice\n2,Alice\n"
    input_path.write_text(original, encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "link",
            str(input_path),
            "--compare",
            "name:exact",
            "--output",
            str(input_path.resolve()),
        ],
    )

    assert result.exit_code == 2
    assert "must not overwrite" in result.output
    assert input_path.read_text(encoding="utf-8") == original


def test_deduplicated_output_requires_an_audit_path(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "records.csv"
    input_path.write_text("id,name\n1,Alice\n2,Alice\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "link",
            str(input_path),
            "--compare",
            "name:exact",
            "--deduplicated-output",
            str(tmp_path / "clean.csv"),
        ],
    )

    assert result.exit_code == 2
    assert "requires --output" in result.output


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("duplicates.csv", "id,id\n1,2\n", "column names must be unique"),
        (
            "duplicates.jsonl",
            '{"id":1,"id":2}\n',
            "duplicate object keys",
        ),
    ],
)
def test_link_rejects_ambiguous_input_files(
    tmp_path: Path,
    name: str,
    content: str,
    message: str,
) -> None:
    input_path = tmp_path / name
    input_path.write_text(content, encoding="utf-8")

    result = runner.invoke(
        app,
        ["link", str(input_path), "--compare", "id:exact"],
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)
    assert message in str(result.exception)


@pytest.mark.parametrize(
    "spec",
    ["name:fuzzy:not-a-number", "name:numeric:"],
)
def test_link_rejects_invalid_comparison(
    tmp_path: Path,
    spec: str,
) -> None:
    input_path = tmp_path / "records.csv"
    input_path.write_text("id,name\n1,Alice\n2,Alice\n", encoding="utf-8")

    result = runner.invoke(
        app,
        ["link", str(input_path), "--compare", spec],
    )

    assert result.exit_code == 2
    assert "invalid linkage configuration" in result.output
    assert "Traceback" not in result.output


def test_link_report_includes_reproducible_provenance(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "records.csv"
    output_path = tmp_path / "linkage.json"
    input_path.write_text(
        "id,name\n1,Alice\n2,Alice\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "link",
            str(input_path),
            "--compare",
            "name:exact",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "1.0"
    assert report["package_version"]
    assert report["source"] == str(input_path.resolve())
    assert len(report["source_sha256"]) == 64
    assert len(report["config_hash"]) == 64
    assert report["config"]["unique_id_column"] == "id"
    assert "parameters" in report


def test_link_report_streams_matched_pairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "records.csv"
    output_path = tmp_path / "linkage.json"
    input_path.write_text(
        "id,name\n1,Alice\n2,Alice\n",
        encoding="utf-8",
    )

    def reject_whole_frame_conversion(*args: object, **kwargs: object) -> None:
        raise AssertionError("to_dicts materializes the whole result")

    monkeypatch.setattr(
        pl.DataFrame, "to_dicts", reject_whole_frame_conversion
    )

    result = runner.invoke(
        app,
        [
            "link",
            str(input_path),
            "--compare",
            "name:exact",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert (
        json.loads(output_path.read_text(encoding="utf-8"))["matched_pairs"]
        == []
    )


def test_link_report_rejects_input_changed_during_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qualipilot.linking import RecordLinker

    input_path = tmp_path / "records.csv"
    output_path = tmp_path / "linkage.json"
    input_path.write_text(
        "id,name\n1,Alice\n2,Alice\n",
        encoding="utf-8",
    )
    original_run = RecordLinker.run

    def run_then_change_input(self: RecordLinker) -> object:
        result = original_run(self)
        input_path.write_text(
            "id,name\n1,Alice\n2,Alice\n3,Alice\n",
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(RecordLinker, "run", run_then_change_input)

    result = runner.invoke(
        app,
        [
            "link",
            str(input_path),
            "--compare",
            "name:exact",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 2
    assert "input changed during linkage" in result.output
    assert not output_path.exists()


def test_link_consolidates_records_with_lineage_and_normalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qualipilot.linking import LinkageResult, RecordLinker

    input_path = tmp_path / "records.csv"
    audit_path = tmp_path / "linkage.json"
    clean_path = tmp_path / "clean.csv"
    input_path.write_text(
        "id,name,email,updated,phone\n"
        '1," Alice  Smith ",,2024-01-01,111\n'
        "2,alice smith,alice@example.com,2024-03-01,\n"
        "3,Bob,bob@example.com,2024-02-01,333\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def deterministic_run(self: RecordLinker) -> LinkageResult:
        captured["normalization"] = self.config.normalization
        return LinkageResult(
            pairs=pl.DataFrame(
                {
                    "id_l": [1],
                    "id_r": [2],
                    "match_probability": [0.99],
                }
            ),
            clusters={1: 0, 2: 0, 3: 1},
            parameters={"threshold": 0.9, "lambda": 0.01},
        )

    monkeypatch.setattr(RecordLinker, "run", deterministic_run)

    result = runner.invoke(
        app,
        [
            "link",
            str(input_path),
            "--compare",
            "name:exact",
            "--output",
            str(audit_path),
            "--deduplicated-output",
            str(clean_path),
            "--survivor-sort",
            "updated:desc",
        ],
    )

    assert result.exit_code == 0, result.output
    normalization = captured["normalization"]
    assert isinstance(normalization, dict)
    assert set(normalization) == {"name", "email", "updated"}
    clean = pl.read_csv(clean_path)
    assert clean.to_dicts() == [
        {
            "id": 2,
            "name": "alice smith",
            "email": "alice@example.com",
            "updated": "2024-03-01",
            "phone": 111,
        },
        {
            "id": 3,
            "name": "bob",
            "email": "bob@example.com",
            "updated": "2024-02-01",
            "phone": 333,
        },
    ]
    audit_text = audit_path.read_text(encoding="utf-8")
    audit = json.loads(audit_text)
    assert audit["summary"]["consolidation"]["removed_count"] == 1
    assert audit["consolidation"]["lineage"] == {
        "1": 2,
        "2": 2,
        "3": 3,
    }
    assert audit["consolidation"]["output"] == {
        "path": str(clean_path.resolve()),
        "sha256": hashlib.sha256(clean_path.read_bytes()).hexdigest(),
    }
    assert "alice@example.com" not in audit_text
    assert pl.read_csv(input_path).height == 3


def test_audit_staging_failure_does_not_publish_clean_data(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "records.csv"
    input_path.write_text("id,name\n1,Alice\n2,Alice\n", encoding="utf-8")
    invalid_parent = tmp_path / "not-a-directory"
    invalid_parent.write_text("occupied", encoding="utf-8")
    clean_path = tmp_path / "clean.csv"

    result = runner.invoke(
        app,
        [
            "link",
            str(input_path),
            "--compare",
            "name:exact",
            "--output",
            str(invalid_parent / "audit.json"),
            "--deduplicated-output",
            str(clean_path),
        ],
    )

    assert result.exit_code == 1
    assert not clean_path.exists()
    assert list(tmp_path.glob("tmp*")) == []


def test_paired_output_failure_restores_previous_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "records.csv"
    input_path.write_text("id,name\n1,Alice\n2,Alice\n", encoding="utf-8")
    audit_path = tmp_path / "audit.json"
    clean_path = tmp_path / "clean.csv"
    audit_path.write_text("previous audit", encoding="utf-8")
    clean_path.write_text("previous data", encoding="utf-8")
    real_replace = os.replace
    replace_count = 0

    def fail_audit_publish(source: str | Path, target: str | Path) -> None:
        nonlocal replace_count
        replace_count += 1
        if replace_count == 2:
            raise OSError("simulated audit publish failure")
        real_replace(source, target)

    monkeypatch.setattr("qualipilot.cli.os.replace", fail_audit_publish)

    result = runner.invoke(
        app,
        [
            "link",
            str(input_path),
            "--compare",
            "name:exact",
            "--output",
            str(audit_path),
            "--deduplicated-output",
            str(clean_path),
        ],
    )

    assert result.exit_code == 1
    assert audit_path.read_text(encoding="utf-8") == "previous audit"
    assert clean_path.read_text(encoding="utf-8") == "previous data"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "audit.json",
        "clean.csv",
        "records.csv",
    ]


@pytest.mark.parametrize("suffix", [".csv", ".parquet", ".jsonl"])
def test_deduplicated_frame_staging_supports_each_format(
    tmp_path: Path,
    suffix: str,
) -> None:
    frame = pl.DataFrame({"id": [1], "name": ["Alice"]})
    path = tmp_path / f"clean{suffix}"

    staged = _stage_frame(path, frame)

    if suffix == ".csv":
        loaded = pl.read_csv(staged)
    elif suffix == ".parquet":
        loaded = pl.read_parquet(staged)
    else:
        loaded = pl.read_ndjson(staged)
    assert loaded.equals(frame)
    assert not path.exists()
    staged.unlink()


def test_consolidation_option_parsers_reject_ambiguous_rules() -> None:
    assert _parse_survivor_sort("updated:desc").descending
    column, rule = _parse_merge_rule("email:latest:updated")
    assert (column, rule.strategy, rule.order_by) == (
        "email",
        "latest",
        "updated",
    )
    with pytest.raises(typer.BadParameter, match="require an order_by"):
        _parse_merge_rule("email:latest")
    with pytest.raises(typer.BadParameter, match=r"asc\|desc"):
        _parse_survivor_sort("updated")

    frame = pl.DataFrame({"id": [1], "name": ["Alice"]})
    config = _build_consolidation_config(
        frame,
        id_column="id",
        survivor_sort=[],
        completeness=None,
        rank_by_completeness=False,
        merge=[],
    )
    assert config.completeness_columns == ()
    with pytest.raises(typer.BadParameter, match="cannot be combined"):
        _build_consolidation_config(
            frame,
            id_column="id",
            survivor_sort=[],
            completeness=["name"],
            rank_by_completeness=False,
            merge=[],
        )
    with pytest.raises(typer.BadParameter, match="must be unique"):
        _build_consolidation_config(
            frame,
            id_column="id",
            survivor_sort=[],
            completeness=None,
            rank_by_completeness=True,
            merge=["name:survivor", " name:first_non_null"],
        )


def test_llm_failure_is_an_operational_error() -> None:
    report = QualityReport(
        dataset=DatasetStats(
            row_count=0,
            column_count=0,
            columns=[],
            dtypes={},
            engine="polars",
        ),
        results=[],
        llm_status="failed",
        llm_error="network failure",
    )

    assert _compute_exit_code(report, SeverityChoice.error) == 2


def test_check_execution_failure_is_an_operational_error() -> None:
    report = QualityReport(
        dataset=DatasetStats(
            row_count=0,
            column_count=0,
            columns=[],
            dtypes={},
            engine="polars",
        ),
        results=[
            CheckResult(
                name="custom",
                severity="error",
                status="failed",
                duration_seconds=0,
                error="engine failure",
            )
        ],
    )

    assert _compute_exit_code(report, SeverityChoice.error) == 2
