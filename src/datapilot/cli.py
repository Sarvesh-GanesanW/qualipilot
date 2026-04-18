"""Typer-based CLI.

Goal: one command (`datapilot check`) should take a CSV/Parquet/etc.
and produce a machine-readable report plus optional LLM narrative,
without editing any Python files.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from datapilot import __version__
from datapilot.checker import DataQualityChecker
from datapilot.logging_setup import configure_logging
from datapilot.models.config import (
    ColumnRange,
    DatapilotConfig,
    LLMConfig,
)
from datapilot.models.results import QualityReport
from datapilot.reporting import render_html, render_markdown

app = typer.Typer(
    name="datapilot",
    help="Run data quality checks and (optionally) an LLM report.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()


@app.callback()
def _root(
    log_level: Annotated[
        str, typer.Option("--log-level", envvar="DATAPILOT_LOG_LEVEL")
    ] = "INFO",
    json_logs: Annotated[
        bool,
        typer.Option(
            "--json-logs/--rich-logs",
            envvar="DATAPILOT_JSON_LOGS",
        ),
    ] = False,
) -> None:
    """Global options that apply to every sub-command."""
    configure_logging(level=log_level, json_logs=json_logs)


@app.command()
def version() -> None:
    """Print the installed package version."""
    console.print(f"datapilot {__version__}")


@app.command()
def check(
    input_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            readable=True,
            help="CSV/Parquet/JSON/NDJSON file to inspect.",
        ),
    ],
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            exists=True,
            readable=True,
            help="YAML/JSON config with checks + llm settings.",
        ),
    ] = None,
    engine: Annotated[
        str,
        typer.Option(
            "--engine",
            "-e",
            help="auto | polars | pandas | dask | cudf",
        ),
    ] = "auto",
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Write the report to this path (json/html/md).",
        ),
    ] = None,
    report_format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="json | html | markdown (derived from --output if omitted).",
        ),
    ] = "json",
    llm_provider: Annotated[
        str,
        typer.Option(
            "--llm",
            help="none | bedrock | ollama | openai",
        ),
    ] = "none",
    llm_model: Annotated[
        str,
        typer.Option("--model", help="Model id/name for the chosen LLM."),
    ] = "",
    bedrock_region: Annotated[
        str,
        typer.Option("--region", envvar="AWS_REGION"),
    ] = "us-east-1",
    aws_profile: Annotated[
        str | None,
        typer.Option("--profile", envvar="AWS_PROFILE"),
    ] = None,
    base_url: Annotated[
        str,
        typer.Option("--base-url"),
    ] = "http://localhost:11434",
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", envvar="DATAPILOT_LLM_API_KEY"),
    ] = None,
    range_spec: Annotated[
        list[str] | None,
        typer.Option(
            "--range",
            help='Per-column range: "col=min,max" (repeatable).',
        ),
    ] = None,
    fail_on: Annotated[
        str,
        typer.Option(
            "--fail-on",
            help="Exit non-zero when any check hits this severity "
            "(ok | warn | error).",
        ),
    ] = "error",
) -> None:
    """Run data quality checks over ``input_path``."""
    cfg = _build_config(
        config=config,
        engine=engine,
        report_format=report_format,
        llm_provider=llm_provider,
        llm_model=llm_model,
        bedrock_region=bedrock_region,
        aws_profile=aws_profile,
        base_url=base_url,
        api_key=api_key,
        range_spec=range_spec,
    )

    checker = DataQualityChecker(input_path, cfg)
    report = checker.run()

    _write_output(report, output, cfg.report_format)
    _print_summary(report)

    exit_code = _compute_exit_code(report, fail_on)
    raise typer.Exit(code=exit_code)


# ---- helpers ------------------------------------------------------------


def _build_config(
    *,
    config: Path | None,
    engine: str,
    report_format: str,
    llm_provider: str,
    llm_model: str,
    bedrock_region: str,
    aws_profile: str | None,
    base_url: str,
    api_key: str | None,
    range_spec: list[str] | None,
) -> DatapilotConfig:
    """Merge CLI flags on top of a YAML/JSON base config if supplied."""
    cfg = DatapilotConfig.from_file(config) if config else DatapilotConfig()

    # cli flags win over file/env unless flag is still at its default
    if engine != "auto":
        cfg.engine = engine  # type: ignore[assignment]
    if report_format:
        cfg.report_format = report_format  # type: ignore[assignment]

    if llm_provider and llm_provider != "none":
        cfg.llm = LLMConfig(
            provider=llm_provider,  # type: ignore[arg-type]
            model=llm_model or cfg.llm.model,
            region=bedrock_region,
            aws_profile=aws_profile,
            base_url=base_url,
            api_key=api_key,
        )

    if range_spec:
        ranges = _parse_ranges(range_spec)
        merged = dict(cfg.checks.column_ranges)
        merged.update(ranges)
        cfg.checks = cfg.checks.model_copy(update={"column_ranges": merged})

    return cfg


def _parse_ranges(specs: list[str]) -> dict[str, ColumnRange]:
    out: dict[str, ColumnRange] = {}
    for raw in specs:
        if "=" not in raw or "," not in raw:
            raise typer.BadParameter(
                f"--range expects 'col=min,max', got {raw!r}"
            )
        col, bounds = raw.split("=", 1)
        lo_s, hi_s = bounds.split(",", 1)
        out[col.strip()] = ColumnRange(min=float(lo_s), max=float(hi_s))
    return out


def _write_output(
    report: QualityReport,
    output: Path | None,
    fmt: str,
) -> None:
    if output is None:
        return
    fmt_effective = _infer_format(output, fmt)
    if fmt_effective == "html":
        payload = render_html(report)
    elif fmt_effective == "markdown":
        payload = render_markdown(report)
    else:
        payload = report.to_json()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    console.print(f"report written to [bold]{output}[/bold]")


def _infer_format(output: Path, fmt: str) -> str:
    suffix = output.suffix.lower()
    if suffix == ".html":
        return "html"
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix == ".json":
        return "json"
    return fmt


def _print_summary(report: QualityReport) -> None:
    table = Table(title="Data Quality Summary", show_lines=False)
    table.add_column("Check", style="cyan", no_wrap=True)
    table.add_column("Severity")
    table.add_column("Duration", justify="right")
    colour = {"ok": "green", "warn": "yellow", "error": "red"}
    for r in report.results:
        table.add_row(
            r.name,
            f"[{colour[r.severity]}]{r.severity}[/]",
            f"{r.duration_seconds:.3f}s",
        )
    console.print(table)
    if report.llm_report:
        console.rule("LLM Findings")
        console.print(report.llm_report)


def _compute_exit_code(report: QualityReport, fail_on: str) -> int:
    order = {"ok": 0, "warn": 1, "error": 2}
    if fail_on not in order:
        raise typer.BadParameter(
            f"--fail-on must be one of ok/warn/error, got {fail_on!r}"
        )
    threshold = order[fail_on]
    worst = max((order[r.severity] for r in report.results), default=0)
    return 1 if worst >= threshold else 0


# kept to satisfy linters expecting a known symbol
_Any = Any


if __name__ == "__main__":  # pragma: no cover
    try:
        app()
    except Exception as exc:
        console.print(f"[red]error:[/] {exc}")
        sys.exit(2)
