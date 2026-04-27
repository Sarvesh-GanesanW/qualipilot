"""Typer-based CLI.

Goal: one command (`qualipilot check`) should take a CSV/Parquet/etc.
and produce a machine-readable report plus optional LLM narrative,
without editing any Python files.
"""

from __future__ import annotations

import enum
import sys
from pathlib import Path
from typing import Annotated, Any, cast

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from qualipilot import __version__
from qualipilot.checker import DataQualityChecker
from qualipilot.logging_setup import configure_logging
from qualipilot.models.config import (
    ColumnRange,
    EngineName,
    LLMConfig,
    LLMProvider,
    QualipilotConfig,
    ReportFormat,
)
from qualipilot.models.results import QualityReport
from qualipilot.reporting import render_html, render_markdown


class EngineChoice(enum.StrEnum):
    """CLI-accepted engines. Validated by Typer; typos error cleanly."""

    auto = "auto"
    polars = "polars"
    pandas = "pandas"
    duckdb = "duckdb"
    dask = "dask"
    cudf = "cudf"
    spark = "spark"


class LLMChoice(enum.StrEnum):
    none = "none"
    bedrock = "bedrock"
    ollama = "ollama"
    openai = "openai"


class FormatChoice(enum.StrEnum):
    json = "json"
    html = "html"
    markdown = "markdown"


class SeverityChoice(enum.StrEnum):
    ok = "ok"
    warn = "warn"
    error = "error"


app = typer.Typer(
    name="qualipilot",
    help="Run data quality checks and (optionally) an LLM report.",
    no_args_is_help=True,
    add_completion=True,
    pretty_exceptions_enable=False,
)

console = Console()

CONFIG_FILENAMES = (
    "qualipilot.yaml",
    "qualipilot.yml",
    ".qualipilot.yaml",
    ".qualipilot.yml",
)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"qualipilot {__version__}")
        raise typer.Exit


def _autodiscover_config() -> Path | None:
    """Return the first qualipilot config found in cwd, else None."""
    cwd = Path.cwd()
    for name in CONFIG_FILENAMES:
        candidate = cwd / name
        if candidate.exists():
            return candidate
    return None


@app.callback()
def _root(
    log_level: Annotated[
        str, typer.Option("--log-level", envvar="QUALIPILOT_LOG_LEVEL")
    ] = "WARNING",
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            "-q",
            help="Only show errors. Equivalent to --log-level ERROR.",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show INFO logs. Equivalent to --log-level INFO.",
        ),
    ] = False,
    json_logs: Annotated[
        bool,
        typer.Option(
            "--json-logs/--rich-logs",
            envvar="QUALIPILOT_JSON_LOGS",
        ),
    ] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the installed version and exit.",
        ),
    ] = False,
) -> None:
    """Global options that apply to every sub-command."""
    if quiet and verbose:
        raise typer.BadParameter(
            "--quiet and --verbose are mutually exclusive"
        )
    if quiet:
        log_level = "ERROR"
    elif verbose:
        log_level = "INFO"
    configure_logging(level=log_level, json_logs=json_logs)


@app.command()
def version() -> None:
    """Print the installed package version."""
    console.print(f"qualipilot {__version__}")


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
        EngineChoice,
        typer.Option(
            "--engine",
            "-e",
            help="Dataframe backend.",
        ),
    ] = EngineChoice.auto,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Write the report to this path (json/html/md).",
        ),
    ] = None,
    report_format: Annotated[
        FormatChoice,
        typer.Option(
            "--format",
            "-f",
            help=(
                "Report format. Auto-derived from --output suffix when known."
            ),
        ),
    ] = FormatChoice.json,
    llm_provider: Annotated[
        LLMChoice,
        typer.Option(
            "--llm",
            help=(
                "LLM provider for the narrative report. With anything "
                "other than 'none' you will usually also want --model."
            ),
        ),
    ] = LLMChoice.none,
    llm_model: Annotated[
        str,
        typer.Option(
            "--model",
            help=(
                "Model id/name for the chosen LLM. "
                "Bedrock default is provider.model-3-5-haiku-... "
                "Ollama default depends on what you have pulled."
            ),
        ),
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
        typer.Option("--api-key", envvar="QUALIPILOT_LLM_API_KEY"),
    ] = None,
    range_spec: Annotated[
        list[str] | None,
        typer.Option(
            "--range",
            help='Per-column range: "col=min,max" (repeatable).',
        ),
    ] = None,
    fail_on: Annotated[
        SeverityChoice,
        typer.Option(
            "--fail-on",
            help="Exit non-zero when any check hits this severity.",
        ),
    ] = SeverityChoice.error,
) -> None:
    """Run data quality checks against a CSV/Parquet/JSON/NDJSON file."""
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
    engine: EngineChoice,
    report_format: FormatChoice,
    llm_provider: LLMChoice,
    llm_model: str,
    bedrock_region: str,
    aws_profile: str | None,
    base_url: str,
    api_key: str | None,
    range_spec: list[str] | None,
) -> QualipilotConfig:
    """Merge CLI flags on top of a YAML/JSON base config if supplied.

    If ``config`` is None we look for ``qualipilot.{yaml,yml}`` or
    ``.qualipilot.{yaml,yml}`` in the current directory and use that.
    Pass ``--config /dev/null`` (or any other empty file) to force
    defaults.
    """
    if config is None:
        config = _autodiscover_config()
        if config is not None:
            console.print(
                f"[dim]using config from {config}[/dim]",
            )
    cfg = QualipilotConfig.from_file(config) if config else QualipilotConfig()

    # cli flags win over file/env unless flag is still at its default
    if engine is not EngineChoice.auto:
        cfg.engine = cast(EngineName, engine.value)
    cfg.report_format = cast(ReportFormat, report_format.value)

    if llm_provider is not LLMChoice.none:
        cfg.llm = LLMConfig(
            provider=cast(LLMProvider, llm_provider.value),
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


def _compute_exit_code(report: QualityReport, fail_on: SeverityChoice) -> int:
    order = {"ok": 0, "warn": 1, "error": 2}
    threshold = order[fail_on.value]
    worst = max((order[r.severity] for r in report.results), default=0)
    return 1 if worst >= threshold else 0


@app.command("link")
def link_command(
    input_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            readable=True,
            help="CSV/Parquet file to dedupe.",
        ),
    ],
    id_column: Annotated[
        str,
        typer.Option("--id", help="Unique id column."),
    ] = "id",
    compare: Annotated[
        list[str] | None,
        typer.Option(
            "--compare",
            help=(
                'Repeatable. "<col>:exact" | "<col>:fuzzy:0.92,0.80" '
                '| "<col>:numeric:1.0,5.0"'
            ),
        ),
    ] = None,
    block: Annotated[
        list[str] | None,
        typer.Option(
            "--block",
            help=(
                "Repeatable. Comma-joined column list; records that "
                "agree on every column block together."
            ),
        ),
    ] = None,
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            help="Probability at which pairs count as matches.",
        ),
    ] = 0.9,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Write linkage result JSON here.",
        ),
    ] = None,
) -> None:
    """Run in-house probabilistic record linkage / dedup."""
    from qualipilot.linking import (
        LinkConfig,
        RecordLinker,
    )

    if not compare:
        raise typer.BadParameter("at least one --compare spec is required")

    comparisons = [_parse_compare(spec) for spec in compare]
    blocking_rules = [
        [c.strip() for c in b.split(",") if c.strip()] for b in (block or [])
    ]

    cfg = LinkConfig(
        mode="dedupe",
        unique_id_column=id_column,
        comparisons=comparisons,
        blocking_rules=blocking_rules,
        match_threshold_probability=threshold,
    )

    df = _read_any(input_path)
    result = RecordLinker(df, cfg).run()

    summary = result.summary()
    console.print_json(data=summary)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        import json

        payload = {
            "summary": summary,
            "matched_pairs": (
                result.pairs.filter(
                    pl.col("match_probability") >= threshold
                ).to_dicts()
            ),
            "clusters": {str(k): v for k, v in result.clusters.items()},
        }
        output.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        console.print(f"linkage written to [bold]{output}[/bold]")


def _parse_compare(spec: str) -> Any:
    """Turn ``"name:fuzzy:0.92,0.80"`` into a ComparisonSpec."""
    from qualipilot.linking import ExactMatch, FuzzyString, NumericDiff

    parts = spec.split(":", 2)
    if len(parts) < 2:
        raise typer.BadParameter(f"invalid --compare spec: {spec!r}")
    column, kind = parts[0], parts[1]
    if kind == "exact":
        return ExactMatch(column=column)
    if kind == "fuzzy":
        thresholds = (
            _parse_floats(parts[2]) if len(parts) == 3 else (0.92, 0.80)
        )
        return FuzzyString(column=column, thresholds=thresholds)
    if kind == "numeric":
        thresholds = _parse_floats(parts[2]) if len(parts) == 3 else (1.0, 5.0)
        return NumericDiff(column=column, thresholds=thresholds)
    raise typer.BadParameter(f"unknown comparison kind: {kind!r}")


def _parse_floats(raw: str) -> tuple[float, ...]:
    return tuple(float(x) for x in raw.split(",") if x.strip())


def _read_any(path: Path) -> pl.DataFrame:
    """Minimal reader used by the link subcommand."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pl.read_parquet(path)
    if suffix in {".ndjson", ".jsonl"}:
        return pl.read_ndjson(path)
    raise typer.BadParameter(f"unsupported file type: {suffix}")


def _run() -> None:  # pragma: no cover
    """Entry point with friendly rendering of optional-dependency errors.

    Without this wrapper, missing extras (e.g. running ``qualipilot link``
    without ``[linking]``) surface as a 50-line Rich traceback. The actual
    install command is in the ImportError message itself; we pull it out
    and exit cleanly so users see one line, not a stack. ``escape`` keeps
    ``[bedrock]`` / ``[linking]`` from being parsed as Rich style markup.
    """
    from rich.markup import escape

    try:
        app()
    except ImportError as exc:
        console.print(f"[red]error:[/] {escape(str(exc))}")
        sys.exit(2)


if __name__ == "__main__":  # pragma: no cover
    _run()
