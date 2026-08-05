"""Typer-based CLI.

Goal: one command (`qualipilot check`) should take a CSV/Parquet/etc.
and produce a machine-readable report plus optional LLM narrative,
without editing any Python files.
"""

from __future__ import annotations

import enum
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
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


class EngineChoice(enum.StrEnum):
    """CLI-accepted engines. Validated by Typer; typos error cleanly."""

    auto = "auto"
    polars = "polars"
    pandas = "pandas"
    duckdb = "duckdb"
    dask = "dask"
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


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"qualipilot {__version__}")
        raise typer.Exit


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
    try:
        configure_logging(level=log_level, json_logs=json_logs)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--log-level") from exc


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
            help="CSV/Parquet/JSONL/NDJSON file to inspect.",
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
        EngineChoice | None,
        typer.Option(
            "--engine",
            "-e",
            help="Dataframe backend.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Write the report to this path (json/html/md).",
        ),
    ] = None,
    report_format: Annotated[
        FormatChoice | None,
        typer.Option(
            "--format",
            "-f",
            help=(
                "Report format. Auto-derived from --output suffix when known."
            ),
        ),
    ] = None,
    llm_provider: Annotated[
        LLMChoice | None,
        typer.Option(
            "--llm",
            help=(
                "LLM provider for the narrative report. With anything "
                "other than 'none' you will usually also want --model."
            ),
        ),
    ] = None,
    llm_model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help=("Required model id/name for the chosen LLM provider."),
        ),
    ] = None,
    bedrock_region: Annotated[
        str | None,
        typer.Option("--region", envvar="AWS_REGION"),
    ] = None,
    aws_profile: Annotated[
        str | None,
        typer.Option("--profile", envvar="AWS_PROFILE"),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option("--base-url"),
    ] = None,
    allow_insecure_http: Annotated[
        bool | None,
        typer.Option(
            "--allow-insecure-http/--require-https",
            help="Allow a non-TLS remote HTTP endpoint.",
        ),
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
    """Run data quality checks against CSV, Parquet, JSONL, or NDJSON."""
    cfg = _build_config(
        config=config,
        engine=engine,
        report_format=report_format,
        llm_provider=llm_provider,
        llm_model=llm_model,
        bedrock_region=bedrock_region,
        aws_profile=aws_profile,
        base_url=base_url,
        allow_insecure_http=allow_insecure_http,
        range_spec=range_spec,
    )

    if output is not None:
        cfg.output_path = output
    with DataQualityChecker(input_path, cfg) as checker:
        report = checker.run()

    if cfg.output_path is not None:
        console.print(f"report written to {cfg.output_path}", markup=False)
    _print_summary(report)

    exit_code = _compute_exit_code(report, fail_on)
    raise typer.Exit(code=exit_code)


# ---- helpers ------------------------------------------------------------


def _build_config(
    *,
    config: Path | None,
    engine: EngineChoice | None,
    report_format: FormatChoice | None,
    llm_provider: LLMChoice | None,
    llm_model: str | None,
    bedrock_region: str | None,
    aws_profile: str | None,
    base_url: str | None,
    allow_insecure_http: bool | None,
    range_spec: list[str] | None,
) -> QualipilotConfig:
    """Merge CLI flags over file and environment configuration."""
    try:
        cfg = (
            QualipilotConfig.from_file(config)
            if config
            else QualipilotConfig()
        )
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(
            f"invalid configuration: {exc}", param_hint="--config"
        ) from exc

    # cli flags win over file/env unless flag is still at its default
    if engine is not None:
        cfg.engine = cast(EngineName, engine.value)
    if report_format is not None:
        cfg.report_format = cast(ReportFormat, report_format.value)

    provider_value = (
        cast(LLMProvider, llm_provider.value)
        if llm_provider is not None
        else None
    )
    supplied_llm_values = {
        "provider": provider_value,
        "model": llm_model,
        "region": bedrock_region,
        "aws_profile": aws_profile,
        "base_url": base_url,
        "allow_insecure_http": allow_insecure_http,
    }
    llm_updates: dict[str, Any] = {
        name: value
        for name, value in supplied_llm_values.items()
        if value is not None
    }
    if (
        provider_value is not None
        and provider_value != cfg.llm.provider
        and llm_model is None
    ):
        llm_updates["model"] = ""
    if provider_value is not None:
        llm_updates["connection_name"] = None
    if llm_updates:
        current_llm = {
            field_name: getattr(cfg.llm, field_name)
            for field_name in LLMConfig.model_fields
        }
        try:
            cfg.llm = LLMConfig.model_validate(current_llm | llm_updates)
        except ValueError as exc:
            raise typer.BadParameter(
                str(exc), param_hint="--llm/--model"
            ) from exc

    if range_spec:
        ranges = _parse_ranges(range_spec)
        merged = dict(cfg.checks.column_ranges)
        merged.update(ranges)
        cfg.checks = cfg.checks.model_copy(update={"column_ranges": merged})

    return cfg


def _parse_ranges(specs: list[str]) -> dict[str, ColumnRange]:
    out: dict[str, ColumnRange] = {}
    seen: set[str] = set()
    for raw in specs:
        if "=" not in raw or "," not in raw:
            raise typer.BadParameter(
                f"--range expects 'col=min,max', got {raw!r}"
            )
        col, bounds = raw.split("=", 1)
        lo_s, hi_s = bounds.split(",", 1)
        column = col.strip()
        if not column:
            raise typer.BadParameter("--range column must not be empty")
        normalized = column.casefold()
        if normalized in seen:
            raise typer.BadParameter(f"--range repeats column {column!r}")
        seen.add(normalized)
        try:
            out[column] = ColumnRange(min=float(lo_s), max=float(hi_s))
        except ValueError as exc:
            raise typer.BadParameter(
                f"invalid --range {raw!r}: {exc}"
            ) from exc
    return out


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
        console.print(report.llm_report, markup=False)
    elif report.llm_error:
        console.print(
            f"LLM report failed: {report.llm_error}",
            style="red",
            markup=False,
        )


def _compute_exit_code(report: QualityReport, fail_on: SeverityChoice) -> int:
    if report.llm_status == "failed" or any(
        result.status == "failed" for result in report.results
    ):
        return 2
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
            help="CSV, JSONL, NDJSON, or Parquet file to dedupe.",
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
    normalize_strings: Annotated[
        bool,
        typer.Option(
            "--normalize-strings/--raw-strings",
            help=(
                "Normalize Unicode, case, and whitespace in string match "
                "keys and consolidated output."
            ),
        ),
    ] = True,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Write linkage result JSON here.",
        ),
    ] = None,
    deduplicated_output: Annotated[
        Path | None,
        typer.Option(
            "--deduplicated-output",
            help="Write consolidated CSV, Parquet, JSONL, or NDJSON here.",
        ),
    ] = None,
    survivor_sort: Annotated[
        list[str] | None,
        typer.Option(
            "--survivor-sort",
            help='Repeatable survivor key in the form "column:asc|desc".',
        ),
    ] = None,
    completeness: Annotated[
        list[str] | None,
        typer.Option(
            "--completeness",
            help=(
                "Repeatable field counted when ranking survivors. "
                "Defaults to every non-ID field."
            ),
        ),
    ] = None,
    rank_by_completeness: Annotated[
        bool,
        typer.Option(
            "--rank-by-completeness/--no-completeness",
            help="Use populated fields when ranking cluster survivors.",
        ),
    ] = True,
    merge: Annotated[
        list[str] | None,
        typer.Option(
            "--merge",
            help=(
                'Repeatable "column:strategy[:order_by]". Strategies: '
                "survivor, first_non_null, most_frequent, latest."
            ),
        ),
    ] = None,
) -> None:
    """Find duplicate records and optionally consolidate each cluster."""
    from qualipilot.linking import (
        RecordLinker,
    )

    if not compare:
        raise typer.BadParameter("at least one --compare spec is required")
    _validate_link_outputs(input_path, output, deduplicated_output)

    input_hash = _sha256_file(input_path) if output is not None else None
    df = _read_any(input_path)
    cfg = _build_link_config(
        df,
        id_column=id_column,
        compare=compare,
        block=block or [],
        threshold=threshold,
        normalize_strings=normalize_strings,
    )

    linker = RecordLinker(df, cfg)
    result = None
    consolidation_result = None
    consolidation_config = None
    if deduplicated_output is not None:
        consolidation_config = _build_consolidation_config(
            df,
            id_column=id_column,
            survivor_sort=survivor_sort or [],
            completeness=completeness,
            rank_by_completeness=rank_by_completeness,
            merge=merge or [],
        )
        deduplication = linker.deduplicate(consolidation_config)
        result = deduplication.linkage
        consolidation_result = deduplication.consolidation
    else:
        result = linker.run()
    if input_hash is not None and _sha256_file(input_path) != input_hash:
        raise typer.BadParameter(
            "input changed during linkage; rerun the command"
        )

    summary = result.summary()
    if consolidation_result is not None:
        summary["consolidation"] = consolidation_result.summary()
    console.print_json(data=summary)

    if output is not None:
        if input_hash is None:  # pragma: no cover - narrows the optional type
            raise RuntimeError("missing input hash")
        config_data = cfg.model_dump(mode="json")
        consolidation_data = (
            consolidation_config.model_dump(mode="json")
            if consolidation_config is not None
            else None
        )
        fingerprint_data = {
            "linkage": config_data,
            "consolidation": consolidation_data,
        }
        config_json = json.dumps(
            fingerprint_data,
            sort_keys=True,
            separators=(",", ":"),
        )
        payload = {
            "schema_version": "1.0",
            "generated_at": datetime.now(UTC).isoformat(),
            "package_version": __version__,
            "source": str(input_path.resolve()),
            "source_sha256": input_hash,
            "config": config_data,
            "config_hash": hashlib.sha256(config_json.encode()).hexdigest(),
            "summary": summary,
            "parameters": result.parameters,
        }
        if consolidation_result is not None:
            payload["consolidation"] = {
                "config": consolidation_data,
                "lineage": {
                    str(source_id): survivor_id
                    for source_id, survivor_id in (
                        consolidation_result.lineage.items()
                    )
                },
                "audit": [
                    asdict(entry) for entry in consolidation_result.audit
                ],
            }
            if deduplicated_output is None:  # pragma: no cover - invariant
                raise RuntimeError("missing deduplicated output path")
            _write_deduplication_outputs(
                output,
                payload,
                matched_pairs=result.match_pairs(threshold),
                clusters=result.clusters,
                data_path=deduplicated_output,
                frame=consolidation_result.frame,
            )
        else:
            _write_linkage_json_atomic(
                output,
                payload,
                matched_pairs=result.match_pairs(threshold),
                clusters=result.clusters,
            )
        console.print(f"linkage written to {output}", markup=False)
        if deduplicated_output is not None:
            console.print(
                f"deduplicated data written to {deduplicated_output}",
                markup=False,
            )


def _validate_link_outputs(
    input_path: Path,
    output: Path | None,
    deduplicated_output: Path | None,
) -> None:
    targets = [path for path in (output, deduplicated_output) if path]
    if any(input_path.resolve() == path.resolve() for path in targets):
        raise typer.BadParameter("output paths must not overwrite the input")
    if len({path.resolve() for path in targets}) != len(targets):
        raise typer.BadParameter("output paths must be different")
    if deduplicated_output is not None and output is None:
        raise typer.BadParameter(
            "--deduplicated-output requires --output for its audit"
        )
    if deduplicated_output is not None:
        _validate_frame_output_suffix(deduplicated_output)


def _build_link_config(
    frame: pl.DataFrame,
    *,
    id_column: str,
    compare: list[str],
    block: list[str],
    threshold: float,
    normalize_strings: bool,
) -> Any:
    from qualipilot.linking import (
        LinkConfig,
        StringNormalization,
    )

    try:
        comparisons = [_parse_compare(spec) for spec in compare]
        blocking_rules = [
            [column.strip() for column in raw.split(",") if column.strip()]
            for raw in block
        ]
        string_types = {pl.String, pl.Categorical, pl.Enum}
        normalization = (
            {
                column: StringNormalization()
                for column, dtype in frame.schema.items()
                if column != id_column and dtype.base_type() in string_types
            }
            if normalize_strings
            else {}
        )
        return LinkConfig(
            mode="dedupe",
            unique_id_column=id_column,
            comparisons=comparisons,
            blocking_rules=blocking_rules,
            normalization=normalization,
            match_threshold_probability=threshold,
        )
    except ValueError as exc:
        raise typer.BadParameter(
            f"invalid linkage configuration: {exc}"
        ) from exc


def _build_consolidation_config(
    frame: pl.DataFrame,
    *,
    id_column: str,
    survivor_sort: list[str],
    completeness: list[str] | None,
    rank_by_completeness: bool,
    merge: list[str],
) -> Any:
    from qualipilot.linking import (
        ConsolidationConfig,
        MergeRule,
    )

    merge_rules = {
        column: MergeRule(strategy="first_non_null")
        for column in frame.columns
        if column != id_column
    }
    supplied_rules = [_parse_merge_rule(value) for value in merge]
    supplied_columns = [column for column, _ in supplied_rules]
    if len(set(supplied_columns)) != len(supplied_columns):
        raise typer.BadParameter("--merge columns must be unique")
    merge_rules.update(supplied_rules)
    if completeness is not None and not rank_by_completeness:
        raise typer.BadParameter(
            "--completeness cannot be combined with --no-completeness"
        )
    try:
        return ConsolidationConfig(
            sort_keys=tuple(
                _parse_survivor_sort(value) for value in survivor_sort
            ),
            completeness_columns=tuple(
                (
                    completeness
                    if completeness is not None
                    else [
                        column
                        for column in frame.columns
                        if column != id_column
                    ]
                )
                if rank_by_completeness
                else ()
            ),
            merge_rules=merge_rules,
        )
    except ValueError as exc:
        raise typer.BadParameter(
            f"invalid consolidation configuration: {exc}"
        ) from exc


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


def _parse_survivor_sort(raw: str) -> Any:
    from qualipilot.linking import SurvivorSortKey

    parts = raw.rsplit(":", 1)
    if len(parts) != 2 or parts[1] not in {"asc", "desc"}:
        raise typer.BadParameter(
            f"--survivor-sort expects 'column:asc|desc', got {raw!r}"
        )
    try:
        return SurvivorSortKey(
            column=parts[0],
            descending=parts[1] == "desc",
        )
    except ValueError as exc:
        raise typer.BadParameter(
            f"invalid --survivor-sort {raw!r}: {exc}"
        ) from exc


def _parse_merge_rule(raw: str) -> tuple[str, Any]:
    from qualipilot.linking import MergeRule

    parts = raw.split(":")
    if len(parts) not in {2, 3} or not parts[0].strip():
        raise typer.BadParameter(
            f"--merge expects 'column:strategy[:order_by]', got {raw!r}"
        )
    column, strategy = parts[:2]
    if strategy not in {
        "survivor",
        "first_non_null",
        "most_frequent",
        "latest",
    }:
        raise typer.BadParameter(f"unknown --merge strategy: {strategy!r}")
    if (strategy == "latest") != (len(parts) == 3):
        raise typer.BadParameter(
            "latest merge rules require an order_by column; "
            "other strategies do not accept one"
        )
    try:
        return (
            column.strip(),
            MergeRule(
                strategy=cast(Any, strategy),
                order_by=parts[2] if len(parts) == 3 else None,
            ),
        )
    except ValueError as exc:
        raise typer.BadParameter(f"invalid --merge {raw!r}: {exc}") from exc


def _json_default(value: Any) -> Any:
    to_list = getattr(value, "tolist", None)
    return to_list() if callable(to_list) else str(value)


def _sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _write_linkage_json_atomic(
    path: Path,
    payload: dict[str, Any],
    *,
    matched_pairs: pl.DataFrame,
    clusters: dict[object, int],
) -> None:
    """Stream a linkage result without materializing every matched row."""
    temp_path = _stage_linkage_json(
        path,
        payload,
        matched_pairs=matched_pairs,
        clusters=clusters,
    )
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _stage_linkage_json(
    path: Path,
    payload: dict[str, Any],
    *,
    matched_pairs: pl.DataFrame,
    clusters: dict[object, int],
) -> Path:
    target = path
    target.parent.mkdir(parents=True, exist_ok=True)
    encoder = json.JSONEncoder(
        default=_json_default,
        allow_nan=False,
        separators=(",", ":"),
    )
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            stream.write("{")
            for index, (key, value) in enumerate(payload.items()):
                if index:
                    stream.write(",")
                stream.writelines(encoder.iterencode(key))
                stream.write(":")
                stream.writelines(encoder.iterencode(value))

            if payload:
                stream.write(",")
            stream.writelines(encoder.iterencode("matched_pairs"))
            stream.write(":[")
            for index, row in enumerate(matched_pairs.iter_rows(named=True)):
                if index:
                    stream.write(",")
                stream.writelines(encoder.iterencode(row))
            stream.write("],")
            stream.writelines(encoder.iterencode("clusters"))
            stream.write(":{")
            for index, (record_id, cluster_id) in enumerate(clusters.items()):
                if index:
                    stream.write(",")
                stream.writelines(encoder.iterencode(str(record_id)))
                stream.write(":")
                stream.writelines(encoder.iterencode(cluster_id))
            stream.write("}}\n")
        return temp_path
    finally:
        if temp_path is not None and sys.exc_info()[0] is not None:
            temp_path.unlink(missing_ok=True)


def _validate_frame_output_suffix(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".parquet", ".pq", ".jsonl", ".ndjson"}:
        raise typer.BadParameter(
            f"unsupported deduplicated output type: {suffix or '(none)'}"
        )


def _stage_frame(path: Path, frame: pl.DataFrame) -> Path:
    _validate_frame_output_suffix(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            suffix=path.suffix,
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            frame.write_csv(temp_path)
        elif suffix in {".parquet", ".pq"}:
            frame.write_parquet(temp_path)
        else:
            frame.write_ndjson(temp_path)
        return temp_path
    finally:
        if temp_path is not None and sys.exc_info()[0] is not None:
            temp_path.unlink(missing_ok=True)


def _write_deduplication_outputs(
    audit_path: Path,
    payload: dict[str, Any],
    *,
    matched_pairs: pl.DataFrame,
    clusters: dict[object, int],
    data_path: Path,
    frame: pl.DataFrame,
) -> None:
    data_temp: Path | None = None
    audit_temp: Path | None = None
    data_backup: Path | None = None
    audit_backup: Path | None = None
    data_published = False
    audit_published = False
    try:
        data_temp = _stage_frame(data_path, frame)
        consolidation = payload["consolidation"]
        if not isinstance(consolidation, dict):  # pragma: no cover - invariant
            raise RuntimeError("missing consolidation audit")
        consolidation["output"] = {
            "path": str(data_path.resolve()),
            "sha256": _sha256_file(data_temp),
        }
        audit_temp = _stage_linkage_json(
            audit_path,
            payload,
            matched_pairs=matched_pairs,
            clusters=clusters,
        )
        data_backup = _backup_file(data_path)
        audit_backup = _backup_file(audit_path)
        os.replace(data_temp, data_path)
        data_temp = None
        data_published = True
        os.replace(audit_temp, audit_path)
        audit_temp = None
        audit_published = True
    except Exception:
        if audit_published:
            _restore_file(audit_path, audit_backup)
            audit_backup = None
        if data_published:
            _restore_file(data_path, data_backup)
            data_backup = None
        raise
    finally:
        for temporary in (
            audit_temp,
            data_temp,
            audit_backup,
            data_backup,
        ):
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def _backup_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError(f"output path is not a file: {path}")
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        delete=False,
    ) as stream:
        backup = Path(stream.name)
    backup.unlink()
    try:
        os.link(path, backup)
    except OSError:
        shutil.copy2(path, backup)
    return backup


def _restore_file(path: Path, backup: Path | None) -> None:
    if backup is None:
        path.unlink(missing_ok=True)
    else:
        os.replace(backup, path)


def _read_any(path: Path) -> pl.DataFrame:
    """Minimal reader used by the link subcommand."""
    from qualipilot.engines._file_formats import (
        require_unique_csv_columns,
        require_valid_json_lines,
    )

    suffix = path.suffix.lower()
    if suffix == ".csv":
        require_unique_csv_columns(path)
        return pl.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pl.read_parquet(path)
    if suffix in {".ndjson", ".jsonl"}:
        scalar_types = require_valid_json_lines(path)
        dtypes = {
            "string": pl.String,
            "integer": pl.Int64,
            "number": pl.Float64,
            "boolean": pl.Boolean,
        }
        return pl.read_ndjson(
            path,
            schema={
                column: dtypes[family]
                for column, family in scalar_types.items()
            },
            infer_schema_length=None,
        )
    raise typer.BadParameter(f"unsupported file type: {suffix}")


def _run() -> None:  # pragma: no cover
    """Render command failures without exposing a traceback."""
    from rich.markup import escape

    try:
        app()
    except Exception as exc:
        message = str(exc) or type(exc).__name__
        console.print(f"[red]error:[/] {escape(message)}")
        sys.exit(2)


if __name__ == "__main__":  # pragma: no cover
    _run()
