"""Orchestrator tying engines, checks and LLM reporting together."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qualipilot.checks import (
    CardinalityCheck,
    Check,
    CheckContext,
    DatasetContractCheck,
    DataTypesCheck,
    DuplicatesCheck,
    FreshnessCheck,
    LinkageCheck,
    MissingValuesCheck,
    OutliersCheck,
    RangesCheck,
)
from qualipilot.engines import build_engine
from qualipilot.engines._file_formats import safe_source_name
from qualipilot.engines.base import (
    validate_column_names as _validate_column_names,
)
from qualipilot.models.config import QualipilotConfig, ReportFormat
from qualipilot.models.results import (
    CheckResult,
    DatasetStats,
    QualityReport,
)

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection
    from pyspark.sql import SparkSession

    from qualipilot.engines.base import Engine

logger = logging.getLogger(__name__)
_LLM_SUMMARY_LIMIT = 50


class DataQualityChecker:
    """Run configurable data quality checks against a dataframe/file.

    Caller-supplied ``spark_session`` and ``duckdb_connection`` objects bind
    execution to an existing runtime and remain owned by the caller.

    Example:
        >>> import pandas as pd
        >>> from qualipilot import DataQualityChecker, QualipilotConfig
        >>> df = pd.read_csv("orders.csv")
        >>> checker = DataQualityChecker(df, QualipilotConfig())
        >>> report = checker.run()
        >>> print(report.to_json())
    """

    def __init__(
        self,
        data: Any,
        config: QualipilotConfig | None = None,
        *,
        source: str | None = None,
        source_version: str | None = None,
        spark_session: SparkSession | None = None,
        duckdb_connection: DuckDBPyConnection | None = None,
    ) -> None:
        self.config = config or QualipilotConfig()
        if (
            isinstance(data, str | Path)
            and self.config.output_path is not None
            and Path(data).resolve() == self.config.output_path.resolve()
        ):
            raise ValueError("output path must not overwrite the input")
        input_columns = (
            getattr(data, "column_names", None)
            if type(data).__module__.startswith("pyarrow")
            else getattr(
                data,
                "columns",
                getattr(data, "column_names", None),
            )
        )
        if input_columns is not None:
            _validate_column_names(list(input_columns))
        self.engine: Engine = build_engine(
            data,
            kind=self.config.engine,
            spark_session=spark_session,
            duckdb_connection=duckdb_connection,
        )
        if self.config.checks.linkage is not None and self.engine.name not in {
            "polars",
            "pandas",
        }:
            raise ValueError(
                "checks.linkage requires the polars or pandas engine"
            )
        raw_source = (
            source
            if source is not None
            else str(data)
            if isinstance(data, str | Path)
            else None
        )
        self._source = (
            safe_source_name(raw_source) if raw_source is not None else None
        )
        self._source_version = source_version
        logger.info(
            "initialised checker with %s engine",
            self.engine.name,
            extra={"engine": self.engine.name},
        )

    def run(self, *, include_llm: bool = True) -> QualityReport:
        """Run every enabled check and return the aggregate report."""
        row_count = self.engine.row_count()
        columns = self.engine.columns()
        _validate_column_names(columns)
        dtypes = self.engine.dtypes()
        ctx = CheckContext(
            engine=self.engine,
            config=self.config.checks,
            row_count=row_count,
            columns=columns,
            dtypes=dtypes,
        )
        results: list[CheckResult] = []
        for check in self._build_check_list():
            logger.info(
                "running check: %s",
                check.name,
                extra={"check": check.name, "engine": self.engine.name},
            )
            result = check.run(ctx)
            logger.info(
                "check %s finished in %.3fs (severity=%s)",
                result.name,
                result.duration_seconds,
                result.severity,
                extra={
                    "check": result.name,
                    "duration_seconds": result.duration_seconds,
                    "severity": result.severity,
                    "status": result.status,
                },
            )
            results.append(result)

        dataset = DatasetStats(
            row_count=row_count,
            column_count=len(columns),
            columns=columns,
            dtypes=dtypes,
            engine=self.engine.name,
            source=self._source,
            source_version=self._source_version,
        )

        report = QualityReport(
            dataset=dataset,
            results=results,
            config_hash=config_fingerprint(self.config),
        )

        if include_llm:
            self.enrich_with_llm(report)

        if self.config.output_path is not None:
            self.save(
                report,
                self.config.output_path,
                self.config.report_format,
            )

        return report

    def enrich_with_llm(self, report: QualityReport) -> None:
        """Add the configured narrative to an existing quality report."""
        if self.config.llm.provider == "none":
            return
        report.llm_provider = self.config.llm.provider
        report.llm_model = self.config.llm.model
        llm_report, llm_error = self._maybe_render_llm_report(report)
        report.llm_report = llm_report
        report.llm_error = llm_error
        report.llm_status = "failed" if llm_error else "completed"

    def close(self) -> None:
        """Release resources owned by the selected engine."""
        close = getattr(self.engine, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> DataQualityChecker:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ---- helpers ------------------------------------------------------

    def _build_check_list(self) -> list[Check]:
        cfg = self.config.checks
        checks: list[Check] = [DatasetContractCheck()]
        if cfg.missing_values:
            checks.append(MissingValuesCheck())
        if cfg.duplicates:
            checks.append(DuplicatesCheck())
        if cfg.data_types:
            checks.append(DataTypesCheck())
        if cfg.outliers:
            checks.append(OutliersCheck())
        if cfg.ranges:
            checks.append(RangesCheck())
        if cfg.cardinality:
            checks.append(CardinalityCheck())
        if cfg.freshness:
            checks.append(FreshnessCheck())
        if cfg.linkage is not None:
            checks.append(LinkageCheck())
        return checks

    def _maybe_render_llm_report(
        self, report: QualityReport
    ) -> tuple[str | None, str | None]:
        try:
            from qualipilot.llm import build_provider

            provider = build_provider(self.config.llm)
            prompt = _build_llm_prompt(report)
            text = provider.generate(
                system=self.config.llm.system_prompt, user=prompt
            )
            return text, None
        except Exception as exc:
            logger.error("llm report generation failed: %s", exc)
            return None, f"{type(exc).__name__}: {exc}"

    @staticmethod
    def save(
        report: QualityReport,
        path: str | Path,
        report_format: ReportFormat = "json",
    ) -> None:
        """Persist a report atomically in the requested format."""
        target = Path(path)
        effective_format = _infer_report_format(target, report_format)
        if effective_format == "html":
            from qualipilot.reporting import render_html

            payload = render_html(report)
        elif effective_format == "markdown":
            from qualipilot.reporting import render_markdown

            payload = render_markdown(report)
        else:
            payload = report.to_json()

        write_text_atomic(target, payload)
        logger.info(
            "report written to %s",
            target,
            extra={
                "output_path": str(target),
                "report_format": effective_format,
            },
        )


def write_text_atomic(path: str | Path, payload: str) -> None:
    """Replace a text file without exposing partial output."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            delete=False,
        ) as temp_file:
            temp_file.write(payload)
            temp_path = Path(temp_file.name)
        os.replace(temp_path, target)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def config_fingerprint(cfg: QualipilotConfig) -> str:
    """Return a canonical, non-secret SHA-256 config fingerprint."""
    serialisable = cfg.model_dump(
        mode="json",
        exclude={"output_path": True, "llm": {"api_key"}},
    )
    payload = json.dumps(
        serialisable,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_llm_prompt(report: QualityReport) -> str:
    dataset = {
        "row_count": report.dataset.row_count,
        "column_count": report.dataset.column_count,
        "engine": report.dataset.engine,
        "dtypes": dict(list(report.dataset.dtypes.items())[:100]),
    }
    compact = {
        "dataset": dataset,
        "results": [
            {
                "name": r.name,
                "severity": r.severity,
                "status": r.status,
                "duration_seconds": round(r.duration_seconds, 3),
                "summary": _summarise_payload(r.payload),
            }
            for r in report.results
        ],
    }
    return (
        "Analyse this data quality report and produce actionable "
        "findings. Keep samples out; focus on what to fix.\n\n"
        + json.dumps(compact, default=str)
    )


def _summarise_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove row samples while retaining capped actionable findings."""
    filtered = [
        (key, value)
        for key, value in payload.items()
        if key not in {"sample", "top_values"}
    ]
    ordered = sorted(filtered, key=lambda item: not _is_actionable(item[1]))
    items = {
        key: _summarise_value(value)
        for key, value in ordered[:_LLM_SUMMARY_LIMIT]
    }
    if len(filtered) <= _LLM_SUMMARY_LIMIT:
        return items
    return {
        "items": items,
        "total_count": len(filtered),
        "omitted_count": len(filtered) - _LLM_SUMMARY_LIMIT,
    }


def _summarise_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _summarise_payload(value)
    if isinstance(value, list):
        ordered = sorted(value, key=lambda item: not _is_actionable(item))
        return {
            "items": [
                _summarise_value(item) for item in ordered[:_LLM_SUMMARY_LIMIT]
            ],
            "total_count": len(value),
            "omitted_count": max(0, len(value) - _LLM_SUMMARY_LIMIT),
        }
    return value


def _is_actionable(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    count_keys = {
        "null_count",
        "outlier_count",
        "violation_count",
        "duplicate_count",
    }
    has_count = any(
        isinstance(value.get(key), int | float) and value[key] > 0
        for key in count_keys
    )
    has_detail = any(
        value.get(key) for key in ("note", "skipped", "expected", "actual")
    )
    has_freshness_issue = bool(value.get("is_stale") or value.get("is_future"))
    distinct_count = value.get("distinct_count")
    is_constant = (
        isinstance(distinct_count, int | float) and distinct_count <= 1
    )
    return has_count or has_detail or has_freshness_issue or is_constant


def _infer_report_format(path: Path, configured: ReportFormat) -> ReportFormat:
    suffix = path.suffix.lower()
    if suffix == ".html":
        return "html"
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix == ".json":
        return "json"
    return configured
