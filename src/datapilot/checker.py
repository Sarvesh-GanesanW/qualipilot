"""Orchestrator tying engines, checks and LLM reporting together."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from datapilot.checks import (
    CardinalityCheck,
    Check,
    CheckContext,
    DataTypesCheck,
    DuplicatesCheck,
    FreshnessCheck,
    MissingValuesCheck,
    OutliersCheck,
    RangesCheck,
)
from datapilot.engines import build_engine
from datapilot.models.config import DatapilotConfig
from datapilot.models.results import (
    CheckResult,
    DatasetStats,
    QualityReport,
)

if TYPE_CHECKING:
    from datapilot.engines.base import Engine
    from datapilot.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class DataQualityChecker:
    """Run configurable data quality checks against a dataframe/file.

    Example:
        >>> import pandas as pd
        >>> from datapilot import DataQualityChecker, DatapilotConfig
        >>> df = pd.read_csv("orders.csv")
        >>> checker = DataQualityChecker(df, DatapilotConfig())
        >>> report = checker.run()
        >>> print(report.to_json())
    """

    def __init__(
        self,
        data: Any,
        config: DatapilotConfig | None = None,
    ) -> None:
        self.config = config or DatapilotConfig()
        self.engine: Engine = build_engine(
            data, kind=self.config.engine
        )
        logger.info(
            "initialised checker with %s engine over %d rows",
            self.engine.name,
            self.engine.row_count(),
        )

    def run(self) -> QualityReport:
        """Run every enabled check and return the aggregate report."""
        ctx = CheckContext(
            engine=self.engine, config=self.config.checks
        )
        results: list[CheckResult] = []
        for check in self._build_check_list():
            logger.info("running check: %s", check.name)
            result = check.run(ctx)
            logger.info(
                "check %s finished in %.3fs (severity=%s)",
                result.name,
                result.duration_seconds,
                result.severity,
            )
            results.append(result)

        dataset = DatasetStats(
            row_count=self.engine.row_count(),
            column_count=len(self.engine.columns()),
            columns=self.engine.columns(),
            dtypes=self.engine.dtypes(),
            engine=self.engine.name,
        )

        report = QualityReport(
            dataset=dataset,
            results=results,
            config_hash=_config_fingerprint(self.config),
        )

        llm_report = self._maybe_render_llm_report(report)
        if llm_report:
            report.llm_report = llm_report

        if self.config.output_path is not None:
            self.save(report, self.config.output_path)

        return report

    # ---- helpers ------------------------------------------------------

    def _build_check_list(self) -> list[Check]:
        cfg = self.config.checks
        checks: list[Check] = []
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
        return checks

    def _maybe_render_llm_report(
        self, report: QualityReport
    ) -> str | None:
        provider_name = self.config.llm.provider
        if provider_name == "none":
            return None
        provider = _build_llm_provider(self.config.llm)
        prompt = _build_llm_prompt(report)
        try:
            return provider.generate(
                system=self.config.llm.system_prompt, user=prompt
            )
        except Exception as exc:
            # llm problems should not fail the check pipeline
            logger.error("llm report generation failed: %s", exc)
            return f"LLM report failed: {exc}"

    @staticmethod
    def save(report: QualityReport, path: str | Path) -> None:
        """Persist the JSON report to disk (utf-8, pretty-printed)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report.to_json(), encoding="utf-8")
        logger.info("report written to %s", target)


def _config_fingerprint(cfg: DatapilotConfig) -> str:
    """Stable SHA-1 of the config, handy for report dedup in pipelines."""
    payload = cfg.model_dump_json(exclude={"output_path"})
    return hashlib.sha1(
        payload.encode("utf-8"), usedforsecurity=False
    ).hexdigest()


def _build_llm_provider(cfg: Any) -> LLMProvider:
    # late import so the optional boto3/openai deps stay optional
    from datapilot.llm import build_provider

    return build_provider(cfg)


def _build_llm_prompt(report: QualityReport) -> str:
    # compact the report before shipping to the model so we do not
    # waste tokens on huge sample arrays
    compact = {
        "dataset": report.dataset.model_dump(),
        "results": [
            {
                "name": r.name,
                "severity": r.severity,
                "duration_seconds": round(r.duration_seconds, 3),
                "summary": _summarise_payload(r.payload),
                "error": r.error,
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
    """Strip large sample arrays so prompts stay cheap."""
    keep: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, list):
            # keep length only, drop row-level samples
            keep[key] = {"count": len(value)}
        else:
            keep[key] = value
    return keep
