"""Render a ``QualityReport`` to markdown for PR comments / terminals."""

from __future__ import annotations

from datapilot.models.results import CheckResult, QualityReport

SEVERITY_BADGE = {
    "ok": "OK",
    "warn": "WARN",
    "error": "FAIL",
}


def render_markdown(report: QualityReport) -> str:
    """Return a self-contained markdown document describing the report."""
    parts: list[str] = []
    parts.append("# Data Quality Report")
    parts.append("")
    parts.append(f"- **Generated**: {report.generated_at.isoformat()}")
    parts.append(f"- **Engine**: `{report.dataset.engine}`")
    parts.append(f"- **Rows**: {report.dataset.row_count:,}")
    parts.append(f"- **Columns**: {report.dataset.column_count}")
    if report.config_hash:
        parts.append(f"- **Config hash**: `{report.config_hash[:12]}`")
    parts.append("")

    parts.append("## Summary")
    parts.append("")
    parts.append("| Check | Severity | Duration (s) |")
    parts.append("|---|---|---|")
    for r in report.results:
        parts.append(
            f"| {r.name} | {SEVERITY_BADGE[r.severity]} "
            f"| {r.duration_seconds:.3f} |"
        )
    parts.append("")

    for r in report.results:
        parts.append(f"## {r.name}")
        parts.append("")
        parts.append(f"- severity: **{r.severity}**")
        parts.append(f"- duration: {r.duration_seconds:.3f}s")
        if r.error:
            parts.append(f"- error: `{r.error}`")
        _append_payload_details(parts, r)
        parts.append("")

    if report.llm_report:
        parts.append("## LLM Findings")
        parts.append("")
        parts.append(report.llm_report)
        parts.append("")

    return "\n".join(parts)


def _append_payload_details(parts: list[str], result: CheckResult) -> None:
    """Render per-check payload highlights without dumping everything."""
    payload = result.payload
    if not payload:
        return

    if "total_null_count" in payload:
        parts.append(f"- total nulls: {payload['total_null_count']}")
        parts.append(
            f"- worst column: {payload.get('worst_column_pct', 0):.2f}%"
        )
    if "total_duplicate_rows" in payload:
        parts.append(f"- duplicate rows: {payload['total_duplicate_rows']}")
    if "rollup" in payload:
        rollup = payload["rollup"]
        parts.append("- dtype rollup:")
        for dtype, count in rollup.items():
            parts.append(f"  - `{dtype}`: {count}")
    if "per_column" in payload and isinstance(payload["per_column"], list):
        total = len(payload["per_column"])
        parts.append(f"- columns evaluated: {total}")
