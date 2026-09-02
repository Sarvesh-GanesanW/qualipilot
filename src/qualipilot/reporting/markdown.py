"""Render a ``QualityReport`` to markdown for PR comments / terminals."""

from __future__ import annotations

import html
import json
from collections.abc import Callable
from typing import Any

from qualipilot.models.results import CheckResult, QualityReport

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
    parts.append(f"- **Engine**: {_inline_code(report.dataset.engine)}")
    parts.append(f"- **Rows**: {report.dataset.row_count:,}")
    parts.append(f"- **Columns**: {report.dataset.column_count}")
    if report.dataset.source:
        parts.append(f"- **Source**: {_inline_code(report.dataset.source)}")
    if report.config_hash:
        parts.append(
            f"- **Config hash**: {_inline_code(report.config_hash[:12])}"
        )
    parts.append("")

    parts.append("## Summary")
    parts.append("")
    parts.append("| Check | Severity | Duration (s) |")
    parts.append("|---|---|---|")
    for r in report.results:
        parts.append(
            f"| {_markdown_text(r.name)} | {SEVERITY_BADGE[r.severity]} "
            f"| {r.duration_seconds:.3f} |"
        )
    parts.append("")

    for r in report.results:
        parts.append(f"## {_markdown_text(r.name)}")
        parts.append("")
        parts.append(f"- severity: **{r.severity}**")
        parts.append(f"- status: **{r.status}**")
        parts.append(f"- duration: {r.duration_seconds:.3f}s")
        if r.error:
            parts.append(f"- error: {_inline_code(r.error)}")
        _append_payload_details(parts, r)
        parts.append("")

    if report.llm_report:
        parts.append("## LLM Findings")
        parts.append("")
        parts.append("AI-generated advisory; validate before acting.")
        if report.llm_provider:
            parts.append(f"- provider: {_inline_code(report.llm_provider)}")
        if report.llm_model:
            parts.append(f"- model: {_inline_code(report.llm_model)}")
        parts.append("")
        parts.extend(
            f"    {html.escape(line, quote=False)}"
            for line in report.llm_report.splitlines()
        )
        parts.append("")
    elif report.llm_error:
        parts.append("## LLM Failure")
        parts.append("")
        parts.append(_inline_code(report.llm_error))
        parts.append("")

    return "\n".join(parts)


def _append_payload_details(parts: list[str], result: CheckResult) -> None:
    """Render per-check payload highlights without dumping everything."""
    payload = result.payload
    if not payload:
        return

    section = _PAYLOAD_SECTIONS.get(result.name)
    if section:
        section(parts, payload)

    parts.append("")
    parts.append("Raw payload:")
    parts.append("")
    parts.extend(
        f"    {_markdown_text(line)}"
        for line in json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ).splitlines()
    )


def _missing_section(parts: list[str], payload: dict[str, Any]) -> None:
    parts.append(f"- total nulls: {payload.get('total_null_count', 0):,}")
    parts.append(f"- worst column: {payload.get('worst_column_pct', 0):.2f}%")
    affected = [
        c for c in payload.get("per_column", []) if c.get("null_count", 0) > 0
    ]
    if not affected:
        parts.append("- columns with nulls: none")
        return
    affected.sort(key=lambda c: c["null_count"], reverse=True)
    parts.append("")
    parts.append("| Column | Nulls | Percent |")
    parts.append("|---|---:|---:|")
    for c in affected[:10]:
        parts.append(
            f"| {_inline_code(c['column'])} | {c['null_count']:,} "
            f"| {c['null_percentage']:.2f}% |"
        )


def _duplicates_section(parts: list[str], payload: dict[str, Any]) -> None:
    parts.append(
        f"- duplicate rows: {payload.get('total_duplicate_rows', 0):,}"
    )
    subset = payload.get("subset")
    if subset:
        parts.append(
            "- subset checked: "
            + ", ".join(_inline_code(column) for column in subset)
        )
    sample = payload.get("sample") or []
    if sample:
        keys = list(sample[0].keys())[:5]
        parts.append("")
        parts.append("Sample:")
        parts.append("")
        parts.append("| " + " | ".join(map(_markdown_text, keys)) + " |")
        parts.append("|" + "|".join(["---"] * len(keys)) + "|")
        for row in sample[:5]:
            parts.append(
                "| "
                + " | ".join(_markdown_text(row.get(k, "")) for k in keys)
                + " |"
            )


def _types_section(parts: list[str], payload: dict[str, Any]) -> None:
    rollup = payload.get("rollup") or {}
    if rollup:
        parts.append("- dtype rollup:")
        for dtype, count in rollup.items():
            parts.append(f"  - {_inline_code(dtype)}: {count}")


def _outliers_section(parts: list[str], payload: dict[str, Any]) -> None:
    per_col = payload.get("per_column", [])
    affected = [c for c in per_col if c.get("outlier_count", 0) > 0]
    skipped = [c for c in per_col if c.get("skipped")]
    parts.append(
        f"- numeric columns scanned: {len(per_col)} "
        f"(affected: {len(affected)}, skipped: {len(skipped)})"
    )
    if affected:
        affected.sort(key=lambda c: c["outlier_count"], reverse=True)
        parts.append("")
        parts.append("| Column | Outliers | Bounds (IQR) |")
        parts.append("|---|---:|---|")
        for c in affected[:10]:
            bounds = f"[{c['lower_bound']:.2f}, {c['upper_bound']:.2f}]"
            parts.append(
                f"| {_inline_code(c['column'])} | {c['outlier_count']:,} "
                f"| {_inline_code(bounds)} |"
            )
    for c in skipped:
        parts.append(
            f"- skipped {_inline_code(c['column'])}: "
            f"{_markdown_text(c['skipped'])}"
        )


def _ranges_section(parts: list[str], payload: dict[str, Any]) -> None:
    per_col = payload.get("per_column", [])
    affected = [c for c in per_col if c.get("violation_count", 0) > 0]
    notes = [c for c in per_col if c.get("note")]
    parts.append(f"- ranges configured: {len(per_col)}")
    if affected:
        parts.append("")
        parts.append("| Column | Allowed | Violations |")
        parts.append("|---|---|---:|")
        for c in affected:
            bounds = f"[{c['min_allowed']}, {c['max_allowed']}]"
            parts.append(
                f"| {_inline_code(c['column'])} "
                f"| {_inline_code(bounds)} "
                f"| {c['violation_count']:,} |"
            )
    for c in notes:
        parts.append(
            f"- note on {_inline_code(c['column'])}: "
            f"{_markdown_text(c['note'])}"
        )


def _cardinality_section(parts: list[str], payload: dict[str, Any]) -> None:
    per_col = payload.get("per_column", [])
    constants = [c for c in per_col if c.get("distinct_count", 1) <= 1]
    parts.append(f"- columns profiled: {len(per_col)}")
    if constants:
        parts.append(
            "- constant columns: "
            + ", ".join(_inline_code(c["column"]) for c in constants)
        )


def _freshness_section(parts: list[str], payload: dict[str, Any]) -> None:
    per_col = payload.get("per_column", [])
    invalid = [c for c in per_col if c.get("is_stale") or c.get("is_future")]
    parts.append(
        f"- columns checked: {len(per_col)} "
        f"(invalid freshness: {len(invalid)})"
    )
    for c in invalid:
        ts = c.get("max_timestamp", "n/a")
        age = c.get("age_hours")
        if age is not None:
            timing = "in the future" if c.get("is_future") else "ago"
            parts.append(
                f"- {_inline_code(c['column'])} last seen "
                f"{_markdown_text(ts)} ({abs(age):.1f}h {timing})"
            )
        else:
            parts.append(
                f"- {_inline_code(c['column'])} has no non-null values"
            )


def _linkage_section(parts: list[str], payload: dict[str, Any]) -> None:
    if payload.get("skipped"):
        parts.append("- skipped (no linkage config supplied)")
        return
    parts.append(f"- candidate pairs: {payload.get('candidate_pairs', 0):,}")
    parts.append(f"- matched pairs: {payload.get('matched_pairs', 0):,}")
    parts.append(
        f"- match threshold: {payload.get('match_threshold_probability', 0.9)}"
    )
    parts.append(
        f"- duplicate clusters: {payload.get('duplicate_clusters', 0)}"
    )
    parts.append(
        "- records in duplicate groups: "
        f"{payload.get('records_in_duplicate_groups', 0)}"
    )


def _dataset_contract_section(
    parts: list[str], payload: dict[str, Any]
) -> None:
    parts.append(
        f"- rows: {payload.get('row_count', 0):,} "
        f"(minimum: {payload.get('min_rows', 0):,})"
    )
    missing = payload.get("missing_required_columns") or []
    if missing:
        parts.append(
            "- missing required columns: "
            + ", ".join(_inline_code(column) for column in missing)
        )
    mismatches = payload.get("dtype_mismatches") or []
    for mismatch in mismatches:
        parts.append(
            f"- {_inline_code(mismatch['column'])}: expected "
            f"{_inline_code(mismatch['expected'])}, got "
            f"{_inline_code(mismatch['actual'])}"
        )


def _markdown_text(value: Any) -> str:
    return (
        html.escape(str(value), quote=False)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("!", "\\!")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace("(", "\\(")
        .replace(")", "\\)")
        .replace("\r", "")
        .replace("\n", "&#10;")
    )


def _inline_code(value: Any) -> str:
    return f"`{_markdown_text(value).replace('`', '&#96;')}`"


_PAYLOAD_SECTIONS: dict[str, Callable[[list[str], dict[str, Any]], None]] = {
    "dataset_contract": _dataset_contract_section,
    "missing_values": _missing_section,
    "duplicates": _duplicates_section,
    "data_types": _types_section,
    "outliers": _outliers_section,
    "ranges": _ranges_section,
    "cardinality": _cardinality_section,
    "freshness": _freshness_section,
    "linkage": _linkage_section,
}
