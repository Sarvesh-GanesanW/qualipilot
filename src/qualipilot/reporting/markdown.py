"""Render a ``QualityReport`` to markdown for PR comments / terminals."""

from __future__ import annotations

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

    name = result.name
    if name == "missing_values":
        _missing_section(parts, payload)
    elif name == "duplicates":
        _duplicates_section(parts, payload)
    elif name == "data_types":
        _types_section(parts, payload)
    elif name == "outliers":
        _outliers_section(parts, payload)
    elif name == "ranges":
        _ranges_section(parts, payload)
    elif name == "cardinality":
        _cardinality_section(parts, payload)
    elif name == "freshness":
        _freshness_section(parts, payload)
    elif name == "linkage":
        _linkage_section(parts, payload)


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
            f"| `{c['column']}` | {c['null_count']:,} "
            f"| {c['null_percentage']:.2f}% |"
        )


def _duplicates_section(parts: list[str], payload: dict[str, Any]) -> None:
    parts.append(
        f"- duplicate rows: {payload.get('total_duplicate_rows', 0):,}"
    )
    subset = payload.get("subset")
    if subset:
        parts.append(
            f"- subset checked: {', '.join(f'`{c}`' for c in subset)}"
        )
    sample = payload.get("sample") or []
    if sample:
        keys = list(sample[0].keys())[:5]
        parts.append("")
        parts.append("Sample:")
        parts.append("")
        parts.append("| " + " | ".join(keys) + " |")
        parts.append("|" + "|".join(["---"] * len(keys)) + "|")
        for row in sample[:5]:
            parts.append(
                "| " + " | ".join(str(row.get(k, "")) for k in keys) + " |"
            )


def _types_section(parts: list[str], payload: dict[str, Any]) -> None:
    rollup = payload.get("rollup") or {}
    if rollup:
        parts.append("- dtype rollup:")
        for dtype, count in rollup.items():
            parts.append(f"  - `{dtype}`: {count}")


def _outliers_section(parts: list[str], payload: dict[str, Any]) -> None:
    per_col = payload.get("per_column", [])
    affected = [c for c in per_col if c.get("outlier_count", 0) > 0]
    parts.append(
        f"- numeric columns scanned: {len(per_col)} "
        f"(affected: {len(affected)})"
    )
    if not affected:
        return
    affected.sort(key=lambda c: c["outlier_count"], reverse=True)
    parts.append("")
    parts.append("| Column | Outliers | Bounds (IQR) |")
    parts.append("|---|---:|---|")
    for c in affected[:10]:
        parts.append(
            f"| `{c['column']}` | {c['outlier_count']:,} "
            f"| `[{c['lower_bound']:.2f}, {c['upper_bound']:.2f}]` |"
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
            parts.append(
                f"| `{c['column']}` "
                f"| `[{c['min_allowed']}, {c['max_allowed']}]` "
                f"| {c['violation_count']:,} |"
            )
    for c in notes:
        parts.append(f"- note on `{c['column']}`: {c['note']}")


def _cardinality_section(parts: list[str], payload: dict[str, Any]) -> None:
    per_col = payload.get("per_column", [])
    constants = [c for c in per_col if c.get("distinct_count", 1) <= 1]
    parts.append(f"- columns profiled: {len(per_col)}")
    if constants:
        parts.append(
            "- constant columns: "
            + ", ".join(f"`{c['column']}`" for c in constants)
        )


def _freshness_section(parts: list[str], payload: dict[str, Any]) -> None:
    per_col = payload.get("per_column", [])
    stale = [c for c in per_col if c.get("is_stale")]
    parts.append(f"- columns checked: {len(per_col)} (stale: {len(stale)})")
    for c in stale:
        ts = c.get("max_timestamp", "n/a")
        age = c.get("age_hours")
        if age is not None:
            parts.append(f"- `{c['column']}` last seen {ts} ({age:.1f}h ago)")
        else:
            parts.append(f"- `{c['column']}` has no non-null values")


def _linkage_section(parts: list[str], payload: dict[str, Any]) -> None:
    if payload.get("skipped"):
        parts.append("- skipped (no linkage config supplied)")
        return
    parts.append(f"- candidate pairs: {payload.get('candidate_pairs', 0):,}")
    parts.append(f"- matched pairs: {payload.get('matched_pairs', 0):,}")
    parts.append(
        f"- duplicate clusters: {payload.get('duplicate_clusters', 0)}"
    )
    parts.append(
        "- records in duplicate groups: "
        f"{payload.get('records_in_duplicate_groups', 0)}"
    )
