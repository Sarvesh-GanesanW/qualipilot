"""Minimal dependency-free HTML report.

We intentionally avoid Jinja: a static template keeps the package
tiny and the output predictable for CI artefacts.
"""

from __future__ import annotations

import html
import json
from collections.abc import Callable
from typing import Any

from qualipilot.models.results import CheckResult, QualityReport

_STYLES = """
body { font-family: ui-sans-serif, system-ui, sans-serif;
       margin: 2rem; max-width: 900px; line-height: 1.5; color: #1f2937;}
h1, h2 { color: #111827; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
th, td { border: 1px solid #e5e7eb; padding: 0.5rem 0.75rem;
         text-align: left; }
th { background: #f3f4f6; }
.badge { display: inline-block; padding: 0.15rem 0.6rem;
         border-radius: 9999px; font-size: 0.75rem;
         font-weight: 600; text-transform: uppercase; }
.badge-ok    { background: #dcfce7; color: #166534; }
.badge-warn  { background: #fef9c3; color: #854d0e; }
.badge-error { background: #fee2e2; color: #991b1b; }
details { margin: 0.5rem 0; }
code, pre { background: #f3f4f6; border-radius: 4px; padding: 0.1rem 0.3rem; }
pre { padding: 0.75rem; overflow-x: auto; }
.numeric { text-align: right; font-variant-numeric: tabular-nums; }
.muted { color: #6b7280; font-size: 0.85rem; }
"""


def render_html(report: QualityReport) -> str:
    """Return a full HTML document for the given report."""
    rows_html = "\n".join(
        f"<tr><td>{html.escape(r.name)}</td>"
        f"<td><span class='badge badge-{r.severity}'>"
        f"{r.severity}</span></td>"
        f"<td class='numeric'>{r.duration_seconds:.3f}s</td></tr>"
        for r in report.results
    )

    details_html = "\n".join(_render_check_html(r) for r in report.results)

    llm_html = ""
    if report.llm_report:
        provenance = "AI-generated advisory"
        if report.llm_provider:
            provenance += f" by {html.escape(report.llm_provider)}"
        if report.llm_model:
            provenance += f" / {html.escape(report.llm_model)}"
        llm_html = (
            f"<h2>LLM Findings</h2><p class='muted'>{provenance}. "
            "Validate before acting.</p>"
            f"<pre>{html.escape(report.llm_report)}</pre>"
        )
    elif report.llm_error:
        llm_html = (
            f"<h2>LLM Failure</h2><pre>{html.escape(report.llm_error)}</pre>"
        )

    return f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<title>Data Quality Report</title>
<style>{_STYLES}</style>
</head>
<body>
<h1>Data Quality Report</h1>
<p>
Generated <code>{html.escape(report.generated_at.isoformat())}</code>
using engine <code>{html.escape(report.dataset.engine)}</code>.
Dataset: {report.dataset.row_count:,} rows x
{report.dataset.column_count} columns.
</p>
<h2>Summary</h2>
<table>
<thead><tr><th>Check</th><th>Severity</th><th>Duration</th></tr></thead>
<tbody>{rows_html}</tbody>
</table>
<h2>Details</h2>
{details_html}
{llm_html}
</body>
</html>
"""


def _render_check_html(result: CheckResult) -> str:
    """Render one check's section: human summary plus collapsed JSON."""
    summary = _human_summary_html(result)
    error_html = (
        f"<p><b>Execution failed:</b> {html.escape(result.error)}</p>"
        if result.error
        else ""
    )
    raw = html.escape(json.dumps(result.payload, indent=2, default=str))
    return (
        f"<h3>{html.escape(result.name)} "
        f"<span class='badge badge-{result.severity}'>"
        f"{result.severity}</span></h3>"
        f"{error_html}"
        f"{summary}"
        f"<details><summary class='muted'>raw payload</summary>"
        f"<pre>{raw}</pre></details>"
    )


def _human_summary_html(result: CheckResult) -> str:
    payload = result.payload
    if not payload:
        return ""
    renderer = _SUMMARY_RENDERERS.get(result.name)
    return renderer(payload) if renderer else ""


def _missing_html(payload: dict[str, Any]) -> str:
    affected = [
        c for c in payload.get("per_column", []) if c.get("null_count", 0) > 0
    ]
    affected.sort(key=lambda c: c["null_count"], reverse=True)
    head = (
        f"<p>Total nulls: <b>{payload.get('total_null_count', 0):,}</b>. "
        f"Worst column: {payload.get('worst_column_pct', 0):.2f}%.</p>"
    )
    if not affected:
        return head + "<p class='muted'>No columns have nulls.</p>"
    rows = "".join(
        f"<tr><td><code>{html.escape(c['column'])}</code></td>"
        f"<td class='numeric'>{c['null_count']:,}</td>"
        f"<td class='numeric'>{c['null_percentage']:.2f}%</td></tr>"
        for c in affected[:10]
    )
    return (
        head
        + "<table><thead><tr><th>Column</th><th>Nulls</th>"
        + "<th>Percent</th></tr></thead><tbody>"
        + rows
        + "</tbody></table>"
    )


def _dataset_contract_html(payload: dict[str, Any]) -> str:
    head = (
        f"<p>Rows: <b>{payload.get('row_count', 0):,}</b> "
        f"(minimum: {payload.get('min_rows', 0):,}).</p>"
    )
    missing = payload.get("missing_required_columns") or []
    mismatches = payload.get("dtype_mismatches") or []
    if missing:
        columns = ", ".join(
            f"<code>{html.escape(str(column))}</code>" for column in missing
        )
        head += f"<p>Missing required columns: {columns}.</p>"
    if mismatches:
        rows = "".join(
            f"<tr><td><code>{html.escape(str(item['column']))}</code></td>"
            f"<td><code>{html.escape(str(item['expected']))}</code></td>"
            f"<td><code>{html.escape(str(item['actual']))}</code></td></tr>"
            for item in mismatches
        )
        head += (
            "<table><thead><tr><th>Column</th><th>Expected dtype</th>"
            f"<th>Actual dtype</th></tr></thead><tbody>{rows}</tbody></table>"
        )
    return head


def _duplicates_html(payload: dict[str, Any]) -> str:
    head = (
        "<p>Duplicate rows: "
        f"<b>{payload.get('total_duplicate_rows', 0):,}</b>.</p>"
    )
    subset = payload.get("subset")
    if subset:
        head += (
            "<p class='muted'>Subset: "
            + ", ".join(f"<code>{html.escape(c)}</code>" for c in subset)
            + "</p>"
        )
    return head


def _types_html(payload: dict[str, Any]) -> str:
    rollup = payload.get("rollup") or {}
    if not rollup:
        return ""
    rows = "".join(
        f"<tr><td><code>{html.escape(str(dt))}</code></td>"
        f"<td class='numeric'>{count}</td></tr>"
        for dt, count in rollup.items()
    )
    return (
        "<table><thead><tr><th>dtype</th><th>columns</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _outliers_html(payload: dict[str, Any]) -> str:
    per_col = payload.get("per_column", [])
    affected = [c for c in per_col if c.get("outlier_count", 0) > 0]
    skipped = [c for c in per_col if c.get("skipped")]
    affected.sort(key=lambda c: c["outlier_count"], reverse=True)
    head = (
        f"<p>Numeric columns scanned: {len(per_col)} "
        f"(<b>{len(affected)}</b> with outliers, "
        f"<b>{len(skipped)}</b> skipped).</p>"
    )
    if affected:
        rows = "".join(
            f"<tr><td><code>{html.escape(c['column'])}</code></td>"
            f"<td class='numeric'>{c['outlier_count']:,}</td>"
            f"<td class='numeric'><code>"
            f"[{c['lower_bound']:.2f}, {c['upper_bound']:.2f}]</code></td>"
            "</tr>"
            for c in affected[:10]
        )
        head += (
            "<table><thead><tr><th>Column</th><th>Outliers</th>"
            + "<th>Bounds (IQR)</th></tr></thead><tbody>"
            + rows
            + "</tbody></table>"
        )
    if skipped:
        notes = "".join(
            f"<li><code>{html.escape(c['column'])}</code>: "
            f"{html.escape(str(c['skipped']))}</li>"
            for c in skipped
        )
        head += f"<ul class='muted'>{notes}</ul>"
    return head


def _ranges_html(payload: dict[str, Any]) -> str:
    per_col = payload.get("per_column", [])
    affected = [c for c in per_col if c.get("violation_count", 0) > 0]
    notes = [c for c in per_col if c.get("note")]
    head = f"<p>Ranges configured: {len(per_col)}.</p>"
    body = ""
    if affected:
        rows = "".join(
            f"<tr><td><code>{html.escape(c['column'])}</code></td>"
            f"<td><code>[{c['min_allowed']}, {c['max_allowed']}]</code></td>"
            f"<td class='numeric'>{c['violation_count']:,}</td></tr>"
            for c in affected
        )
        body += (
            "<table><thead><tr><th>Column</th><th>Allowed</th>"
            + "<th>Violations</th></tr></thead><tbody>"
            + rows
            + "</tbody></table>"
        )
    if notes:
        notes_html = "".join(
            f"<li><code>{html.escape(c['column'])}</code>: "
            f"{html.escape(c['note'])}</li>"
            for c in notes
        )
        body += f"<ul class='muted'>{notes_html}</ul>"
    return head + body


def _cardinality_html(payload: dict[str, Any]) -> str:
    per_col = payload.get("per_column", [])
    constants = [c for c in per_col if c.get("distinct_count", 1) <= 1]
    head = f"<p>Columns profiled: {len(per_col)}.</p>"
    if not constants:
        return head
    items = ", ".join(
        f"<code>{html.escape(c['column'])}</code>" for c in constants
    )
    return head + f"<p>Constant columns: {items}.</p>"


def _freshness_html(payload: dict[str, Any]) -> str:
    per_col = payload.get("per_column", [])
    invalid = [c for c in per_col if c.get("is_stale") or c.get("is_future")]
    head = (
        f"<p>Checked: {len(per_col)} "
        f"(invalid freshness: <b>{len(invalid)}</b>).</p>"
    )
    if not invalid:
        return head
    rows = "".join(_freshness_row_html(column) for column in invalid)
    return (
        head
        + "<table><thead><tr><th>Column</th><th>Max Timestamp</th>"
        + "<th>Age</th></tr></thead><tbody>"
        + rows
        + "</tbody></table>"
    )


def _freshness_row_html(column: dict[str, Any]) -> str:
    age = column.get("age_hours")
    detail = (
        html.escape(str(column.get("note", "no non-null values")))
        if age is None
        else (
            f"{abs(age):.1f}h {'future' if column.get('is_future') else 'old'}"
        )
    )
    return (
        "<tr>"
        f"<td><code>{html.escape(column['column'])}</code></td>"
        f"<td>{html.escape(str(column.get('max_timestamp') or 'n/a'))}</td>"
        f"<td class='numeric'>{detail}</td></tr>"
    )


def _linkage_html(payload: dict[str, Any]) -> str:
    if payload.get("skipped"):
        return "<p class='muted'>skipped (no linkage config supplied)</p>"
    return (
        "<ul>"
        f"<li>candidate pairs: {payload.get('candidate_pairs', 0):,}</li>"
        f"<li>matched pairs: {payload.get('matched_pairs', 0):,}</li>"
        "<li>match threshold: "
        f"{payload.get('match_threshold_probability', 0.9)}</li>"
        f"<li>duplicate clusters: {payload.get('duplicate_clusters', 0)}</li>"
        "<li>records in duplicate groups: "
        f"{payload.get('records_in_duplicate_groups', 0)}</li>"
        "</ul>"
    )


_SUMMARY_RENDERERS: dict[str, Callable[[dict[str, Any]], str]] = {
    "dataset_contract": _dataset_contract_html,
    "missing_values": _missing_html,
    "duplicates": _duplicates_html,
    "data_types": _types_html,
    "outliers": _outliers_html,
    "ranges": _ranges_html,
    "cardinality": _cardinality_html,
    "freshness": _freshness_html,
    "linkage": _linkage_html,
}
