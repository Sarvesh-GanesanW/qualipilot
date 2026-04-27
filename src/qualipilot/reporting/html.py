"""Minimal dependency-free HTML report.

We intentionally avoid Jinja: a static template keeps the package
tiny and the output predictable for CI artefacts.
"""

from __future__ import annotations

import html
import json
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
        llm_html = (
            f"<h2>LLM Findings</h2><pre>{html.escape(report.llm_report)}</pre>"
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
    raw = html.escape(json.dumps(result.payload, indent=2, default=str))
    return (
        f"<h3>{html.escape(result.name)} "
        f"<span class='badge badge-{result.severity}'>"
        f"{result.severity}</span></h3>"
        f"{summary}"
        f"<details><summary class='muted'>raw payload</summary>"
        f"<pre>{raw}</pre></details>"
    )


def _human_summary_html(result: CheckResult) -> str:
    payload = result.payload
    if not payload:
        return ""
    name = result.name
    if name == "missing_values":
        return _missing_html(payload)
    if name == "duplicates":
        return _duplicates_html(payload)
    if name == "data_types":
        return _types_html(payload)
    if name == "outliers":
        return _outliers_html(payload)
    if name == "ranges":
        return _ranges_html(payload)
    if name == "cardinality":
        return _cardinality_html(payload)
    if name == "freshness":
        return _freshness_html(payload)
    if name == "linkage":
        return _linkage_html(payload)
    return ""


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
    affected.sort(key=lambda c: c["outlier_count"], reverse=True)
    head = (
        f"<p>Numeric columns scanned: {len(per_col)} "
        f"(<b>{len(affected)}</b> with outliers).</p>"
    )
    if not affected:
        return head
    rows = "".join(
        f"<tr><td><code>{html.escape(c['column'])}</code></td>"
        f"<td class='numeric'>{c['outlier_count']:,}</td>"
        f"<td class='numeric'><code>"
        f"[{c['lower_bound']:.2f}, {c['upper_bound']:.2f}]</code></td></tr>"
        for c in affected[:10]
    )
    return (
        head
        + "<table><thead><tr><th>Column</th><th>Outliers</th>"
        + "<th>Bounds (IQR)</th></tr></thead><tbody>"
        + rows
        + "</tbody></table>"
    )


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
    stale = [c for c in per_col if c.get("is_stale")]
    head = f"<p>Checked: {len(per_col)} (stale: <b>{len(stale)}</b>).</p>"
    if not stale:
        return head
    rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(c['column'])}</code></td>"
        f"<td>{html.escape(str(c.get('max_timestamp', 'n/a')))}</td>"
        f"<td class='numeric'>"
        f"{(c.get('age_hours') or 0):.1f}h</td></tr>"
        for c in stale
    )
    return (
        head
        + "<table><thead><tr><th>Column</th><th>Max Timestamp</th>"
        + "<th>Age</th></tr></thead><tbody>"
        + rows
        + "</tbody></table>"
    )


def _linkage_html(payload: dict[str, Any]) -> str:
    if payload.get("skipped"):
        return "<p class='muted'>skipped (no linkage config supplied)</p>"
    return (
        "<ul>"
        f"<li>candidate pairs: {payload.get('candidate_pairs', 0):,}</li>"
        f"<li>matched pairs: {payload.get('matched_pairs', 0):,}</li>"
        f"<li>duplicate clusters: {payload.get('duplicate_clusters', 0)}</li>"
        "<li>records in duplicate groups: "
        f"{payload.get('records_in_duplicate_groups', 0)}</li>"
        "</ul>"
    )
