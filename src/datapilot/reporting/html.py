"""Minimal dependency-free HTML report.

We intentionally avoid Jinja: a static template keeps the package
tiny and the output predictable for CI artefacts.
"""

from __future__ import annotations

import html
import json

from datapilot.models.results import QualityReport

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
"""


def render_html(report: QualityReport) -> str:
    """Return a full HTML document for the given report."""
    rows_html = "\n".join(
        f"<tr><td>{html.escape(r.name)}</td>"
        f"<td><span class='badge badge-{r.severity}'>"
        f"{r.severity}</span></td>"
        f"<td>{r.duration_seconds:.3f}s</td></tr>"
        for r in report.results
    )

    details_html = "\n".join(
        f"<details><summary><b>{html.escape(r.name)}</b> "
        f"<span class='badge badge-{r.severity}'>{r.severity}</span>"
        f"</summary><pre>"
        f"{html.escape(json.dumps(r.payload, indent=2, default=str))}"
        f"</pre></details>"
        for r in report.results
    )

    llm_html = ""
    if report.llm_report:
        llm_html = (
            "<h2>LLM Findings</h2>"
            f"<pre>{html.escape(report.llm_report)}</pre>"
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
