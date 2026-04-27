"""Reporters that turn a ``QualityReport`` into user-facing artifacts."""

from qualipilot.reporting.html import render_html
from qualipilot.reporting.markdown import render_markdown

__all__ = ["render_html", "render_markdown"]
