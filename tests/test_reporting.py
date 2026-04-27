"""Tests for the markdown + html report renderers.

These are tighter than golden-file comparisons: we assert the human
summary surfaces affected columns by name, not just the count.
"""

from __future__ import annotations

import pandas as pd

from qualipilot import DataQualityChecker, QualipilotConfig
from qualipilot.models.config import CheckConfig, ColumnRange
from qualipilot.reporting import render_html, render_markdown


def _report(dirty_pandas: pd.DataFrame):
    cfg = QualipilotConfig(
        checks=CheckConfig(
            column_ranges={"amount": ColumnRange(min=0, max=100)},
        ),
    )
    return DataQualityChecker(dirty_pandas, cfg).run()


def test_markdown_lists_columns_with_nulls(
    dirty_pandas: pd.DataFrame,
) -> None:
    md = render_markdown(_report(dirty_pandas))
    # `amount` is the column with the null in the fixture
    assert "`amount`" in md
    assert "Nulls" in md


def test_markdown_lists_outlier_columns_with_bounds(
    dirty_pandas: pd.DataFrame,
) -> None:
    md = render_markdown(_report(dirty_pandas))
    # the IQR table must appear with the affected column and bounds
    assert "Bounds (IQR)" in md
    assert "amount" in md


def test_markdown_lists_range_violations(
    dirty_pandas: pd.DataFrame,
) -> None:
    md = render_markdown(_report(dirty_pandas))
    assert "Violations" in md
    # the range we configured surfaces in the affected table
    assert "[0.0, 100.0]" in md


def test_markdown_outlier_phrasing_says_numeric(
    dirty_pandas: pd.DataFrame,
) -> None:
    md = render_markdown(_report(dirty_pandas))
    # previous phrasing was "columns evaluated: 3" which read as
    # "only 3 of 6 columns checked" — we now say "numeric columns".
    assert "numeric columns scanned" in md


def test_html_renders_affected_columns_table(
    dirty_pandas: pd.DataFrame,
) -> None:
    h = render_html(_report(dirty_pandas))
    assert "<th>Outliers</th>" in h
    assert "<th>Bounds (IQR)</th>" in h
    assert "<th>Violations</th>" in h


def test_html_keeps_raw_payload_collapsed(
    dirty_pandas: pd.DataFrame,
) -> None:
    h = render_html(_report(dirty_pandas))
    assert "<details>" in h
    assert "raw payload" in h
