"""Minimal example: run every check on a CSV, print results."""

from __future__ import annotations

from pathlib import Path

from datapilot import DataQualityChecker, DatapilotConfig
from datapilot.models.config import CheckConfig, ColumnRange

SAMPLE = Path(__file__).parent / "sample.csv"


def main() -> None:
    config = DatapilotConfig(
        engine="polars",
        checks=CheckConfig(
            column_ranges={
                "amount": ColumnRange(min=0, max=100_000),
            },
        ),
    )
    report = DataQualityChecker(SAMPLE, config).run()
    print(report.to_json())


if __name__ == "__main__":
    main()
