"""Run checks and ask Bedrock to write the narrative report.

Requires:
    pip install data-pilot-checker[bedrock]
    aws configure --profile <your-profile>
"""

from __future__ import annotations

import os
from pathlib import Path

from datapilot import DataQualityChecker, DatapilotConfig
from datapilot.models.config import CheckConfig, ColumnRange, LLMConfig

SAMPLE = Path(__file__).parent / "sample.csv"


def main() -> None:
    config = DatapilotConfig(
        engine="polars",
        checks=CheckConfig(
            column_ranges={
                "amount": ColumnRange(min=0, max=100_000),
            },
        ),
        llm=LLMConfig(
            provider="bedrock",
            model="provider.model-3-5-haiku-20241022-v1:0",
            region=os.environ.get("AWS_REGION", "us-east-1"),
            aws_profile=os.environ.get("AWS_PROFILE"),
            max_tokens=1500,
            temperature=0.2,
        ),
    )
    report = DataQualityChecker(SAMPLE, config).run()
    print("=" * 80)
    print(report.llm_report or "<no llm report>")


if __name__ == "__main__":
    main()
