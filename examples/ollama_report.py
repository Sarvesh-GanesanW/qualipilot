"""Run checks using a locally-hosted Ollama model for the LLM step.

Requires:
    ollama pull llama3.2
"""

from __future__ import annotations

from pathlib import Path

from datapilot import DataQualityChecker, DatapilotConfig
from datapilot.models.config import LLMConfig

SAMPLE = Path(__file__).parent / "sample.csv"


def main() -> None:
    config = DatapilotConfig(
        engine="polars",
        llm=LLMConfig(
            provider="ollama",
            base_url="http://localhost:11434",
            model="llama3.2:latest",
        ),
    )
    report = DataQualityChecker(SAMPLE, config).run()
    print(report.llm_report or "<no llm report>")


if __name__ == "__main__":
    main()
