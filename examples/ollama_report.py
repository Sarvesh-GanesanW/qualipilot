"""Run checks using a locally-hosted Ollama model for the LLM step.

Requires:
    ollama pull qwen3:4b
"""

from __future__ import annotations

import os
from pathlib import Path

from qualipilot import DataQualityChecker, QualipilotConfig
from qualipilot.models.config import LLMConfig

SAMPLE = Path(__file__).parent / "sample.csv"


def main() -> None:
    config = QualipilotConfig(
        engine="polars",
        llm=LLMConfig(
            provider="ollama",
            base_url="http://localhost:11434",
            model=os.environ.get("OLLAMA_MODEL", "qwen3:4b"),
        ),
    )
    report = DataQualityChecker(SAMPLE, config).run()
    print(report.llm_report or "<no llm report>")


if __name__ == "__main__":
    main()
