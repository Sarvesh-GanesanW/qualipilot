"""Write deterministic evidence for the quality-contract CI artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import pandas as pd

from qualipilot import DataQualityChecker, QualipilotConfig, __version__


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = Path("examples/quality_contract.yaml")
    frame = pd.DataFrame({"id": [1, 2], "start": [1, 2], "end": [2, 1]})
    config = QualipilotConfig.from_file(config_path)
    report = DataQualityChecker(frame, config).run(include_llm=False)
    contract = next(
        result
        for result in report.results
        if result.name == "quality_contract"
    )
    if (
        report.dataset.row_count != 2
        or contract.payload["violation_count"] != 1
    ):
        raise RuntimeError("contract evidence fixture changed semantically")
    stable = report.model_dump(
        mode="json", exclude={"generated_at", "package_version"}
    )
    for result in stable["results"]:
        result["duration_seconds"] = 0.0
    args.output.write_text(
        json.dumps(
            {
                "fixture_sha256": hashlib.sha256(
                    frame.to_csv(index=False).encode()
                ).hexdigest(),
                "config_sha256": hashlib.sha256(
                    config_path.read_bytes()
                ).hexdigest(),
                "report_sha256": hashlib.sha256(
                    json.dumps(
                        stable, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest(),
                "config_hash": report.config_hash,
                "provenance": {
                    "package_version": __version__,
                    "python_version": platform.python_version(),
                    "source_revision": os.environ.get(
                        "GITHUB_SHA", "working-tree"
                    ),
                },
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
