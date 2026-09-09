from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_contract_evidence_is_repeatable(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    outputs = [tmp_path / "one.json", tmp_path / "two.json"]
    for output in outputs:
        subprocess.run(
            [
                sys.executable,
                "scripts/contract_evidence.py",
                "--output",
                str(output),
            ],
            cwd=root,
            check=True,
        )
    first, second = (json.loads(path.read_text()) for path in outputs)
    assert first == second
    assert first["report_sha256"]
