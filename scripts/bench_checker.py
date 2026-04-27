"""Bounded benchmark for the quality-checker pipeline.

Writes results to reports/bench_checker.txt so output survives if the
process is killed.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from qualipilot import DataQualityChecker, QualipilotConfig
from qualipilot.models.config import CheckConfig, ColumnRange

OUT = Path(__file__).parent.parent / "reports" / "bench_checker.txt"
OUT.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(msg, flush=True)
    with OUT.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def build(n: int) -> "pl.DataFrame":
    rng = np.random.default_rng(7)
    return pl.DataFrame(
        {
            "id": list(range(n)),
            "amount": rng.normal(100, 50, n),
            "category": rng.choice(["a", "b", "c", "d", "e"], n),
            "score": rng.uniform(0, 100, n),
            "flag": rng.choice([0, 1], n),
        }
    )


def run(
    df: "pl.DataFrame", engine: str, size: int
) -> None:
    cfg = QualipilotConfig(
        engine=engine,  # type: ignore[arg-type]
        checks=CheckConfig(
            column_ranges={
                "amount": ColumnRange(min=-100, max=500)
            }
        ),
    )
    input_frame = df.to_pandas() if engine == "pandas" else df
    gc.collect()
    start = time.perf_counter()
    report = DataQualityChecker(input_frame, cfg).run()
    ms = (time.perf_counter() - start) * 1000
    per_check = {
        r.name: round(r.duration_seconds * 1000, 2)
        for r in report.results
    }
    log(
        f"n={size:>7,}  engine={engine:>7}  total={ms:>7.1f}ms  "
        f"per_check={per_check}"
    )


def main() -> None:
    OUT.unlink(missing_ok=True)
    log("# quality checker bench")
    log(
        f"# python={sys.version.split()[0]} "
        f"polars={pl.__version__} numpy={np.__version__}"
    )

    for n in [10_000, 100_000, 500_000]:
        df = build(n)
        for engine in ["polars", "pandas", "duckdb"]:
            try:
                run(df, engine, n)
            except Exception as exc:
                log(
                    f"n={n:>7,}  engine={engine:>7}  FAILED: {exc}"
                )
        gc.collect()


if __name__ == "__main__":
    main()
