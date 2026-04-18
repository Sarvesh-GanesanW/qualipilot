"""Bounded, RAM-safe linker benchmark.

Run with:
    python scripts/bench_linking.py
or pass a specific row count:
    python scripts/bench_linking.py --n 50000

We refuse configurations that would blow past ~10 M candidate pairs
so the box stays responsive. Memory usage is reported via psutil if
available, else skipped.
"""

from __future__ import annotations

import argparse
import gc
import string
import time
from typing import Any

import numpy as np
import polars as pl

from datapilot.linking import (
    ExactMatch,
    FuzzyString,
    LinkConfig,
    NumericDiff,
    RecordLinker,
)

# cap candidate pairs per run — stay well under laptop RAM
PAIR_CAP = 3_000_000
# hard memory budget warning (best-effort, needs psutil)
MEM_BUDGET_MB = 6_000


def _rss_mb() -> float:
    # psutil is optional — if not installed we just omit the memory
    # delta column rather than failing the whole run
    try:
        import psutil  # type: ignore[import-not-found]

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return float("nan")


def _build_frame(n: int, dupe_frac: float = 0.01) -> tuple[
    pl.DataFrame, list[int]
]:
    rng = np.random.default_rng(7)
    names = [
        "".join(rng.choice(list(string.ascii_lowercase), 12))
        for _ in range(n)
    ]
    dupe_ids = rng.choice(
        n, max(1, int(n * dupe_frac)), replace=False
    ).tolist()

    def typo(s: str, seed: int) -> str:
        r = np.random.default_rng(seed)
        c = list(s)
        pos = int(r.integers(0, len(c)))
        c[pos] = "q"
        return "".join(c)

    dupe_names = [typo(names[int(i)], int(i)) for i in dupe_ids]

    # tight compound blocking — postcode + dob_year keeps pair count
    # well under the cap even at 500k rows
    df = pl.DataFrame(
        {
            "id": list(range(n)),
            "name": names,
            "postcode": [f"PC{i % 2000:05d}" for i in range(n)],
            "dob": rng.integers(1950, 2005, n),
        }
    )
    dupes = pl.DataFrame(
        {
            "id": list(range(n, n + len(dupe_ids))),
            "name": dupe_names,
            "postcode": [
                df["postcode"][int(i)] for i in dupe_ids
            ],
            "dob": [df["dob"][int(i)] for i in dupe_ids],
        }
    )
    return df.vstack(dupes), dupe_ids


def _run(
    df: pl.DataFrame,
    backend: str,
    comparisons: list[Any],
    rules: list[list[str]],
) -> dict[str, Any]:
    cfg = LinkConfig(
        unique_id_column="id",
        comparisons=comparisons,
        blocking_rules=rules,
        match_threshold_probability=0.9,
        backend=backend,  # type: ignore[arg-type]
        max_pairs_hard_cap=PAIR_CAP,
    )
    gc.collect()
    rss_before = _rss_mb()
    start = time.perf_counter()
    result = RecordLinker(df, cfg).run()
    elapsed_ms = (time.perf_counter() - start) * 1000
    rss_after = _rss_mb()
    return {
        "backend": backend,
        "total_ms": round(elapsed_ms, 1),
        "pairs": result.pairs.height,
        "rss_delta_mb": round(rss_after - rss_before, 1),
        "clusters": len(set(result.clusters.values())),
        "timings": result.timings_ms,
        "lambda": result.parameters.get("lambda", 0.0),
        "_clusters": result.clusters,
    }


def _recall(
    clusters: dict[Any, int], dupe_ids: list[int], offset: int
) -> int:
    return sum(
        1
        for i, oi in enumerate(dupe_ids)
        if clusters.get(offset + i) == clusters.get(int(oi))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None)
    args = parser.parse_args()

    sizes = [args.n] if args.n else [5_000, 25_000, 100_000]
    comparisons = [
        FuzzyString(column="name", thresholds=(0.92, 0.75)),
        ExactMatch(column="postcode"),
        NumericDiff(column="dob", thresholds=(0.0, 1.0)),
    ]
    rules = [["postcode", "dob"]]

    print(
        f"{'rows':>8} {'backend':>7} {'pairs':>10} "
        f"{'total_ms':>9} {'rss_dmb':>8} {'recall':>7}"
    )
    print("-" * 60)
    for n in sizes:
        df, dupe_ids = _build_frame(n)
        for backend in ["polars", "duckdb"]:
            try:
                r = _run(df, backend, comparisons, rules)
            except MemoryError as exc:
                print(
                    f"{n:>8,} {backend:>7} skipped — {exc}"
                )
                continue
            recall = _recall(r["_clusters"], dupe_ids, n)
            print(
                f"{n:>8,} {backend:>7} {r['pairs']:>10,} "
                f"{r['total_ms']:>9.0f} {r['rss_delta_mb']:>8} "
                f"{recall}/{len(dupe_ids)}"
            )
            print(f"    timings: {r['timings']}")
            gc.collect()

        rss = _rss_mb()
        if rss == rss and rss > MEM_BUDGET_MB:
            print(f"stopping: RSS {rss:.0f} MB exceeded budget")
            break


if __name__ == "__main__":
    main()
