"""Bounded, reproducible record-linkage benchmark.

Each size runs the full input/backend matrix: pandas and Polars inputs with
Polars and DuckDB compute backends. Input generation is outside the timed
region, and candidate pairs are capped at three million per trial.

Run the default sizes with three measured trials per cell:

    python scripts/bench_linking.py

Run one small size and save the complete trial data:

    python scripts/bench_linking.py --n 5000 --trials 3 --output result.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import string
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

import numpy as np
import pandas as pd
import polars as pl

from qualipilot.linking import (
    ExactMatch,
    FuzzyString,
    LinkConfig,
    NumericDiff,
    RecordLinker,
)
from qualipilot.linking.config import Backend

if not __debug__:  # pragma: no cover - fail closed under python -O.
    raise RuntimeError("benchmark gates require assertions")

PAIR_CAP = 3_000_000
MEM_BUDGET_MB = 6_000
DEFAULT_TRIALS = 3
SEED = 7
BLOCK_BUCKETS = 2_000
DUPLICATE_FRACTION = 0.01

InputType = Literal["pandas", "polars"]
Frame: TypeAlias = pd.DataFrame | pl.DataFrame
INPUT_TYPES: tuple[InputType, ...] = ("pandas", "polars")
BACKENDS: tuple[Backend, ...] = ("polars", "duckdb")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n",
        type=_positive_int,
        default=None,
        help="base rows; defaults to 5k, 25k, and 100k",
    )
    parser.add_argument("--trials", type=_positive_int, default=DEFAULT_TRIALS)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional path for the complete benchmark JSON",
    )
    return parser.parse_args()


def _rss_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return float(psutil.Process().memory_info().rss) / (1024 * 1024)


def _build_frame(n: int) -> tuple[pl.DataFrame, list[int]]:
    rng = np.random.default_rng(SEED)
    alphabet = list(string.ascii_lowercase)
    names = ["".join(rng.choice(alphabet, 12)) for _ in range(n)]
    duplicate_count = max(1, int(n * DUPLICATE_FRACTION))
    duplicate_ids = [
        int(value) for value in rng.choice(n, duplicate_count, replace=False)
    ]

    def typo(value: str, seed: int) -> str:
        typo_rng = np.random.default_rng(seed)
        characters = list(value)
        position = int(typo_rng.integers(0, len(characters)))
        characters[position] = "q"
        return "".join(characters)

    frame = pl.DataFrame(
        {
            "id": list(range(n)),
            "name": names,
            "postcode": [f"PC{i % BLOCK_BUCKETS:05d}" for i in range(n)],
            "dob": rng.integers(1950, 2005, n),
        }
    )
    duplicates = pl.DataFrame(
        {
            "id": list(range(n, n + duplicate_count)),
            "name": [typo(names[row_id], row_id) for row_id in duplicate_ids],
            "postcode": [
                frame["postcode"][row_id] for row_id in duplicate_ids
            ],
            "dob": [frame["dob"][row_id] for row_id in duplicate_ids],
        }
    )
    return frame.vstack(duplicates), duplicate_ids


def _candidate_pair_count(frame: pl.DataFrame) -> int:
    counts = frame.group_by("postcode").len()
    value = counts.select(
        ((pl.col("len") * (pl.col("len") - 1)) // 2).sum()
    ).item()
    return int(value)


def _build_input(
    n: int, input_type: InputType
) -> tuple[Frame, list[int], int]:
    canonical, duplicate_ids = _build_frame(n)
    expected_pairs = _candidate_pair_count(canonical)
    if expected_pairs > PAIR_CAP:
        raise ValueError(
            f"{n:,} base rows create {expected_pairs:,} candidate pairs; "
            f"the cap is {PAIR_CAP:,}"
        )
    if input_type == "pandas":
        pandas_frame: pd.DataFrame = canonical.to_pandas()
        return pandas_frame, duplicate_ids, expected_pairs
    return canonical, duplicate_ids, expected_pairs


def _link_config(backend: Backend) -> LinkConfig:
    return LinkConfig(
        unique_id_column="id",
        comparisons=[
            FuzzyString(column="name", thresholds=(0.92, 0.75)),
            ExactMatch(column="postcode"),
            NumericDiff(column="dob", thresholds=(0.0, 1.0)),
        ],
        blocking_rules=[["postcode"]],
        match_threshold_probability=0.9,
        backend=backend,
        em_random_seed=0,
        max_pairs_warning=PAIR_CAP,
        max_pairs_hard_cap=PAIR_CAP,
    )


def _recall(clusters: dict[Any, int], duplicate_ids: list[int], n: int) -> int:
    return sum(
        1
        for index, original_id in enumerate(duplicate_ids)
        if n + index in clusters
        and original_id in clusters
        and clusters[n + index] == clusters[original_id]
    )


def _run_once(
    frame: Frame,
    *,
    backend: Backend,
    n: int,
    duplicate_ids: list[int],
    expected_pairs: int,
    trial: int,
) -> dict[str, Any]:
    config = _link_config(backend)
    gc.collect()
    rss_before = _rss_mb()
    started = time.perf_counter()
    # RecordLinker normalizes pandas inputs internally, so construction belongs
    # in the timed region even though its public annotation is still Polars.
    result = RecordLinker(frame, config).run()
    elapsed_ms = (time.perf_counter() - started) * 1000
    rss_after = _rss_mb()
    summary = result.summary()

    expected_duplicates = len(duplicate_ids)
    input_rows = n + expected_duplicates
    candidate_pairs = int(summary["candidate_pairs"])
    matched_pairs = int(summary["matched_pairs"])
    cluster_count = int(summary["clusters"])
    recalled = _recall(result.clusters, duplicate_ids, n)
    assert candidate_pairs == expected_pairs, (
        f"candidate pairs: expected {expected_pairs}, got {candidate_pairs}"
    )
    assert matched_pairs == expected_duplicates, (
        f"matched pairs: expected {expected_duplicates}, got {matched_pairs}"
    )
    assert recalled == expected_duplicates, (
        f"recall: expected {expected_duplicates}, got {recalled}"
    )
    assert len(result.clusters) == input_rows, (
        "cluster membership: expected "
        f"{input_rows}, got {len(result.clusters)}"
    )
    assert cluster_count == n, (
        f"cluster count: expected {n}, got {cluster_count}"
    )

    rss_delta = (
        None
        if rss_before is None or rss_after is None
        else rss_after - rss_before
    )
    return {
        "trial": trial,
        "status": "PASS",
        "elapsed_ms": round(elapsed_ms, 6),
        "rss_before_mb": None if rss_before is None else round(rss_before, 3),
        "rss_after_mb": None if rss_after is None else round(rss_after, 3),
        "rss_delta_mb": None if rss_delta is None else round(rss_delta, 3),
        "candidate_pairs": candidate_pairs,
        "matched_pairs": matched_pairs,
        "recalled_duplicates": recalled,
        "cluster_count": cluster_count,
        "stage_ms": {
            name: round(float(value), 6)
            for name, value in result.timings_ms.items()
        },
        "lambda": float(summary["lambda"]),
        "fit_status": str(summary["fit_status"]),
        "fit_warnings": summary["fit_warnings"],
    }


def _median_stages(runs: list[dict[str, Any]]) -> dict[str, float]:
    stage_names = {
        name
        for run in runs
        for name in cast(dict[str, float], run["stage_ms"])
    }
    return {
        name: round(
            statistics.median(
                cast(dict[str, float], run["stage_ms"])[name] for run in runs
            ),
            6,
        )
        for name in sorted(stage_names)
    }


def _benchmark_cell(
    n: int,
    input_type: InputType,
    backend: Backend,
    trials: int,
) -> dict[str, Any]:
    frame, duplicate_ids, expected_pairs = _build_input(n, input_type)
    warmup = _run_once(
        frame,
        backend=backend,
        n=n,
        duplicate_ids=duplicate_ids,
        expected_pairs=expected_pairs,
        trial=0,
    )
    runs = [
        _run_once(
            frame,
            backend=backend,
            n=n,
            duplicate_ids=duplicate_ids,
            expected_pairs=expected_pairs,
            trial=trial,
        )
        for trial in range(1, trials + 1)
    ]
    elapsed = [float(run["elapsed_ms"]) for run in runs]
    rss_deltas = [
        float(run["rss_delta_mb"])
        for run in runs
        if run["rss_delta_mb"] is not None
    ]
    return {
        "status": "PASS",
        "base_rows": n,
        "input_rows": n + len(duplicate_ids),
        "input_type": input_type,
        "compute_backend": backend,
        "expected_candidate_pairs": expected_pairs,
        "candidate_pairs": int(runs[0]["candidate_pairs"]),
        "expected_duplicates": len(duplicate_ids),
        "recalled_duplicates": int(runs[0]["recalled_duplicates"]),
        "median_elapsed_ms": round(statistics.median(elapsed), 6),
        "max_rss_delta_mb": max(rss_deltas, default=None),
        "median_stage_ms": _median_stages(runs),
        "warmup": warmup,
        "trials": runs,
    }


def _git_info() -> dict[str, Any]:
    root = Path(__file__).parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {}
    return {"commit": commit, "dirty": bool(status.strip())}


def _print_result(result: dict[str, Any]) -> None:
    rss = result["max_rss_delta_mb"]
    rss_display = "n/a" if rss is None else f"{float(rss):.1f}"
    print(
        f"{int(result['base_rows']):>9,} "
        f"{int(result['input_rows']):>10,} "
        f"{result['input_type']!s:>7} "
        f"{result['compute_backend']!s:>7} "
        f"{int(result['candidate_pairs']):>10,} "
        f"{float(result['median_elapsed_ms']):>10.1f} "
        f"{rss_display:>9} "
        f"{int(result['recalled_duplicates'])}/"
        f"{int(result['expected_duplicates'])}"
    )
    trial_times = [
        float(run["elapsed_ms"])
        for run in cast(list[dict[str, Any]], result["trials"])
    ]
    print(f"    trial_ms: {trial_times}")
    print(f"    median_stage_ms: {result['median_stage_ms']}")


def _report(args: argparse.Namespace) -> dict[str, Any]:
    sizes = [args.n] if args.n is not None else [5_000, 25_000, 100_000]
    print(
        f"{'base_rows':>9} {'input_rows':>10} {'input':>7} "
        f"{'compute':>7} {'pairs':>10} {'median_ms':>10} "
        f"{'rss_dmb':>9} {'recall':>9}"
    )
    print("-" * 89)
    results: list[dict[str, Any]] = []
    stopped_for_memory = False
    for n in sizes:
        for input_type in INPUT_TYPES:
            for backend in BACKENDS:
                result = _benchmark_cell(n, input_type, backend, args.trials)
                results.append(result)
                _print_result(result)
                gc.collect()
                rss = _rss_mb()
                if rss is not None and rss > MEM_BUDGET_MB:
                    print(f"stopping: RSS {rss:.0f} MB exceeded budget")
                    stopped_for_memory = True
                    break
            if stopped_for_memory:
                break
        if stopped_for_memory:
            break

    return {
        "benchmark": "qualipilot record linkage input/backend matrix",
        "recorded_at": datetime.now(UTC).isoformat(),
        "command": [sys.executable, *sys.argv],
        "status": "STOPPED_MEMORY" if stopped_for_memory else "PASS",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "packages": {
                package: version(package)
                for package in (
                    "qualipilot",
                    "numpy",
                    "pandas",
                    "polars",
                    "duckdb",
                    "rapidfuzz",
                )
            },
            "git": _git_info(),
        },
        "method": {
            "seed": SEED,
            "base_sizes": sizes,
            "duplicate_fraction": DUPLICATE_FRACTION,
            "postcode_block_buckets": BLOCK_BUCKETS,
            "input_types": list(INPUT_TYPES),
            "compute_backends": list(BACKENDS),
            "blocking_rules": [["postcode"]],
            "comparisons": [
                {
                    "type": "fuzzy_string",
                    "column": "name",
                    "thresholds": [0.92, 0.75],
                },
                {"type": "exact_match", "column": "postcode"},
                {
                    "type": "numeric_diff",
                    "column": "dob",
                    "thresholds": [0.0, 1.0],
                },
            ],
            "match_threshold_probability": 0.9,
            "em_random_seed": 0,
            "pair_cap": PAIR_CAP,
            "memory_budget_mb": MEM_BUDGET_MB,
            "trials_per_cell": args.trials,
            "warmups_per_cell": 1,
            "timed_scope": (
                "RecordLinker construction plus run; deterministic input "
                "generation excluded"
            ),
        },
        "results": results,
        "stopped_for_memory": stopped_for_memory,
    }


def main() -> None:
    args = _arguments()
    report = _report(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.output}")
    if report["stopped_for_memory"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
