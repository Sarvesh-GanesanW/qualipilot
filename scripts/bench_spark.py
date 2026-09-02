"""Reproducible Spark benchmark for batched range checks.

Run the 100-million-row default:

    python scripts/bench_spark.py

Run a quick smoke test:

    python scripts/bench_spark.py --rows 100000 --master local[2] \
        --partitions 8 --trials 1 --driver-memory 1g
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from qualipilot import CheckConfig, DataQualityChecker, QualipilotConfig
from qualipilot.models.config import ColumnRange

if not __debug__:  # pragma: no cover - fail closed under python -O.
    raise RuntimeError("benchmark gates require assertions")

DEFAULT_ROWS = 100_000_000
DEFAULT_PARTITIONS = 64
DEFAULT_TRIALS = 5
RANGE_SPECS = {
    "amount": (1_000, 0, 899),
    "quantity": (100, 10, 89),
    "score": (200, 20, 179),
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=_positive_int, default=DEFAULT_ROWS)
    parser.add_argument(
        "--partitions", type=_positive_int, default=DEFAULT_PARTITIONS
    )
    parser.add_argument("--trials", type=_positive_int, default=DEFAULT_TRIALS)
    parser.add_argument("--master", default="local[8]")
    parser.add_argument("--driver-memory", default="4g")
    return parser.parse_args()


def _expected_outside(rows: int, modulus: int, low: int, high: int) -> int:
    full_cycles, remainder = divmod(rows, modulus)
    allowed = full_cycles * (high - low + 1)
    allowed += max(0, min(remainder, high + 1) - low)
    return rows - allowed


def _proc_status(pid: int | None) -> dict[str, str]:
    if pid is None:
        return {}
    wanted = {"VmHWM", "VmPeak", "VmRSS", "VmSize"}
    try:
        lines = (
            Path(f"/proc/{pid}/status")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    except OSError:
        return {}
    return {
        key: value.strip()
        for line in lines
        for key, value in [line.split(":", maxsplit=1)]
        if key in wanted
    }


def _host_info() -> dict[str, Any]:
    info: dict[str, Any] = {"logical_cpus": os.cpu_count()}
    try:
        cpu_lines = (
            Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
        )
        info["cpu_model"] = next(
            line.split(":", maxsplit=1)[1].strip()
            for line in cpu_lines
            if line.startswith("model name")
        )
    except (OSError, StopIteration):
        pass
    try:
        memory = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
        info["memory"] = {
            key: value.strip()
            for line in memory
            for key, value in [line.split(":", maxsplit=1)]
            if key in {"MemAvailable", "MemTotal", "SwapFree", "SwapTotal"}
        }
    except OSError:
        pass
    return info


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


def _run_once(
    checker: DataQualityChecker,
    spark: SparkSession,
    label: str,
    rows: int,
    expected_counts: dict[str, int],
    action_calls: dict[str, int],
) -> dict[str, Any]:
    before = dict(action_calls)
    spark.sparkContext.setJobGroup(label, label)
    started = perf_counter()
    report = checker.run(include_llm=False)
    elapsed = perf_counter() - started
    result = next(item for item in report.results if item.name == "ranges")
    counts = {
        item["column"]: item["violation_count"]
        for item in result.payload["per_column"]
    }
    actions = {key: action_calls[key] - before[key] for key in action_calls}
    assert report.dataset.row_count == rows
    assert counts == expected_counts
    assert actions == {"row_count": 1, "counts_outside": 1}
    assert result.status == "completed"
    assert result.severity == "error"
    assert all(not item["sample"] for item in result.payload["per_column"])
    return {
        "label": label,
        "quality_wall_seconds": round(elapsed, 6),
        "range_seconds": round(result.duration_seconds, 6),
        "engine_actions": actions,
        "spark_job_ids": sorted(
            spark.sparkContext.statusTracker().getJobIdsForGroup(label)
        ),
    }


def _benchmark(
    args: argparse.Namespace, spark: SparkSession
) -> dict[str, Any]:
    plan_started = perf_counter()
    frame = spark.range(args.rows, numPartitions=args.partitions).select(
        F.col("id"),
        *(
            F.pmod(F.col("id"), F.lit(modulus)).alias(column)
            for column, (modulus, _, _) in RANGE_SPECS.items()
        ),
    )
    plan_seconds = perf_counter() - plan_started
    expected_counts = {
        column: _expected_outside(args.rows, modulus, low, high)
        for column, (modulus, low, high) in RANGE_SPECS.items()
    }
    config = QualipilotConfig(
        engine="spark",
        checks=CheckConfig(
            missing_values=False,
            duplicates=False,
            data_types=False,
            outliers=False,
            cardinality=False,
            freshness=False,
            column_ranges={
                column: ColumnRange(min=low, max=high)
                for column, (_, low, high) in RANGE_SPECS.items()
            },
            sample_size=0,
        ),
    )
    checker = DataQualityChecker(frame, config, spark_session=spark)
    original_row_count = checker.engine.row_count
    original_counts_outside = checker.engine.counts_outside
    action_calls = {"row_count": 0, "counts_outside": 0}
    expected_ranges = {
        column: (float(low), float(high))
        for column, (_, low, high) in RANGE_SPECS.items()
    }

    def tracked_row_count() -> int:
        action_calls["row_count"] += 1
        return original_row_count()

    def tracked_counts_outside(
        ranges: dict[str, tuple[float, float]],
    ) -> dict[str, int]:
        action_calls["counts_outside"] += 1
        assert ranges == expected_ranges
        return original_counts_outside(ranges)

    def reject_scalar_call(*_args: Any, **_kwargs: Any) -> int:
        raise AssertionError("RangesCheck used scalar count_outside")

    checker.engine.row_count = tracked_row_count  # type: ignore[method-assign]
    checker.engine.counts_outside = (  # type: ignore[method-assign]
        tracked_counts_outside
    )
    checker.engine.count_outside = (  # type: ignore[method-assign]
        reject_scalar_call
    )

    cold = _run_once(
        checker, spark, "cold", args.rows, expected_counts, action_calls
    )
    warmup = _run_once(
        checker, spark, "warmup", args.rows, expected_counts, action_calls
    )
    trials = [
        _run_once(
            checker,
            spark,
            f"trial-{index}",
            args.rows,
            expected_counts,
            action_calls,
        )
        for index in range(1, args.trials + 1)
    ]
    expected_runs = args.trials + 2
    assert action_calls == {
        "row_count": expected_runs,
        "counts_outside": expected_runs,
    }
    seconds = [trial["quality_wall_seconds"] for trial in trials]
    return {
        "method": {
            "source": "spark.range",
            "rows": args.rows,
            "columns": ["id", *RANGE_SPECS],
            "partitions": frame.rdd.getNumPartitions(),
            "generated_columns": {
                column: {
                    "expression": f"id % {modulus}",
                    "allowed_inclusive": [low, high],
                }
                for column, (modulus, low, high) in RANGE_SPECS.items()
            },
            "enabled_checks": ["dataset_contract", "ranges"],
            "cached": bool(
                frame.storageLevel.useMemory or frame.storageLevel.useDisk
            ),
            "sample_size": 0,
            "row_collection": (
                "none; Spark collects only scalar aggregate result rows"
            ),
        },
        "results": {
            "status": "PASS",
            "expected_violation_counts": expected_counts,
            "actual_violation_counts": expected_counts,
            "cold": cold,
            "warmup": warmup,
            "trials": trials,
            "trial_wall_seconds": {
                "minimum": min(seconds),
                "median": statistics.median(seconds),
                "mean": statistics.fmean(seconds),
                "maximum": max(seconds),
            },
            "total_engine_actions": action_calls,
        },
        "lazy_plan_seconds": round(plan_seconds, 6),
    }


def main() -> None:
    args = _arguments()
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    session_started = perf_counter()
    spark: SparkSession | None = None
    try:
        spark = (
            SparkSession.builder.master(args.master)
            .appName("qualipilot-spark-range-benchmark")
            .config("spark.ui.enabled", "false")
            .config("spark.driver.memory", args.driver_memory)
            .config("spark.driver.maxResultSize", "64m")
            .config("spark.sql.shuffle.partitions", str(args.partitions))
            .config("spark.sql.adaptive.enabled", "true")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")
        session_seconds = perf_counter() - session_started
        benchmark = _benchmark(args, spark)
        gateway_process = getattr(spark.sparkContext._gateway, "proc", None)
        gateway_pid = getattr(gateway_process, "pid", None)
        jvm = spark.sparkContext._jvm
        if jvm is None:
            raise RuntimeError("Spark JVM bridge is unavailable")
        output = {
            "benchmark": "qualipilot Spark batched range checks",
            "recorded_at": datetime.now(UTC).isoformat(),
            "command": [sys.executable, *sys.argv],
            "environment": {
                "python": platform.python_version(),
                "qualipilot": version("qualipilot"),
                "pyspark": version("pyspark"),
                "spark": spark.version,
                "java": jvm.java.lang.System.getProperty("java.version"),
                "platform": platform.platform(),
                "conda_prefix": os.environ.get("CONDA_PREFIX"),
                "git": _git_info(),
                "host": _host_info(),
                "python_memory": _proc_status(os.getpid()),
                "spark_jvm_pid": gateway_pid,
                "spark_jvm_memory": _proc_status(gateway_pid),
            },
            "spark_config": {
                "master": spark.sparkContext.master,
                "default_parallelism": spark.sparkContext.defaultParallelism,
                "driver_memory": spark.conf.get("spark.driver.memory"),
                "driver_max_result_size": spark.sparkContext.getConf().get(
                    "spark.driver.maxResultSize"
                ),
                "shuffle_partitions": spark.conf.get(
                    "spark.sql.shuffle.partitions"
                ),
                "adaptive_enabled": spark.conf.get(
                    "spark.sql.adaptive.enabled"
                ),
            },
            "spark_session_start_seconds": round(session_seconds, 6),
            **benchmark,
        }
        print(json.dumps(output, indent=2, sort_keys=True))
    finally:
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    main()
