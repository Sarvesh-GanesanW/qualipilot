"""Benchmark the same deterministic quality workload on every engine.

Examples:
    python scripts/bench_checker.py --engine duckdb --profile ranges
    python scripts/bench_checker.py --engine dask --profile full --rows 5000000

Run engines in separate processes so eager frames and JVM memory do not
overlap.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from contextlib import suppress
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any

if not __debug__:  # pragma: no cover - fail closed under python -O.
    raise RuntimeError("benchmark gates require assertions")

ENGINES = ("pandas", "polars", "duckdb", "dask", "spark")
RANGE_SPECS = {
    "amount": (1_000, 0, 899),
    "quantity": (100, 10, 89),
    "score": (200, 20, 179),
}
FULL_CHECKS = (
    "dataset_contract",
    "missing_values",
    "duplicates",
    "data_types",
    "outliers",
    "ranges",
    "cardinality",
    "freshness",
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=ENGINES, required=True)
    parser.add_argument(
        "--profile", choices=("ranges", "full"), default="ranges"
    )
    parser.add_argument("--rows", type=_positive_int)
    parser.add_argument("--partitions", type=_positive_int, default=64)
    parser.add_argument("--threads", type=_positive_int, default=8)
    parser.add_argument("--trials", type=_positive_int, default=5)
    parser.add_argument(
        "--max-trial-slowdown",
        type=_positive_float,
        help="fail if the slowest measured trial exceeds this median multiple",
    )
    parser.add_argument(
        "--max-process-hwm-mib",
        type=_positive_float,
        help=(
            "fail if an observed Python or Spark JVM high-water mark "
            "exceeds this"
        ),
    )
    parser.add_argument(
        "--max-hwm-growth-mib",
        type=_positive_float,
        help="fail if a process high-water mark grows this much after warmup",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rows is None:
        args.rows = 100_000_000 if args.profile == "ranges" else 5_000_000
    if args.profile == "full" and args.rows < 20:
        parser.error("the full profile requires at least 20 rows")
    return args


def _expected_outside(rows: int, modulus: int, low: int, high: int) -> int:
    full_cycles, remainder = divmod(rows, modulus)
    allowed = full_cycles * (high - low + 1)
    allowed += max(0, min(remainder, high + 1) - low)
    return rows - allowed


def _pandas_partition(
    start: int,
    stop: int,
    profile: str,
    reference_epoch: int,
    total_rows: int,
) -> Any:
    import numpy as np
    import pandas as pd

    index = np.arange(start, stop, dtype=np.int64)
    if profile == "ranges":
        return pd.DataFrame(
            {
                "id": index,
                **{
                    column: index % modulus
                    for column, (modulus, _, _) in RANGE_SPECS.items()
                },
            },
            copy=False,
        )

    amount = (index % 1_000).astype(np.float64)
    amount[index % 1_000 < 10] = np.nan
    score = (index % 200).astype(np.float64)
    score[index % 1_000 == 0] = 10_000.0
    event_seconds = (
        reference_epoch
        - (index % 48) * 3_600
        + np.where(index % 1_000 == 0, 7_200, 0)
    )
    return pd.DataFrame(
        {
            "id": index,
            "entity_id": index % max(1, total_rows // 10),
            "amount": amount,
            "quantity": index % 100,
            "score": score,
            "event_time": pd.to_datetime(event_seconds, unit="s", utc=True),
        },
        copy=False,
    )


def _build_pandas(
    args: argparse.Namespace, reference_epoch: int
) -> tuple[Any, Any]:
    return (
        _pandas_partition(
            0,
            args.rows,
            args.profile,
            reference_epoch,
            args.rows,
        ),
        None,
    )


def _build_polars(
    args: argparse.Namespace, reference_epoch: int
) -> tuple[Any, Any]:
    import polars as pl

    frame = pl.DataFrame({"id": pl.arange(0, args.rows, eager=True)})
    row = pl.col("id")
    if args.profile == "ranges":
        return (
            frame.with_columns(
                (row % modulus).alias(column)
                for column, (modulus, _, _) in RANGE_SPECS.items()
            ),
            None,
        )

    seconds = (
        pl.lit(reference_epoch)
        - (row % 48) * 3_600
        + pl.when(row % 1_000 == 0).then(7_200).otherwise(0)
    )
    return (
        frame.with_columns(
            (row % max(1, args.rows // 10)).alias("entity_id"),
            pl.when(row % 1_000 < 10)
            .then(None)
            .otherwise((row % 1_000).cast(pl.Float64))
            .alias("amount"),
            (row % 100).alias("quantity"),
            pl.when(row % 1_000 == 0)
            .then(10_000.0)
            .otherwise((row % 200).cast(pl.Float64))
            .alias("score"),
            pl.from_epoch(seconds, time_unit="s")
            .dt.replace_time_zone("UTC")
            .alias("event_time"),
        ),
        None,
    )


def _build_duckdb(
    args: argparse.Namespace, reference_epoch: int
) -> tuple[Any, Any]:
    import duckdb

    connection = duckdb.connect(database=":memory:")
    connection.execute("SET threads = ?", [args.threads])
    if args.profile == "ranges":
        select = ", ".join(
            f"i % {modulus} AS {column}"
            for column, (modulus, _, _) in RANGE_SPECS.items()
        )
        query = f"SELECT i AS id, {select} FROM range(?) AS t(i)"
        return connection.sql(query, params=[args.rows]), connection

    query = """
        SELECT
            i AS id,
            i % ? AS entity_id,
            CASE WHEN i % 1000 < 10 THEN NULL
                 ELSE CAST(i % 1000 AS DOUBLE) END AS amount,
            i % 100 AS quantity,
            CASE WHEN i % 1000 = 0 THEN 10000.0
                 ELSE CAST(i % 200 AS DOUBLE) END AS score,
            to_timestamp(? - (i % 48) * 3600
                         + CASE WHEN i % 1000 = 0 THEN 7200 ELSE 0 END)
                AS event_time
        FROM range(?) AS t(i)
    """
    relation = connection.sql(
        query,
        params=[max(1, args.rows // 10), reference_epoch, args.rows],
    )
    return relation, connection


def _build_dask(
    args: argparse.Namespace, reference_epoch: int
) -> tuple[Any, Any]:
    import dask
    import dask.dataframe as dd
    from dask.delayed import delayed

    config = dask.config.set(
        {
            "scheduler": "threads",
            "num_workers": args.threads,
            "dataframe.convert-string": False,
        }
    )
    chunk = (args.rows + args.partitions - 1) // args.partitions
    bounds = [
        (start, min(start + chunk, args.rows))
        for start in range(0, args.rows, chunk)
    ]
    meta = _pandas_partition(
        0,
        0,
        args.profile,
        reference_epoch,
        args.rows,
    )
    partitions = [
        delayed(_pandas_partition)(
            start,
            stop,
            args.profile,
            reference_epoch,
            args.rows,
        )
        for start, stop in bounds
    ]
    return dd.from_delayed(partitions, meta=meta, verify_meta=True), config


def _build_spark(
    args: argparse.Namespace, reference_epoch: int
) -> tuple[Any, Any]:
    from pyspark.sql import SparkSession, functions

    spark = (
        SparkSession.builder.master(f"local[{args.threads}]")
        .appName(f"qualipilot-{args.profile}-benchmark")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.maxResultSize", "64m")
        .config("spark.sql.shuffle.partitions", str(args.partitions))
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    row = functions.col("id")
    frame = spark.range(args.rows, numPartitions=args.partitions)
    if args.profile == "ranges":
        return (
            frame.select(
                row,
                *(
                    functions.pmod(row, functions.lit(modulus)).alias(column)
                    for column, (modulus, _, _) in RANGE_SPECS.items()
                ),
            ),
            spark,
        )

    seconds = (
        functions.lit(reference_epoch)
        - functions.pmod(row, functions.lit(48)) * functions.lit(3_600)
        + functions.when(
            functions.pmod(row, functions.lit(1_000)) == 0,
            functions.lit(7_200),
        ).otherwise(functions.lit(0))
    )
    return (
        frame.select(
            row,
            functions.pmod(row, functions.lit(max(1, args.rows // 10))).alias(
                "entity_id"
            ),
            functions.when(
                functions.pmod(row, functions.lit(1_000)) < 10,
                functions.lit(None),
            )
            .otherwise(
                functions.pmod(row, functions.lit(1_000)).cast("double")
            )
            .alias("amount"),
            functions.pmod(row, functions.lit(100)).alias("quantity"),
            functions.when(
                functions.pmod(row, functions.lit(1_000)) == 0,
                functions.lit(10_000.0),
            )
            .otherwise(functions.pmod(row, functions.lit(200)).cast("double"))
            .alias("score"),
            functions.to_timestamp(functions.from_unixtime(seconds)).alias(
                "event_time"
            ),
        ),
        spark,
    )


def _build(args: argparse.Namespace, reference_epoch: int) -> tuple[Any, Any]:
    builders = {
        "pandas": _build_pandas,
        "polars": _build_polars,
        "duckdb": _build_duckdb,
        "dask": _build_dask,
        "spark": _build_spark,
    }
    return builders[args.engine](args, reference_epoch)


def _config(args: argparse.Namespace) -> Any:
    from qualipilot import CheckConfig, QualipilotConfig
    from qualipilot.models.config import ColumnRange

    ranges = {
        column: ColumnRange(min=low, max=high)
        for column, (_, low, high) in RANGE_SPECS.items()
    }
    if args.profile == "ranges":
        checks = CheckConfig(
            missing_values=False,
            duplicates=False,
            data_types=False,
            outliers=False,
            cardinality=False,
            freshness=False,
            min_rows=args.rows,
            required_columns=["id", *RANGE_SPECS],
            expected_dtypes={
                column: "integer" for column in ["id", *RANGE_SPECS]
            },
            column_ranges=ranges,
            sample_size=0,
        )
    else:
        checks = CheckConfig(
            min_rows=args.rows,
            required_columns=[
                "id",
                "entity_id",
                "amount",
                "quantity",
                "score",
                "event_time",
            ],
            expected_dtypes={
                "id": "integer",
                "entity_id": "integer",
                "amount": "float",
                "quantity": "integer",
                "score": "float",
                "event_time": "datetime",
            },
            duplicate_subset=["entity_id"],
            column_ranges=ranges,
            freshness=True,
            freshness_columns=["event_time"],
            freshness_max_age_hours=24,
            sample_size=0,
            include_top_values=False,
        )
    return QualipilotConfig(engine=args.engine, checks=checks)


def _range_counts(report: Any) -> dict[str, int]:
    result = next(item for item in report.results if item.name == "ranges")
    return {
        item["column"]: item["violation_count"]
        for item in result.payload["per_column"]
    }


def _validate(report: Any, args: argparse.Namespace) -> None:
    assert report.dataset.row_count == args.rows
    results = {item.name: item for item in report.results}
    expected_names = (
        {"dataset_contract", "ranges"}
        if args.profile == "ranges"
        else set(FULL_CHECKS)
    )
    assert set(results) == expected_names
    assert all(item.status == "completed" for item in results.values())
    assert results["dataset_contract"].payload["dtype_mismatches"] == []
    expected_ranges = {
        column: _expected_outside(args.rows, modulus, low, high)
        for column, (modulus, low, high) in RANGE_SPECS.items()
    }
    assert _range_counts(report) == expected_ranges
    if args.profile == "ranges":
        return

    full_cycles, remainder = divmod(args.rows, 1_000)
    expected_nulls = full_cycles * 10 + min(remainder, 10)
    assert (
        results["missing_values"].payload["total_null_count"] == expected_nulls
    )
    assert results["duplicates"].payload["total_duplicate_rows"] == args.rows
    assert results["outliers"].severity == "warn"
    freshness = results["freshness"].payload["per_column"]
    assert len(freshness) == 1
    assert freshness[0]["is_future"] is True


def _run_once(
    checker: Any,
    args: argparse.Namespace,
    label: str,
    spark_jvm_pid: int | None,
) -> dict[str, Any]:
    started = perf_counter()
    report = checker.run(include_llm=False)
    elapsed = perf_counter() - started
    _validate(report, args)
    result = {
        "label": label,
        "quality_wall_seconds": round(elapsed, 6),
        "logical_rows_per_second": round(args.rows / elapsed),
        "check_seconds": {
            item.name: round(item.duration_seconds, 6)
            for item in report.results
        },
        "severities": {item.name: item.severity for item in report.results},
        "range_violation_counts": _range_counts(report),
        "python_memory": _proc_status(os.getpid()),
    }
    if spark_jvm_pid is not None:
        result["spark_jvm_memory"] = _proc_status(spark_jvm_pid)
    return result


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


def _memory_kib(run: dict[str, Any], key: str, field: str) -> int | None:
    value = run.get(key, {}).get(field)
    if value is None:
        return None
    amount, unit = value.split()
    if unit != "kB":
        raise RuntimeError(f"unexpected /proc memory unit: {unit}")
    return int(amount)


def _memory_metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {}
    for label, key in (
        ("python", "python_memory"),
        ("spark_jvm", "spark_jvm_memory"),
    ):
        high_water = [_memory_kib(run, key, "VmHWM") for run in runs]
        rss = [_memory_kib(run, key, "VmRSS") for run in runs]
        if not any(value is not None for value in high_water):
            continue
        if any(value is None for value in [*high_water, *rss]):
            raise RuntimeError(f"incomplete {label} memory evidence")
        measured_hwm = [value for value in high_water if value is not None]
        measured_rss = [value for value in rss if value is not None]
        metrics[label] = {
            "peak_hwm_mib": round(max(measured_hwm) / 1024, 3),
            "hwm_growth_after_warmup_mib": round(
                (max(measured_hwm) - measured_hwm[0]) / 1024,
                3,
            ),
            "rss_change_after_warmup_mib": round(
                (measured_rss[-1] - measured_rss[0]) / 1024,
                3,
            ),
        }
    return metrics


def _validate_resilience(
    args: argparse.Namespace,
    walls: list[float],
    memory: dict[str, Any],
) -> dict[str, Any]:
    slowdown = max(walls) / statistics.median(walls)
    limits = {
        "max_trial_slowdown": args.max_trial_slowdown,
        "max_process_hwm_mib": args.max_process_hwm_mib,
        "max_hwm_growth_mib": args.max_hwm_growth_mib,
    }
    failures = []
    if (
        args.max_trial_slowdown is not None
        and slowdown > args.max_trial_slowdown
    ):
        failures.append(
            f"trial slowdown {slowdown:.3f} exceeds "
            f"{args.max_trial_slowdown:.3f}"
        )
    failures.extend(_memory_failures(args, memory))
    if failures:
        raise RuntimeError("resilience gate failed: " + "; ".join(failures))
    return {
        "status": "PASS",
        "trial_max_to_median": round(slowdown, 6),
        "process_memory": memory,
        "limits": limits,
        "memory_scope": "per process; not aggregate host or cgroup memory",
    }


def _memory_failures(
    args: argparse.Namespace, memory: dict[str, Any]
) -> list[str]:
    failures = []
    memory_limit_requested = any(
        limit is not None
        for limit in (args.max_process_hwm_mib, args.max_hwm_growth_mib)
    )
    required_memory = {"python"}
    if args.engine == "spark":
        required_memory.add("spark_jvm")
    missing_memory = required_memory - memory.keys()
    if memory_limit_requested and missing_memory:
        failures.append(
            "process memory evidence is unavailable for "
            + ", ".join(sorted(missing_memory))
        )
    for process, observed in memory.items():
        peak = observed["peak_hwm_mib"]
        growth = observed["hwm_growth_after_warmup_mib"]
        if (
            args.max_process_hwm_mib is not None
            and peak > args.max_process_hwm_mib
        ):
            failures.append(
                f"{process} HWM {peak:.3f} MiB exceeds "
                f"{args.max_process_hwm_mib:.3f} MiB"
            )
        if (
            args.max_hwm_growth_mib is not None
            and growth > args.max_hwm_growth_mib
        ):
            failures.append(
                f"{process} HWM growth {growth:.3f} MiB exceeds "
                f"{args.max_hwm_growth_mib:.3f} MiB"
            )
    return failures


def _host_info() -> dict[str, Any]:
    info: dict[str, Any] = {"logical_cpus": os.cpu_count()}
    try:
        cpu = Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
        info["cpu_model"] = next(
            line.split(":", maxsplit=1)[1].strip()
            for line in cpu
            if line.startswith("model name")
        )
        memory = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
        info["memory"] = {
            key: value.strip()
            for line in memory
            for key, value in [line.split(":", maxsplit=1)]
            if key in {"MemAvailable", "MemTotal", "SwapFree", "SwapTotal"}
        }
    except (OSError, StopIteration):
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


def _package_versions() -> dict[str, str]:
    found = {}
    for package in (
        "qualipilot",
        "numpy",
        "pandas",
        "polars",
        "duckdb",
        "dask",
        "pyspark",
    ):
        with suppress(PackageNotFoundError):
            found[package] = version(package)
    return found


def _runtime_shape(
    engine: str,
    frame: Any,
    owner: Any,
    requested_workers: int,
) -> dict[str, int | None]:
    partitions = None
    effective_workers = None
    if engine == "polars":
        import polars as pl

        effective_workers = pl.thread_pool_size()
    elif engine == "duckdb":
        effective_workers = int(
            owner.execute("SELECT current_setting('threads')").fetchone()[0]
        )
    elif engine == "dask":
        partitions = int(frame.npartitions)
        effective_workers = requested_workers
    elif engine == "spark":
        partitions = int(frame.rdd.getNumPartitions())
        effective_workers = int(owner.sparkContext.defaultParallelism)
    return {
        "partitions": partitions,
        "requested_workers": (
            None if engine == "pandas" else requested_workers
        ),
        "effective_workers": effective_workers,
    }


def _schema(profile: str) -> list[dict[str, str]]:
    if profile == "ranges":
        return [
            {
                "column": "id",
                "portable_dtype": "integer",
                "expression": "row index",
            },
            *[
                {
                    "column": column,
                    "portable_dtype": "integer",
                    "expression": f"id % {modulus}",
                }
                for column, (modulus, _, _) in RANGE_SPECS.items()
            ],
        ]
    return [
        {
            "column": "id",
            "portable_dtype": "integer",
            "expression": "row index",
        },
        {
            "column": "entity_id",
            "portable_dtype": "integer",
            "expression": "id % max(1, rows // 10)",
        },
        {
            "column": "amount",
            "portable_dtype": "float",
            "expression": (
                "NULL for the first 10 rows of each 1000-row cycle; "
                "otherwise id % 1000"
            ),
        },
        {
            "column": "quantity",
            "portable_dtype": "integer",
            "expression": "id % 100",
        },
        {
            "column": "score",
            "portable_dtype": "float",
            "expression": "10000 every 1000th row; otherwise id % 200",
        },
        {
            "column": "event_time",
            "portable_dtype": "datetime",
            "expression": (
                "run start - (id % 48) hours; +2 hours every 1000th row"
            ),
        },
    ]


def _cleanup(engine: str, checker: Any, owner: Any) -> None:
    checker.close()
    if engine == "spark":
        owner.stop()
    elif engine == "duckdb":
        owner.close()
    elif engine == "dask":
        owner.__exit__(None, None, None)


def main() -> None:
    args = _arguments()
    os.environ["POLARS_MAX_THREADS"] = str(args.threads)
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    reference_epoch = int(datetime.now(UTC).timestamp())
    build_started = perf_counter()
    frame, owner = _build(args, reference_epoch)
    build_seconds = perf_counter() - build_started

    from qualipilot import DataQualityChecker

    checker_started = perf_counter()
    checker = DataQualityChecker(
        frame,
        _config(args),
        spark_session=owner if args.engine == "spark" else None,
    )
    checker_init_seconds = perf_counter() - checker_started
    gateway = (
        getattr(owner.sparkContext._gateway, "proc", None)
        if args.engine == "spark"
        else None
    )
    spark_jvm_pid = getattr(gateway, "pid", None)
    try:
        gc.collect()
        cold = _run_once(checker, args, "cold", spark_jvm_pid)
        warmup = _run_once(checker, args, "warmup", spark_jvm_pid)
        trials = []
        for index in range(1, args.trials + 1):
            trials.append(
                _run_once(
                    checker,
                    args,
                    f"trial-{index}",
                    spark_jvm_pid,
                )
            )
            memory_failures = _memory_failures(
                args,
                _memory_metrics([warmup, *trials]),
            )
            if memory_failures:
                raise RuntimeError(
                    "resilience gate failed: " + "; ".join(memory_failures)
                )
        walls = [trial["quality_wall_seconds"] for trial in trials]
        names = list(trials[0]["check_seconds"])
        medians = {
            name: round(
                statistics.median(
                    trial["check_seconds"][name] for trial in trials
                ),
                6,
            )
            for name in names
        }
        memory = _memory_metrics([warmup, *trials])
        resilience = _validate_resilience(args, walls, memory)
        runtime_shape = _runtime_shape(
            args.engine,
            frame,
            owner,
            args.threads,
        )
        result = {
            "benchmark": "Qualipilot deterministic quality-check matrix",
            "recorded_at": datetime.now(UTC).isoformat(),
            "command": [sys.executable, *sys.argv],
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": _package_versions(),
                "git": _git_info(),
                "host": _host_info(),
                "python_memory": _proc_status(os.getpid()),
                "spark_jvm_memory": _proc_status(
                    getattr(gateway, "pid", None)
                ),
            },
            "method": {
                "engine": args.engine,
                "profile": args.profile,
                "rows": args.rows,
                **runtime_shape,
                "materialization": (
                    "eager" if args.engine in {"pandas", "polars"} else "lazy"
                ),
                "quantiles": checker.engine.quantile_provenance,
                "schema": _schema(args.profile),
                "enabled_checks": (
                    ["dataset_contract", "ranges"]
                    if args.profile == "ranges"
                    else list(FULL_CHECKS)
                ),
                "sample_size": 0,
                "llm": "disabled",
            },
            "results": {
                "status": "PASS",
                "build_or_plan_seconds": round(build_seconds, 6),
                "checker_init_seconds": round(checker_init_seconds, 6),
                "cold": cold,
                "warmup": warmup,
                "trials": trials,
                "trial_wall_seconds": {
                    "minimum": min(walls),
                    "median": statistics.median(walls),
                    "mean": statistics.fmean(walls),
                    "maximum": max(walls),
                },
                "median_logical_rows_per_second": round(
                    args.rows / statistics.median(walls)
                ),
                "median_check_seconds": medians,
                "actual_range_violation_counts": trials[-1][
                    "range_violation_counts"
                ],
                "resilience": resilience,
            },
        }
        rendered = json.dumps(result, indent=2, sort_keys=True)
        if args.output is not None:
            from qualipilot.checker import write_text_atomic

            write_text_atomic(args.output, rendered + "\n")
        print(rendered)
    finally:
        _cleanup(args.engine, checker, owner)


if __name__ == "__main__":
    main()
