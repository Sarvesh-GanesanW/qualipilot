"""Persisted-data correctness gate for Dask processes and Spark workers.

The default Spark mode launches a single-host standalone cluster with separate
master and worker JVMs. Pass ``--spark-master`` and a shared ``--data`` URI to
exercise an existing cluster; use ``--minimum-hosts 2`` for multi-host
evidence. Timings are diagnostic only and are not production capacity claims.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from shutil import which
from time import perf_counter
from typing import Any, Literal, TextIO, cast
from urllib.request import urlopen

if not __debug__:  # pragma: no cover - fail closed under python -O.
    raise RuntimeError("benchmark gates require assertions")

RANGE_SPECS = {
    "amount": (1_000, 0, 899),
    "quantity": (100, 10, 89),
    "score": (200, 20, 179),
}
DEFAULT_ROWS = 5_000_000
DEFAULT_PARTITIONS = 8
DEFAULT_WORKERS = 2
_SPARK_GROUP = "qualipilot-persisted-quality-gate"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("dask", "spark"), required=True)
    parser.add_argument("--rows", type=_positive_int, default=DEFAULT_ROWS)
    parser.add_argument(
        "--partitions", type=_positive_int, default=DEFAULT_PARTITIONS
    )
    parser.add_argument(
        "--workers", type=_positive_int, default=DEFAULT_WORKERS
    )
    parser.add_argument(
        "--data",
        help=(
            "existing deterministic Parquet directory/URI; a missing local "
            "path is generated, and omission uses a temporary directory"
        ),
    )
    parser.add_argument(
        "--spark-master",
        help=(
            "existing non-local Spark master; omission launches local "
            "standalone worker JVMs"
        ),
    )
    parser.add_argument(
        "--minimum-hosts",
        type=_positive_int,
        default=1,
        help="minimum distinct Spark executor hosts required",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.workers < 2:
        parser.error("workers must be at least 2 for distributed evidence")
    if args.partitions < args.workers:
        parser.error("partitions must be at least workers")
    if args.engine == "dask" and args.spark_master is not None:
        parser.error("--spark-master applies only to Spark")
    if args.engine == "dask" and args.minimum_hosts != 1:
        parser.error("--minimum-hosts applies only to Spark")
    if args.engine == "spark" and args.spark_master:
        if args.spark_master.startswith("local"):
            parser.error("local[...] is threaded; use standalone workers")
        if args.data is None:
            parser.error("an existing shared --data URI is required")
    return args


def _expected_outside(rows: int, modulus: int, low: int, high: int) -> int:
    cycles, remainder = divmod(rows, modulus)
    allowed = cycles * (high - low + 1)
    allowed += max(0, min(remainder, high + 1) - low)
    return rows - allowed


def _generate_parquet(path: Path, rows: int, partitions: int) -> float:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    started = perf_counter()
    path.mkdir(parents=True)
    schema = pa.schema(
        [(column, pa.int64()) for column in ["id", *RANGE_SPECS]]
    )
    for partition in range(partitions):
        first = rows * partition // partitions
        stop = rows * (partition + 1) // partitions
        target = path / f"part-{partition:05}.parquet"
        with pq.ParquetWriter(target, schema, compression="snappy") as writer:
            for start in range(first, stop, 1_000_000):
                ids = np.arange(
                    start, min(start + 1_000_000, stop), dtype=np.int64
                )
                writer.write_table(
                    pa.table(
                        {
                            "id": ids,
                            **{
                                column: ids % modulus
                                for column, (
                                    modulus,
                                    _,
                                    _,
                                ) in RANGE_SPECS.items()
                            },
                        },
                        schema=schema,
                    )
                )
    return perf_counter() - started


def _prepare_data(
    requested: str | None,
    temporary_root: Path,
    rows: int,
    partitions: int,
) -> tuple[str, dict[str, Any]]:
    generated = False
    generation_seconds: float | None = None
    if requested is None:
        path = temporary_root / "input.parquet"
        generation_seconds = _generate_parquet(path, rows, partitions)
        data = str(path)
        generated = True
    elif "://" in requested:
        data = requested
        path = None
    else:
        path = Path(requested).resolve()
        if not path.exists():
            generation_seconds = _generate_parquet(path, rows, partitions)
            generated = True
        data = str(path)

    files = sorted(path.glob("*.parquet")) if path is not None else []
    if path is not None and not files:
        raise ValueError(f"no Parquet files found in {path}")
    if "://" in data:
        from qualipilot.engines._file_formats import require_safe_remote_url

        require_safe_remote_url(data)
    return data, {
        "uri": data,
        "generated": generated,
        "generation_seconds": (
            None
            if generation_seconds is None
            else round(generation_seconds, 6)
        ),
        "generation_excluded_from_quality_timing": True,
        "files": len(files) if path is not None else None,
        "bytes": sum(file.stat().st_size for file in files) if files else None,
        "cache_state": "uncontrolled; newly generated local data may be hot",
    }


def _quality_config(engine: Literal["dask", "spark"], rows: int) -> Any:
    from qualipilot import CheckConfig, QualipilotConfig
    from qualipilot.models.config import ColumnRange

    return QualipilotConfig(
        engine=engine,
        checks=CheckConfig(
            missing_values=False,
            duplicates=False,
            data_types=False,
            outliers=False,
            cardinality=False,
            freshness=False,
            min_rows=rows,
            required_columns=["id", *RANGE_SPECS],
            expected_dtypes={
                column: "integer" for column in ["id", *RANGE_SPECS]
            },
            column_ranges={
                column: ColumnRange(min=low, max=high)
                for column, (_, low, high) in RANGE_SPECS.items()
            },
            sample_size=0,
        ),
    )


def _run_quality(checker: Any, engine: str, rows: int) -> dict[str, Any]:
    started = perf_counter()
    report = checker.run(include_llm=False)
    elapsed = perf_counter() - started
    results = {item.name: item for item in report.results}
    ranges = results["ranges"]
    actual = {
        item["column"]: item["violation_count"]
        for item in ranges.payload["per_column"]
    }
    expected = {
        column: _expected_outside(rows, modulus, low, high)
        for column, (modulus, low, high) in RANGE_SPECS.items()
    }
    assert report.dataset.engine == engine
    assert report.dataset.row_count == rows
    assert set(results) == {"dataset_contract", "ranges"}
    assert all(item.status == "completed" for item in results.values())
    assert results["dataset_contract"].payload["dtype_mismatches"] == []
    assert actual == expected
    assert ranges.severity == "error"
    return {
        "status": "PASS",
        "quality_wall_seconds": round(elapsed, 6),
        "logical_rows_per_second": round(rows / elapsed),
        "expected_range_violation_counts": expected,
        "actual_range_violation_counts": actual,
        "check_seconds": {
            item.name: round(item.duration_seconds, 6)
            for item in report.results
        },
    }


def _mark_dask_worker(
    partition: Any, marker_dir: str, required_workers: int
) -> Any:
    marker = Path(marker_dir) / str(os.getpid())
    marker.touch(exist_ok=True)
    deadline = time.monotonic() + 10
    while len(tuple(Path(marker_dir).iterdir())) < required_workers:
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Dask did not schedule {required_workers} worker processes"
            )
        time.sleep(0.01)
    return partition


def _marker_pids(path: Path) -> list[int]:
    return sorted(int(marker.name) for marker in path.iterdir())


def _clear_markers(path: Path) -> None:
    for marker in path.iterdir():
        marker.unlink()


def _run_dask(
    args: argparse.Namespace,
    data: str,
    temporary_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import dask
    import dask.dataframe as dd

    from qualipilot import DataQualityChecker

    marker_dir = temporary_root / "dask-worker-markers"
    marker_dir.mkdir()
    with dask.config.set(
        {
            "scheduler": "processes",
            "num_workers": args.workers,
            "chunksize": 1,
            "multiprocessing.context": "spawn",
            "dataframe.convert-string": False,
        }
    ):
        frame = dd.read_parquet(data).repartition(npartitions=args.partitions)
        frame = frame.map_partitions(
            _mark_dask_worker,
            str(marker_dir),
            args.workers,
            meta=frame._meta,
        )
        checker = DataQualityChecker(
            frame,
            _quality_config("dask", args.rows),
            source=data,
        )
        evidence: dict[str, list[int]] = {}
        original_row_count = checker.engine.row_count
        original_counts_outside = checker.engine.counts_outside

        def tracked_row_count() -> int:
            _clear_markers(marker_dir)
            result = original_row_count()
            evidence["row_count"] = _marker_pids(marker_dir)
            return result

        def tracked_counts_outside(
            ranges: dict[str, tuple[float, float]],
        ) -> dict[str, int]:
            _clear_markers(marker_dir)
            result = original_counts_outside(ranges)
            evidence["ranges"] = _marker_pids(marker_dir)
            return result

        checker.engine.row_count = tracked_row_count  # type: ignore[method-assign]
        checker.engine.counts_outside = tracked_counts_outside  # type: ignore[method-assign]
        try:
            result = _run_quality(checker, "dask", args.rows)
        finally:
            checker.close()

    for action, worker_pids in evidence.items():
        assert os.getpid() not in worker_pids
        assert len(worker_pids) >= args.workers, (
            f"{action} used {len(worker_pids)} worker processes; "
            f"required {args.workers}"
        )
    return result, {
        "scheduler": "dask multiprocessing",
        "configured_workers": args.workers,
        "parent_pid": os.getpid(),
        "worker_pids_by_action": evidence,
        "distinct_worker_pids": sorted(
            {pid for pids in evidence.values() for pid in pids}
        ),
        "distinct_hosts": [socket.gethostname()],
        "scope": "single-host, separate Python worker processes",
        "probe_overhead": (
            "one marker-file touch per partition and a worker rendezvous "
            "per action"
        ),
    }


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _spark_class() -> str:
    executable = which("spark-class")
    if executable is not None:
        return executable
    import pyspark

    bundled = Path(pyspark.__file__).parent / "bin" / "spark-class"
    if bundled.is_file():
        return str(bundled)
    raise RuntimeError(
        "spark-class was not found in PATH or the PySpark install"
    )


def _start_spark_process(
    spark_class: str,
    arguments: list[str],
    log_path: Path,
    environment: dict[str, str],
) -> tuple[subprocess.Popen[str], TextIO]:
    log = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [spark_class, *arguments],
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
        start_new_session=True,
    )
    return process, log


def _master_state(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=1) as response:
        return cast(dict[str, Any], json.load(response))


def _wait_for_spark_workers(
    master_ui: str,
    workers: int,
    processes: list[subprocess.Popen[str]],
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        exited = [
            process.pid for process in processes if process.poll() is not None
        ]
        if exited:
            raise RuntimeError(
                f"Spark daemon exited before readiness: {exited}"
            )
        try:
            state = _master_state(master_ui)
            alive = [
                worker
                for worker in state["workers"]
                if worker["state"] == "ALIVE"
            ]
            if len(alive) >= workers:
                return alive
        except (OSError, KeyError, json.JSONDecodeError):
            pass
        time.sleep(0.1)
    raise RuntimeError(
        f"Spark master did not register {workers} workers in 30 seconds"
    )


def _stop_spark_processes(
    processes: list[tuple[subprocess.Popen[str], TextIO]],
) -> None:
    for process, _ in reversed(processes):
        if process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
    for process, log in reversed(processes):
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        log.close()


@contextmanager
def _local_spark_cluster(
    workers: int,
    temporary_root: Path,
) -> Iterator[tuple[str, dict[str, Any]]]:
    if os.name != "posix":
        raise RuntimeError("local standalone Spark launch requires POSIX")
    spark_class = _spark_class()
    rpc_port = _free_port()
    web_port = _free_port()
    master = f"spark://127.0.0.1:{rpc_port}"
    master_ui = f"http://127.0.0.1:{web_port}/json/"
    environment = dict(os.environ)
    environment["SPARK_LOCAL_IP"] = "127.0.0.1"
    processes: list[tuple[subprocess.Popen[str], TextIO]] = []
    try:
        processes.append(
            _start_spark_process(
                spark_class,
                [
                    "org.apache.spark.deploy.master.Master",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(rpc_port),
                    "--webui-port",
                    str(web_port),
                ],
                temporary_root / "spark-master.log",
                environment,
            )
        )
        for index in range(workers):
            processes.append(
                _start_spark_process(
                    spark_class,
                    [
                        "org.apache.spark.deploy.worker.Worker",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        "0",
                        "--webui-port",
                        str(_free_port()),
                        "--cores",
                        "1",
                        "--memory",
                        "768m",
                        "--work-dir",
                        str(temporary_root / f"spark-worker-{index}"),
                        master,
                    ],
                    temporary_root / f"spark-worker-{index}.log",
                    environment,
                )
            )
        alive = _wait_for_spark_workers(
            master_ui,
            workers,
            [process for process, _ in processes],
        )
        yield (
            master,
            {
                "mode": "local-standalone",
                "master_pid": processes[0][0].pid,
                "worker_daemon_pids": [
                    process.pid for process, _ in processes[1:]
                ],
                "registered_worker_ids": sorted(
                    worker["id"] for worker in alive
                ),
                "worker_memory_mib": 768,
            },
        )
    finally:
        _stop_spark_processes(processes)


def _spark_task_evidence(event_dir: Path) -> dict[str, Any]:
    job_ids: list[int] = []
    stage_ids: set[int] = set()
    tasks: list[dict[str, Any]] = []
    for event_file in event_dir.rglob("*"):
        if not event_file.is_file() or event_file.name.startswith("."):
            continue
        with event_file.open(encoding="utf-8") as events:
            for line in events:
                event = json.loads(line)
                if (
                    event.get("Event") == "SparkListenerJobStart"
                    and event.get("Properties", {}).get("spark.jobGroup.id")
                    == _SPARK_GROUP
                ):
                    job_ids.append(int(event["Job ID"]))
                    stage_ids.update(map(int, event["Stage IDs"]))
                elif event.get("Event") == "SparkListenerTaskEnd":
                    task = event["Task Info"]
                    tasks.append(
                        {
                            "stage": int(event["Stage ID"]),
                            "executor": str(task["Executor ID"]),
                            "host": str(task["Host"]),
                        }
                    )
    quality_tasks = [task for task in tasks if task["stage"] in stage_ids]
    return {
        "job_ids": sorted(job_ids),
        "stage_ids": sorted(stage_ids),
        "task_count": len(quality_tasks),
        "executor_ids": sorted({task["executor"] for task in quality_tasks}),
        "executor_hosts": sorted({task["host"] for task in quality_tasks}),
    }


def _run_spark(
    args: argparse.Namespace,
    data: str,
    event_dir: Path,
    master: str,
    local_cluster: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from pyspark.sql import SparkSession

    from qualipilot import DataQualityChecker

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    if local_cluster:
        os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    builder = (
        SparkSession.builder.master(master)
        .appName("qualipilot-persisted-quality-gate")
        .config("spark.ui.enabled", "false")
        .config("spark.executor.instances", str(args.workers))
        .config("spark.executor.cores", "1")
        .config("spark.cores.max", str(args.workers))
        .config("spark.sql.shuffle.partitions", str(args.partitions))
        .config("spark.eventLog.enabled", "true")
        .config("spark.eventLog.dir", event_dir.as_uri())
        .config("spark.eventLog.compress", "false")
        .config("spark.eventLog.rolling.enabled", "false")
    )
    if local_cluster:
        builder = (
            builder.config("spark.executor.memory", "512m")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.bindAddress", "127.0.0.1")
        )
    spark = builder.getOrCreate()
    try:
        spark.sparkContext.setLogLevel("ERROR")
        frame = spark.read.parquet(data).repartition(args.partitions)
        checker = DataQualityChecker(
            frame,
            _quality_config("spark", args.rows),
            source=data,
            spark_session=spark,
        )
        spark.sparkContext.setJobGroup(_SPARK_GROUP, _SPARK_GROUP)
        try:
            result = _run_quality(checker, "spark", args.rows)
        finally:
            spark.sparkContext.setLocalProperty(
                "spark.jobGroup.id",
                cast(str, None),  # PySpark uses null to clear this property.
            )
            checker.close()
        spark_version = spark.version
        default_parallelism = spark.sparkContext.defaultParallelism
        actual_master = spark.sparkContext.master
        jvm = spark.sparkContext._jvm
        if jvm is None:
            raise RuntimeError("Spark JVM bridge is unavailable")
        java_version = jvm.java.lang.System.getProperty("java.version")
    finally:
        spark.stop()

    evidence = _spark_task_evidence(event_dir)
    assert len(evidence["executor_ids"]) >= args.workers, (
        f"quality jobs used {len(evidence['executor_ids'])} executors; "
        f"required {args.workers}"
    )
    assert "driver" not in evidence["executor_ids"]
    assert len(evidence["executor_hosts"]) >= args.minimum_hosts, (
        f"quality jobs used {len(evidence['executor_hosts'])} hosts; "
        f"required {args.minimum_hosts}"
    )
    return result, {
        "scheduler": "Spark standalone/cluster",
        "spark": spark_version,
        "java": java_version,
        "master": actual_master,
        "configured_workers": args.workers,
        "default_parallelism": default_parallelism,
        **evidence,
        "minimum_hosts_required": args.minimum_hosts,
        "scope": (
            "single-host, separate Spark worker/executor JVMs"
            if local_cluster
            else "externally managed Spark cluster"
        ),
        "evidence_source": "Spark event log for the quality-check job group",
    }


def _environment(engine: str) -> dict[str, Any]:
    package = "pyspark" if engine == "spark" else engine
    packages = {
        "qualipilot": version("qualipilot"),
        package: version(package),
    }
    host: dict[str, Any] = {
        "name": socket.gethostname(),
        "logical_cpus": os.cpu_count(),
    }
    try:
        cpu = Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
        host["cpu_model"] = next(
            line.split(":", maxsplit=1)[1].strip()
            for line in cpu
            if line.startswith("model name")
        )
        memory = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
        host["memory_total"] = next(
            line.split(":", maxsplit=1)[1].strip()
            for line in memory
            if line.startswith("MemTotal")
        )
    except (OSError, StopIteration):
        pass
    try:
        root = Path(__file__).parents[1]
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--short"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        git: dict[str, Any] = {"commit": commit, "dirty": bool(dirty.strip())}
    except (OSError, subprocess.CalledProcessError):
        git = {}
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "host": host,
        "packages": packages,
        "git": git,
    }


def main() -> None:
    args = _arguments()
    with tempfile.TemporaryDirectory(
        prefix="qualipilot-distributed-"
    ) as raw_temp:
        temporary_root = Path(raw_temp)
        data, dataset = _prepare_data(
            args.data,
            temporary_root,
            args.rows,
            args.partitions,
        )
        if args.engine == "dask":
            result, execution = _run_dask(args, data, temporary_root)
        elif args.spark_master is not None:
            event_dir = temporary_root / "spark-events"
            event_dir.mkdir()
            result, execution = _run_spark(
                args,
                data,
                event_dir,
                args.spark_master,
                False,
            )
        else:
            event_dir = temporary_root / "spark-events"
            event_dir.mkdir()
            with _local_spark_cluster(args.workers, temporary_root) as (
                master,
                cluster,
            ):
                result, execution = _run_spark(
                    args,
                    data,
                    event_dir,
                    master,
                    True,
                )
                execution.update(cluster)

        report = {
            "benchmark": (
                "Qualipilot persisted-data distributed correctness gate"
            ),
            "recorded_at": datetime.now(UTC).isoformat(),
            "command": [sys.executable, *sys.argv],
            "environment": _environment(args.engine),
            "method": {
                "engine": args.engine,
                "rows": args.rows,
                "requested_partitions": args.partitions,
                "enabled_checks": ["dataset_contract", "ranges"],
                "input": dataset,
                "execution": execution,
                "timed_scope": "one instrumented DataQualityChecker.run",
                "llm": "disabled",
                "claim_boundary": (
                    "integration evidence only; not production capacity, "
                    "availability, remote-storage, or network evidence"
                ),
            },
            "results": result,
        }

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
