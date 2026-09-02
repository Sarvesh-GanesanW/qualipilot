"""Reproducible local Lambda-handler benchmark with no AWS or LLM calls.

The parent process generates a deterministic CSV. A fresh child process then
uses the real Lambda handler and checker with a local S3-compatible test
double, so data generation does not pollute the handler's high-water memory
reading.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from shutil import copyfile
from time import perf_counter
from typing import Any

if not __debug__:  # pragma: no cover - fail closed under python -O.
    raise RuntimeError("benchmark gates require assertions")


class _S3Error(Exception):
    def __init__(self, code: str) -> None:
        self.response = {"Error": {"Code": code}}


class _LocalS3:
    def __init__(self, source: Path) -> None:
        self.source = source
        self.outputs: dict[str, dict[str, Any]] = {}
        self.report_bodies: dict[str, bytes] = {}

    def head_object(self, **kwargs: Any) -> dict[str, Any]:
        key = kwargs["Key"]
        if key == "input.csv":
            return {
                "ContentLength": self.source.stat().st_size,
                "ETag": '"benchmark"',
                "VersionId": "benchmark-v1",
            }
        try:
            return self.outputs[key]
        except KeyError as exc:
            raise _S3Error("404") from exc

    def download_file(
        self,
        bucket: str,
        key: str,
        filename: str,
        **kwargs: Any,
    ) -> None:
        assert bucket == "quality"
        assert key == "input.csv"
        assert kwargs == {"ExtraArgs": {"VersionId": "benchmark-v1"}}
        copyfile(self.source, filename)

    def put_object(self, **kwargs: Any) -> None:
        key = kwargs["Key"]
        assert kwargs["ServerSideEncryption"] == "AES256"
        assert kwargs["IfNoneMatch"] == "*"
        body = kwargs["Body"]
        assert isinstance(body, bytes)
        self.report_bodies[key] = body
        self.outputs[key] = {"Metadata": kwargs["Metadata"]}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=_positive_int, default=5_000_000)
    parser.add_argument("--trials", type=_positive_int, default=5)
    parser.add_argument("--threads", type=_positive_int, default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.rows < 2_000:
        parser.error("rows must be at least 2,000 for duplicate assertions")
    if (args.worker_input is None) != (args.worker_output is None):
        parser.error("worker input and output must be supplied together")
    return args


def _expected_outside(rows: int, modulus: int, low: int, high: int) -> int:
    cycles, remainder = divmod(rows, modulus)
    allowed = cycles * (high - low + 1)
    allowed += max(0, min(remainder, high + 1) - low)
    return rows - allowed


def _proc_status() -> dict[str, str]:
    wanted = {"VmHWM", "VmPeak", "VmRSS", "VmSize"}
    try:
        lines = (
            Path(f"/proc/{os.getpid()}/status")
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


def _git_info() -> dict[str, str | bool]:
    root = Path(__file__).parents[1]
    try:
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
    except (OSError, subprocess.CalledProcessError):
        return {}
    return {"commit": commit, "dirty": bool(dirty.strip())}


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
        info["memory_total"] = next(
            line.split(":", maxsplit=1)[1].strip()
            for line in memory
            if line.startswith("MemTotal")
        )
    except (OSError, StopIteration):
        pass
    return info


def _validate_report(body: bytes, rows: int) -> None:
    report = json.loads(body)
    assert report["dataset"]["row_count"] == rows
    results = {item["name"]: item for item in report["results"]}
    assert set(results) == {
        "dataset_contract",
        "missing_values",
        "duplicates",
        "data_types",
        "outliers",
        "ranges",
        "cardinality",
    }
    assert all(item["status"] == "completed" for item in results.values())
    assert results["duplicates"]["payload"]["total_duplicate_rows"] == rows
    actual = {
        item["column"]: item["violation_count"]
        for item in results["ranges"]["payload"]["per_column"]
    }
    assert actual == {
        "amount": _expected_outside(rows, 1_000, 0, 899),
        "quantity": _expected_outside(rows, 100, 10, 89),
    }


def _invoke(
    lambda_handler: Any,
    s3: _LocalS3,
    rows: int,
    label: str,
) -> dict[str, Any]:
    started = perf_counter()
    result = lambda_handler.handler(
        {
            "s3_uri": "s3://quality/input.csv",
            "output_key": f"reports/{label}.json",
            "fail_on": "none",
            "config": {
                "engine": "polars",
                "checks": {
                    "min_rows": rows,
                    "required_columns": ["id", "amount", "quantity"],
                    "expected_dtypes": {
                        "id": "integer",
                        "amount": "integer",
                        "quantity": "integer",
                    },
                    "duplicate_subset": ["amount", "quantity"],
                    "column_ranges": {
                        "amount": {"min": 0, "max": 899},
                        "quantity": {"min": 10, "max": 89},
                    },
                    "sample_size": 0,
                    "include_top_values": False,
                },
                "llm": {"provider": "none"},
            },
        },
        None,
    )
    elapsed = perf_counter() - started
    assert result["cached"] is False
    assert result["execution_failures"] == 0
    assert result["llm_status"] == "disabled"
    _validate_report(s3.report_bodies[result["output_key"]], rows)
    return {
        "label": label,
        "wall_seconds": round(elapsed, 6),
        "logical_rows_per_second": round(rows / elapsed),
        "report_bytes": len(s3.report_bodies[result["output_key"]]),
    }


def _worker(args: argparse.Namespace) -> None:
    assert args.worker_input is not None
    assert args.worker_output is not None
    os.environ["POLARS_MAX_THREADS"] = str(args.threads)
    os.environ["QUALIPILOT_LOG_LEVEL"] = "WARNING"
    os.environ["QUALIPILOT_MAX_INPUT_BYTES"] = str(256 * 1024 * 1024)
    os.environ["QUALIPILOT_MAX_DATASET_BYTES"] = str(1024 * 1024 * 1024)
    import_started = perf_counter()
    import polars as pl

    from qualipilot import lambda_handler

    import_seconds = perf_counter() - import_started
    s3 = _LocalS3(args.worker_input)
    lambda_handler._S3_CLIENT = s3
    cold = _invoke(lambda_handler, s3, args.rows, "cold")
    warmup = _invoke(lambda_handler, s3, args.rows, "warmup")
    trials = [
        _invoke(lambda_handler, s3, args.rows, f"trial-{index}")
        for index in range(1, args.trials + 1)
    ]
    walls = [trial["wall_seconds"] for trial in trials]
    result = {
        "benchmark": "Qualipilot local Lambda-handler pipeline",
        "recorded_at": datetime.now(UTC).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "qualipilot": version("qualipilot"),
            "polars": version("polars"),
            "git": _git_info(),
            "host": _host_info(),
            "python_memory": _proc_status(),
        },
        "method": {
            "rows": args.rows,
            "input_bytes": args.worker_input.stat().st_size,
            "schema": {
                "id": "row index (integer)",
                "amount": "id % 1000 (integer)",
                "quantity": "id % 100 (integer)",
            },
            "engine": "polars",
            "checks": [
                "dataset_contract",
                "missing_values",
                "duplicates",
                "data_types",
                "outliers",
                "ranges",
                "cardinality",
            ],
            "s3": "local contract-compatible test double",
            "s3_transport": (
                "download is local copyfile; upload is in-memory; "
                "no boto3, network, TLS, RIE, or AWS"
            ),
            "source_cache": (
                "generated immediately before the child process; "
                "OS cache state is not controlled"
            ),
            "requested_workers": args.threads,
            "effective_polars_workers": pl.thread_pool_size(),
            "limits": {
                "max_input_bytes": 256 * 1024 * 1024,
                "max_dataset_bytes": 1024 * 1024 * 1024,
            },
            "memory_scope": (
                "worker lifetime RSS including imports and all calls; "
                "not Lambda or cgroup memory"
            ),
            "llm": "disabled",
        },
        "results": {
            "status": "PASS",
            "module_import_seconds": round(import_seconds, 6),
            "first_handler_call_after_import": cold,
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
        },
    }
    args.worker_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _generate_csv(path: Path, rows: int) -> float:
    import polars as pl

    started = perf_counter()
    frame = pl.DataFrame({"id": pl.arange(0, rows, eager=True)})
    frame.with_columns(
        (pl.col("id") % 1_000).alias("amount"),
        (pl.col("id") % 100).alias("quantity"),
    ).write_csv(path)
    return perf_counter() - started


def main() -> None:
    args = _arguments()
    if args.worker_input is not None:
        _worker(args)
        return

    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory)
        source = temp / "input.csv"
        worker_output = temp / "result.json"
        generation_seconds = _generate_csv(source, args.rows)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--rows",
            str(args.rows),
            "--trials",
            str(args.trials),
            "--threads",
            str(args.threads),
            "--worker-input",
            str(source),
            "--worker-output",
            str(worker_output),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            sys.stderr.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            raise SystemExit(completed.returncode)
        result = json.loads(worker_output.read_text(encoding="utf-8"))
        result["command"] = [sys.executable, *sys.argv]
        result["setup"] = {
            "generated_csv_seconds": round(generation_seconds, 6),
            "excluded_from_handler_timings": True,
        }

    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
