"""Tests for the repeated-run benchmark gate."""

import argparse

import pytest
from scripts import bench_checker


def test_resilience_gate_includes_spark_jvm_memory() -> None:
    runs = [
        {
            "python_memory": {"VmHWM": "100 kB", "VmRSS": "90 kB"},
            "spark_jvm_memory": {"VmHWM": "1000 kB", "VmRSS": "900 kB"},
        },
        {
            "python_memory": {"VmHWM": "110 kB", "VmRSS": "95 kB"},
            "spark_jvm_memory": {"VmHWM": "3048 kB", "VmRSS": "1900 kB"},
        },
    ]
    memory = bench_checker._memory_metrics(runs)
    limits = argparse.Namespace(
        engine="spark",
        max_trial_slowdown=None,
        max_process_hwm_mib=None,
        max_hwm_growth_mib=1.0,
    )

    with pytest.raises(RuntimeError, match="spark_jvm HWM growth"):
        bench_checker._validate_resilience(limits, [1.0, 1.0], memory)
