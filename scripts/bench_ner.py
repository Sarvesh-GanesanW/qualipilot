"""Benchmark exact-span NER quality and deterministic batch throughput.

Examples:
    python scripts/bench_ner.py
    python scripts/bench_ner.py --docs 100 --trials 1 --output ner.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any

from qualipilot import NamedEntity, SpacyEntityRecognizer

DEFAULT_MODEL = "en_core_web_sm"
DEFAULT_DOCUMENTS = 5_000
DEFAULT_TRIALS = 5
Span = tuple[int, int, str]
LabeledDocument = tuple[str, tuple[Span, ...]]

LABELED_CORPUS: tuple[LabeledDocument, ...] = (
    (
        "OpenAI is based in San Francisco.",
        ((0, 6, "ORG"), (19, 32, "GPE")),
    ),
    (
        "Ada Lovelace worked with Charles Babbage.",
        ((0, 12, "PERSON"), (25, 40, "PERSON")),
    ),
    (
        "Microsoft hired Satya Nadella in Seattle.",
        ((0, 9, "ORG"), (16, 29, "PERSON"), (33, 40, "GPE")),
    ),
    (
        "Apple released the iPhone in 2007.",
        ((0, 5, "ORG"), (19, 25, "PRODUCT"), (29, 33, "DATE")),
    ),
    (
        "Barack Obama visited Paris on Monday.",
        ((0, 12, "PERSON"), (21, 26, "GPE"), (30, 36, "DATE")),
    ),
    (
        "Tesla was founded by Elon Musk.",
        ((0, 5, "ORG"), (21, 30, "PERSON")),
    ),
    (
        "Google operates in India.",
        ((0, 6, "ORG"), (19, 24, "GPE")),
    ),
    ("March forward carefully.", ()),
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--docs", type=_positive_int, default=DEFAULT_DOCUMENTS
    )
    parser.add_argument("--trials", type=_positive_int, default=DEFAULT_TRIALS)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _assert_alignment(
    texts: Sequence[str],
    batches: Sequence[Sequence[NamedEntity]],
) -> int:
    assert len(batches) == len(texts)
    entity_count = 0
    for text, entities in zip(texts, batches, strict=True):
        for entity in entities:
            assert 0 <= entity.start_char < entity.end_char <= len(text)
            assert text[entity.start_char : entity.end_char] == entity.text
            entity_count += 1
    return entity_count


def _evaluate(recognizer: SpacyEntityRecognizer) -> dict[str, Any]:
    texts = [text for text, _ in LABELED_CORPUS]
    started = perf_counter()
    batches = recognizer.extract_many(texts)
    wall_seconds = perf_counter() - started
    _assert_alignment(texts, batches)

    expected = {
        (index, start, end, label)
        for index, (_, spans) in enumerate(LABELED_CORPUS)
        for start, end, label in spans
    }
    predicted = {
        (index, entity.start_char, entity.end_char, entity.label)
        for index, entities in enumerate(batches)
        for entity in entities
    }
    true_positives = len(expected & predicted)
    false_positives = len(predicted - expected)
    false_negatives = len(expected - predicted)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(expected) if expected else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    assert len(expected) == sum(len(spans) for _, spans in LABELED_CORPUS)
    return {
        "documents": len(texts),
        "expected_entities": len(expected),
        "predicted_entities": len(predicted),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "wall_seconds": round(wall_seconds, 6),
    }


def _throughput_documents(count: int) -> list[str]:
    templates = [text for text, _ in LABELED_CORPUS]
    return [templates[index % len(templates)] for index in range(count)]


def _run_trial(
    recognizer: SpacyEntityRecognizer,
    documents: Sequence[str],
    trial: int,
) -> dict[str, int | float]:
    gc.collect()
    started = perf_counter()
    batches = recognizer.extract_many(documents)
    wall_seconds = perf_counter() - started
    entities = _assert_alignment(documents, batches)
    characters = sum(map(len, documents))
    return {
        "trial": trial,
        "wall_seconds": round(wall_seconds, 6),
        "documents_per_second": round(len(documents) / wall_seconds, 3),
        "characters_per_second": round(characters / wall_seconds, 3),
        "entities": entities,
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


def _installed_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def _host_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
    }
    try:
        cpu_lines = (
            Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
        )
        info["cpu_model"] = next(
            line.split(":", maxsplit=1)[1].strip()
            for line in cpu_lines
            if line.startswith("model name")
        )
        memory_lines = (
            Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
        )
        total = next(
            line.split()[1]
            for line in memory_lines
            if line.startswith("MemTotal:")
        )
        info["memory_total_bytes"] = int(total) * 1024
    except (OSError, StopIteration):
        pass
    return info


def _process_memory() -> dict[str, int]:
    memory: dict[str, int] = {}
    try:
        for line in (
            Path("/proc/self/status").read_text(encoding="utf-8").splitlines()
        ):
            key, *values = line.replace(":", "").split()
            if key in {"VmHWM", "VmPeak", "VmRSS", "VmSize"} and values:
                memory[f"{key}_bytes"] = int(values[0]) * 1024
    except OSError:
        pass
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        memory["peak_rss_bytes"] = int(
            peak if sys.platform == "darwin" else peak * 1024
        )
    except (ImportError, OSError):
        pass
    return memory


def main() -> None:
    args = _arguments()
    memory_before_model = _process_memory()
    model_started = perf_counter()
    recognizer = SpacyEntityRecognizer(args.model)
    model_load_seconds = perf_counter() - model_started
    memory_after_model = _process_memory()

    evaluation = _evaluate(recognizer)
    documents = _throughput_documents(args.docs)
    warmup = _run_trial(recognizer, documents[: min(100, args.docs)], 0)
    trials = [
        _run_trial(recognizer, documents, trial)
        for trial in range(1, args.trials + 1)
    ]
    throughput = [float(trial["documents_per_second"]) for trial in trials]
    result = {
        "benchmark": "Qualipilot NER exact-span quality and throughput",
        "recorded_at": datetime.now(UTC).isoformat(),
        "command": [sys.executable, *sys.argv],
        "environment": {
            "python": platform.python_version(),
            "qualipilot": _installed_version("qualipilot"),
            "spacy": _installed_version("spacy"),
            "model": recognizer.metadata,
            "git": _git_info(),
            "host": _host_info(),
            "memory": {
                "before_model": memory_before_model,
                "after_model": memory_after_model,
                "after_benchmark": _process_memory(),
            },
        },
        "method": {
            "model": args.model,
            "labeled_corpus_sha256": hashlib.sha256(
                json.dumps(LABELED_CORPUS, separators=(",", ":")).encode()
            ).hexdigest(),
            "throughput_workload_sha256": hashlib.sha256(
                json.dumps(documents, separators=(",", ":")).encode()
            ).hexdigest(),
            "documents": args.docs,
            "trials": args.trials,
            "batch_size": 1_000,
            "n_process": 1,
            "matching": "exact character span and label",
            "llm": "disabled; benchmark calls only the local NER API",
        },
        "results": {
            "status": "PASS",
            "model_load_seconds": round(model_load_seconds, 6),
            "evaluation": evaluation,
            "warmup": warmup,
            "trials": trials,
            "documents_per_second": {
                "minimum": min(throughput),
                "median": statistics.median(throughput),
                "mean": statistics.fmean(throughput),
                "maximum": max(throughput),
            },
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
