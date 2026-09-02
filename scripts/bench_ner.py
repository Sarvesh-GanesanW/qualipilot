"""Gate NER quality on pinned Few-NERD data and measure throughput.

Examples:
    python scripts/bench_ner.py
    python scripts/bench_ner.py --docs 100 --trials 1 --output ner.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import hmac
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any

from qualipilot import NamedEntity, SpacyEntityRecognizer

if not __debug__:  # pragma: no cover - fail closed under python -O.
    raise RuntimeError("benchmark gates require assertions")

DEFAULT_MODEL = "en_core_web_sm"
DEFAULT_MODEL_VERSION = "3.8.0"
DEFAULT_MODEL_ARTIFACT_SHA256 = (
    "8dcaf7d0276e5e0adf7aba9daeb16042f69d962d9fa7f51c5e01822c17c47e10"
)
DEFAULT_MODEL_ARCHIVE_SHA256 = (
    "1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85"
)
DEFAULT_MODEL_URL = (
    "https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
)
DEFAULT_DOCUMENTS = 5_000
DEFAULT_TRIALS = 5
MAX_CORPUS_BYTES = 64 * 1024 * 1024
DEFAULT_MIN_PRECISION = 0.50
DEFAULT_MIN_RECALL = 0.45
DEFAULT_MIN_F1 = 0.47
DEFAULT_LABEL_F1 = {"PERSON": 0.62, "ORG": 0.28, "GPE": 0.52}

CORPUS_NAME = "Few-NERD supervised test"
CORPUS_REVISION = "205f3e9c9f3577ea2561d43f2f62dc249ab92d5b"
CORPUS_FILE = "supervised/test-00000-of-00001.parquet"
CORPUS_URL = (
    "https://huggingface.co/datasets/DFKI-SLT/few-nerd/resolve/"
    f"{CORPUS_REVISION}/{CORPUS_FILE}"
)
CORPUS_SHA256 = (
    "b7ad746fcbeb68fcc235ba7142d7c3723ea2dc39930089e947284defecf300c6"
)
CORPUS_SELECTION_SHA256 = (
    "ede4ba2d39f35c9a0843d803c65cddd68b794177a807a065b079f266a08a704c"
)
CORPUS_LICENSE = "CC BY-SA 4.0"
CORPUS_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
CORPUS_PROJECT_URL = "https://github.com/thunlp/Few-NERD"
CORPUS_PAPER = "https://doi.org/10.18653/v1/2021.acl-long.248"

EVALUATION_LABELS = ("PERSON", "ORG", "GPE")
REFERENCE_LABEL_BY_FINE_TAG: dict[int, str] = {
    21: "GPE",
    **dict.fromkeys(range(28, 38), "ORG"),
    **dict.fromkeys(range(50, 58), "PERSON"),
}
Span = tuple[int, int, str]
ScoredSpan = tuple[int, int, int, str]


@dataclass(frozen=True, slots=True)
class EvaluationDocument:
    """One reconstructed Few-NERD sentence and its exact reference spans."""

    source_id: str
    text: str
    spans: tuple[Span, ...]


@dataclass(frozen=True, slots=True)
class EvaluationCorpus:
    """The deterministic label-compatible subset used by the gate."""

    documents: tuple[EvaluationDocument, ...]
    source_documents: int
    excluded_documents: int
    selection_sha256: str


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _label_threshold(value: str) -> tuple[str, float]:
    label, separator, raw_threshold = value.partition("=")
    normalized_label = label.strip().upper()
    if separator != "=" or normalized_label not in EVALUATION_LABELS:
        choices = ", ".join(EVALUATION_LABELS)
        raise argparse.ArgumentTypeError(f"use LABEL=VALUE; labels: {choices}")
    return normalized_label, _probability(raw_threshold)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--expected-model-version", default=DEFAULT_MODEL_VERSION
    )
    parser.add_argument(
        "--expected-model-sha256", default=DEFAULT_MODEL_ARTIFACT_SHA256
    )
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--corpus-sha256", default=CORPUS_SHA256)
    parser.add_argument("--selection-sha256", default=CORPUS_SELECTION_SHA256)
    parser.add_argument(
        "--docs", type=_positive_int, default=DEFAULT_DOCUMENTS
    )
    parser.add_argument("--trials", type=_positive_int, default=DEFAULT_TRIALS)
    parser.add_argument(
        "--min-precision", type=_probability, default=DEFAULT_MIN_PRECISION
    )
    parser.add_argument(
        "--min-recall", type=_probability, default=DEFAULT_MIN_RECALL
    )
    parser.add_argument("--min-f1", type=_probability, default=DEFAULT_MIN_F1)
    parser.add_argument(
        "--min-label-f1",
        action="append",
        type=_label_threshold,
        metavar="LABEL=VALUE",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_corpus(destination: Path) -> None:
    request = urllib.request.Request(
        CORPUS_URL,
        headers={"User-Agent": "qualipilot-ner-benchmark/1"},
    )
    try:
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            destination.open("wb") as output,
        ):
            _copy_bounded(response, output)
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(
            f"unable to download pinned Few-NERD corpus: {exc}"
        ) from exc


def _copy_bounded(
    source: Any, output: Any, *, max_bytes: int = MAX_CORPUS_BYTES
) -> None:
    copied = 0
    while chunk := source.read(1024 * 1024):
        copied += len(chunk)
        if copied > max_bytes:
            raise RuntimeError(f"Few-NERD download exceeds {max_bytes} bytes")
        output.write(chunk)


def _verify_corpus(path: Path, expected_sha256: str) -> str:
    actual_sha256 = _sha256_file(path)
    normalized_expected = expected_sha256.strip().lower()
    if not hmac.compare_digest(actual_sha256, normalized_expected):
        raise ValueError(
            f"corpus SHA-256 {actual_sha256} does not match expected "
            f"{normalized_expected}"
        )
    return actual_sha256


def _load_corpus(path: Path) -> EvaluationCorpus:
    import polars as pl

    required = {"id", "tokens", "fine_ner_tags"}
    available = set(pl.scan_parquet(path).collect_schema().names())
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"Few-NERD corpus is missing columns: {missing}")
    frame = pl.read_parquet(path, columns=["id", "tokens", "fine_ner_tags"])
    selected: list[EvaluationDocument] = []
    excluded = 0
    for source_id, tokens, fine_tags in frame.iter_rows():
        document = _evaluation_document(str(source_id), tokens, fine_tags)
        if document is None:
            excluded += 1
        else:
            selected.append(document)
    if not selected:
        raise ValueError("Few-NERD corpus has no label-compatible documents")
    return EvaluationCorpus(
        documents=tuple(selected),
        source_documents=frame.height,
        excluded_documents=excluded,
        selection_sha256=_selection_sha256(selected),
    )


def _evaluation_document(
    source_id: str,
    tokens: Sequence[str],
    fine_tags: Sequence[int],
) -> EvaluationDocument | None:
    if len(tokens) != len(fine_tags):
        raise ValueError(f"token/tag length mismatch in document {source_id}")
    if not tokens or any(not token for token in tokens):
        raise ValueError(f"empty token sequence in document {source_id}")
    present_tags = {tag for tag in fine_tags if tag != 0}
    if not present_tags.issubset(REFERENCE_LABEL_BY_FINE_TAG):
        return None
    offsets = _token_offsets(tokens)
    spans = _reference_spans(tokens, fine_tags, offsets)
    return EvaluationDocument(source_id, " ".join(tokens), spans)


def _token_offsets(tokens: Sequence[str]) -> tuple[int, ...]:
    offsets: list[int] = []
    cursor = 0
    for token in tokens:
        offsets.append(cursor)
        cursor += len(token) + 1
    return tuple(offsets)


def _reference_spans(
    tokens: Sequence[str],
    fine_tags: Sequence[int],
    offsets: Sequence[int],
) -> tuple[Span, ...]:
    spans: list[Span] = []
    index = 0
    while index < len(tokens):
        fine_tag = fine_tags[index]
        if fine_tag == 0:
            index += 1
            continue
        end = index + 1
        while end < len(tokens) and fine_tags[end] == fine_tag:
            end += 1
        spans.append(
            (
                offsets[index],
                offsets[end - 1] + len(tokens[end - 1]),
                REFERENCE_LABEL_BY_FINE_TAG[fine_tag],
            )
        )
        index = end
    return tuple(spans)


def _selection_sha256(documents: Sequence[EvaluationDocument]) -> str:
    digest = hashlib.sha256()
    for document in documents:
        canonical = json.dumps(
            [document.source_id, document.text, document.spans],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest.update(canonical.encode())
        digest.update(b"\n")
    return digest.hexdigest()


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


def _score(
    expected: set[ScoredSpan], predicted: set[ScoredSpan]
) -> dict[str, int | float]:
    true_positives = len(expected & predicted)
    false_positives = len(predicted - expected)
    false_negatives = len(expected - predicted)
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "reference_entities": len(expected),
        "predicted_entities": len(predicted),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def _evaluate(
    recognizer: SpacyEntityRecognizer,
    corpus: EvaluationCorpus,
) -> dict[str, Any]:
    texts = [document.text for document in corpus.documents]
    started = perf_counter()
    batches = recognizer.extract_many(texts)
    wall_seconds = perf_counter() - started
    _assert_alignment(texts, batches)
    expected = {
        (document_index, start, end, label)
        for document_index, document in enumerate(corpus.documents)
        for start, end, label in document.spans
    }
    predicted = {
        (document_index, entity.start_char, entity.end_char, entity.label)
        for document_index, entities in enumerate(batches)
        for entity in entities
    }
    per_label = {
        label: _score(
            {span for span in expected if span[-1] == label},
            {span for span in predicted if span[-1] == label},
        )
        for label in EVALUATION_LABELS
    }
    return {
        "documents": len(corpus.documents),
        "wall_seconds": round(wall_seconds, 6),
        "matching": "exact reconstructed character span and mapped label",
        "micro": _score(expected, predicted),
        "macro_f1": round(
            statistics.fmean(
                float(metrics["f1"]) for metrics in per_label.values()
            ),
            6,
        ),
        "per_label": per_label,
    }


def _quality_gate(
    evaluation: dict[str, Any],
    *,
    min_precision: float,
    min_recall: float,
    min_f1: float,
    min_label_f1: dict[str, float],
) -> dict[str, Any]:
    failures: list[str] = []
    micro = evaluation["micro"]
    thresholds = {
        "micro_precision": min_precision,
        "micro_recall": min_recall,
        "micro_f1": min_f1,
        "per_label_f1": min_label_f1,
    }
    observed = {
        "micro_precision": float(micro["precision"]),
        "micro_recall": float(micro["recall"]),
        "micro_f1": float(micro["f1"]),
    }
    for metric, threshold in (
        ("micro_precision", min_precision),
        ("micro_recall", min_recall),
        ("micro_f1", min_f1),
    ):
        if observed[metric] < threshold:
            failures.append(
                f"{metric} {observed[metric]:.6f} is below {threshold:.6f}"
            )
    for label, threshold in min_label_f1.items():
        value = float(evaluation["per_label"][label]["f1"])
        if value < threshold:
            failures.append(f"{label} F1 {value:.6f} is below {threshold:.6f}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "thresholds": thresholds,
        "failures": failures,
    }


def _throughput_documents(corpus: EvaluationCorpus, count: int) -> list[str]:
    templates = [document.text for document in corpus.documents]
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


def _load_checked_corpus(
    args: argparse.Namespace,
) -> tuple[EvaluationCorpus, str]:
    if args.corpus is not None:
        checksum = _verify_corpus(args.corpus, args.corpus_sha256)
        corpus = _load_corpus(args.corpus)
    else:
        with tempfile.TemporaryDirectory(
            prefix="qualipilot-ner-"
        ) as directory:
            path = Path(directory) / "few-nerd-test.parquet"
            _download_corpus(path)
            checksum = _verify_corpus(path, args.corpus_sha256)
            corpus = _load_corpus(path)
    if not hmac.compare_digest(
        corpus.selection_sha256, args.selection_sha256.strip().lower()
    ):
        raise ValueError(
            f"selection SHA-256 {corpus.selection_sha256} does not match "
            f"expected {args.selection_sha256}"
        )
    return corpus, checksum


def _label_thresholds(args: argparse.Namespace) -> dict[str, float]:
    supplied = args.min_label_f1
    if supplied is None:
        return dict(DEFAULT_LABEL_F1)
    thresholds = dict(supplied)
    missing = sorted(set(EVALUATION_LABELS) - thresholds.keys())
    if missing:
        raise ValueError(f"missing --min-label-f1 thresholds: {missing}")
    return thresholds


def main() -> None:
    args = _arguments()
    preparation_started = perf_counter()
    corpus, corpus_sha256 = _load_checked_corpus(args)
    corpus_preparation_seconds = perf_counter() - preparation_started
    memory_before_model = _process_memory()
    model_started = perf_counter()
    recognizer = SpacyEntityRecognizer(
        args.model,
        labels=EVALUATION_LABELS,
        expected_version=args.expected_model_version,
        expected_sha256=args.expected_model_sha256,
    )
    model_load_seconds = perf_counter() - model_started
    memory_after_model = _process_memory()

    evaluation = _evaluate(recognizer, corpus)
    gate = _quality_gate(
        evaluation,
        min_precision=args.min_precision,
        min_recall=args.min_recall,
        min_f1=args.min_f1,
        min_label_f1=_label_thresholds(args),
    )
    documents = _throughput_documents(corpus, args.docs)
    warmup = _run_trial(recognizer, documents[: min(100, args.docs)], 0)
    trials = [
        _run_trial(recognizer, documents, trial)
        for trial in range(1, args.trials + 1)
    ]
    throughput = [float(trial["documents_per_second"]) for trial in trials]
    result = {
        "benchmark": "Qualipilot NER quality gate and throughput",
        "recorded_at": datetime.now(UTC).isoformat(),
        "command": [sys.executable, *sys.argv],
        "environment": {
            "python": platform.python_version(),
            "qualipilot": _installed_version("qualipilot"),
            "spacy": _installed_version("spacy"),
            "polars": _installed_version("polars"),
            "model": recognizer.metadata,
            "git": _git_info(),
            "host": _host_info(),
            "memory": {
                "before_model": memory_before_model,
                "after_model": memory_after_model,
                "after_benchmark": _process_memory(),
            },
        },
        "provenance": {
            "corpus": {
                "name": CORPUS_NAME,
                "project_url": CORPUS_PROJECT_URL,
                "paper": CORPUS_PAPER,
                "license": CORPUS_LICENSE,
                "license_url": CORPUS_LICENSE_URL,
                "distribution_url": CORPUS_URL,
                "distribution_revision": CORPUS_REVISION,
                "distribution_sha256": corpus_sha256,
            },
            "model": {
                "reference_release_url": (
                    DEFAULT_MODEL_URL if args.model == DEFAULT_MODEL else None
                ),
                "reference_release_archive_sha256": (
                    DEFAULT_MODEL_ARCHIVE_SHA256
                    if args.model == DEFAULT_MODEL
                    else None
                ),
                "expected_installed_artifact_sha256": (
                    args.expected_model_sha256
                ),
            },
        },
        "method": {
            "corpus_preparation_seconds": round(corpus_preparation_seconds, 6),
            "source_documents": corpus.source_documents,
            "evaluated_documents": len(corpus.documents),
            "excluded_incompatible_documents": corpus.excluded_documents,
            "selection_sha256": corpus.selection_sha256,
            "selection": (
                "retain sentences whose reference entities are only PERSON, "
                "ORG, or location-GPE; retain entity-free sentences"
            ),
            "text_reconstruction": "single ASCII space between source tokens",
            "reference_mapping": {
                "Few-NERD person-*": "PERSON",
                "Few-NERD organization-*": "ORG",
                "Few-NERD location-GPE": "GPE",
            },
            "matching": "exact reconstructed character span and mapped label",
            "gate_scope": (
                "fixed cross-domain regression floors, not production "
                "fitness, fairness, or calibration thresholds"
            ),
            "throughput_workload_sha256": hashlib.sha256(
                json.dumps(documents, separators=(",", ":")).encode()
            ).hexdigest(),
            "throughput_documents": args.docs,
            "throughput_trials": args.trials,
            "batch_size": 1_000,
            "n_process": 1,
            "llm": "disabled; benchmark calls only the local NER API",
        },
        "results": {
            "status": gate["status"],
            "quality_gate": gate,
            "model_load_seconds": round(model_load_seconds, 6),
            "evaluation": evaluation,
            "throughput": {
                "warmup": warmup,
                "trials": trials,
                "documents_per_second": {
                    "minimum": min(throughput),
                    "median": statistics.median(throughput),
                    "mean": statistics.fmean(throughput),
                    "maximum": max(throughput),
                },
            },
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if gate["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
