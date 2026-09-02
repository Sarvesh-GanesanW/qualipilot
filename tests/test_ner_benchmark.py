"""Tests for the reproducible NER quality gate."""

from __future__ import annotations

import io
from pathlib import Path

import polars as pl
import pytest
from scripts import bench_ner


def test_corpus_download_copy_is_bounded() -> None:
    output = io.BytesIO()

    with pytest.raises(RuntimeError, match="download exceeds 3 bytes"):
        bench_ner._copy_bounded(io.BytesIO(b"four"), output, max_bytes=3)

    assert output.getvalue() == b""


def test_loads_only_label_compatible_few_nerd_documents(
    tmp_path: Path,
) -> None:
    corpus_path = tmp_path / "few-nerd.parquet"
    pl.DataFrame(
        {
            "id": ["mapped", "excluded", "empty"],
            "tokens": [
                ["Ada", "Lovelace", "joined", "OpenAI", "in", "Paris", "."],
                ["A", "film"],
                ["No", "entities", "."],
            ],
            "fine_ner_tags": [
                [50, 50, 0, 28, 0, 21, 0],
                [0, 2],
                [0, 0, 0],
            ],
        }
    ).write_parquet(corpus_path)

    corpus = bench_ner._load_corpus(corpus_path)

    assert corpus.source_documents == 3
    assert corpus.excluded_documents == 1
    assert len(corpus.documents) == 2
    assert corpus.documents[0].text == (
        "Ada Lovelace joined OpenAI in Paris ."
    )
    assert corpus.documents[0].spans == (
        (0, 12, "PERSON"),
        (20, 26, "ORG"),
        (30, 35, "GPE"),
    )
    assert corpus.documents[1].spans == ()
    assert len(corpus.selection_sha256) == 64


def test_rejects_invalid_corpus_shape_and_checksum(tmp_path: Path) -> None:
    corpus_path = tmp_path / "invalid.parquet"
    pl.DataFrame(
        {
            "id": ["bad"],
            "tokens": [["Ada", "Lovelace"]],
            "fine_ner_tags": [[50]],
        }
    ).write_parquet(corpus_path)

    with pytest.raises(ValueError, match="token/tag length mismatch"):
        bench_ner._load_corpus(corpus_path)
    with pytest.raises(ValueError, match="does not match expected"):
        bench_ner._verify_corpus(corpus_path, "0" * 64)


def test_scores_exact_spans_and_enforces_thresholds() -> None:
    expected = {
        (0, 0, 3, "PERSON"),
        (0, 10, 16, "ORG"),
    }
    predicted = {
        (0, 0, 3, "PERSON"),
        (0, 10, 16, "GPE"),
    }

    metrics = bench_ner._score(expected, predicted)

    assert metrics == {
        "reference_entities": 2,
        "predicted_entities": 2,
        "true_positives": 1,
        "false_positives": 1,
        "false_negatives": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }
    evaluation = {
        "micro": metrics,
        "per_label": {
            "PERSON": {"f1": 1.0},
            "ORG": {"f1": 0.0},
            "GPE": {"f1": 0.0},
        },
    }
    passed = bench_ner._quality_gate(
        evaluation,
        min_precision=0.5,
        min_recall=0.5,
        min_f1=0.5,
        min_label_f1={"PERSON": 1.0, "ORG": 0.0, "GPE": 0.0},
    )
    failed = bench_ner._quality_gate(
        evaluation,
        min_precision=0.6,
        min_recall=0.5,
        min_f1=0.5,
        min_label_f1={"PERSON": 1.0, "ORG": 0.1, "GPE": 0.0},
    )

    assert passed["status"] == "PASS"
    assert passed["failures"] == []
    assert failed["status"] == "FAIL"
    assert len(failed["failures"]) == 2
