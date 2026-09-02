"""NER API and CLI tests against a deterministic local spaCy pipeline."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from qualipilot import SpacyEntityRecognizer
from qualipilot.cli import app

spacy = pytest.importorskip("spacy")

runner = CliRunner()


@pytest.fixture
def ner_model(tmp_path: Path) -> Path:
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(
        [
            {"label": "ORG", "pattern": "OpenAI"},
            {"label": "PERSON", "pattern": "Ada Lovelace"},
            {"label": "GPE", "pattern": "Pune"},
        ]
    )
    nlp.meta["name"] = "qualipilot_test_ner"
    nlp.meta["version"] = "1.2.3"
    path = tmp_path / "model"
    nlp.to_disk(path)
    return path


def test_extracts_stable_character_offsets(ner_model: Path) -> None:
    text = "OpenAI hired Ada Lovelace in Pune."
    recognizer = SpacyEntityRecognizer(ner_model)

    entities = recognizer.extract(text)

    assert [(item.text, item.label) for item in entities] == [
        ("OpenAI", "ORG"),
        ("Ada Lovelace", "PERSON"),
        ("Pune", "GPE"),
    ]
    assert all(
        text[item.start_char : item.end_char] == item.text for item in entities
    )
    metadata = recognizer.metadata
    artifact_sha256 = metadata.pop("artifact_sha256")
    assert len(artifact_sha256) == 64
    assert recognizer.metadata["artifact_sha256"] == artifact_sha256
    assert metadata == {
        "source": str(ner_model),
        "spacy_version": version("spacy"),
        "name": "qualipilot_test_ner",
        "version": "1.2.3",
        "artifact_sha256_scope": (
            "model-tree files excluding __pycache__ and .pyc"
        ),
        "license": "unknown",
        "language": "en",
        "pipeline": ["entity_ruler"],
    }


def test_enforces_pipeline_version_and_artifact_hash(
    ner_model: Path,
) -> None:
    recognizer = SpacyEntityRecognizer(ner_model)
    artifact_sha256 = str(recognizer.metadata["artifact_sha256"])

    verified = SpacyEntityRecognizer(
        ner_model,
        expected_version="1.2.3",
        expected_sha256=artifact_sha256.upper(),
    )

    assert verified.metadata["artifact_sha256"] == artifact_sha256
    with pytest.raises(ValueError, match=r"version.*does not match"):
        SpacyEntityRecognizer(ner_model, expected_version="9.9.9")
    with pytest.raises(ValueError, match=r"SHA-256.*does not match"):
        SpacyEntityRecognizer(ner_model, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="64-character hex digest"):
        SpacyEntityRecognizer(ner_model, expected_sha256="not-a-digest")


def test_batches_and_filters_labels(ner_model: Path) -> None:
    recognizer = SpacyEntityRecognizer(ner_model, labels=["ORG"])

    batches = recognizer.extract_many(["OpenAI in Pune", "Ada Lovelace"])

    assert [[item.text for item in batch] for batch in batches] == [
        ["OpenAI"],
        [],
    ]


def test_rejects_unknown_label_filter(ner_model: Path) -> None:
    with pytest.raises(ValueError, match=r"not provided.*TYPO"):
        SpacyEntityRecognizer(ner_model, labels=["TYPO"])


def test_rejects_invalid_text_and_pipeline(tmp_path: Path) -> None:
    blank = spacy.blank("en")
    model_path = tmp_path / "blank"
    blank.to_disk(model_path)

    with pytest.raises(ValueError, match="no enabled 'ner'"):
        SpacyEntityRecognizer(model_path)


def test_rejects_non_string_batch_item(ner_model: Path) -> None:
    recognizer = SpacyEntityRecognizer(ner_model)

    with pytest.raises(TypeError, match="position 1 must be str"):
        recognizer.extract_many(["OpenAI", 42])  # type: ignore[list-item]


def test_ner_cli_writes_reproducible_audit(
    tmp_path: Path, ner_model: Path
) -> None:
    input_path = tmp_path / "notes.csv"
    output_path = tmp_path / "entities.json"
    input_path.write_text(
        'id,note\nr1,"OpenAI is in Pune"\nr2,\nr3,"Ada Lovelace"\n',
        encoding="utf-8",
    )
    model_sha256 = str(
        SpacyEntityRecognizer(ner_model).metadata["artifact_sha256"]
    )

    result = runner.invoke(
        app,
        [
            "ner",
            str(input_path),
            "--text",
            "note",
            "--id",
            "id",
            "--model",
            str(ner_model),
            "--expected-model-version",
            "1.2.3",
            "--expected-model-sha256",
            model_sha256,
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert (
        payload["source_sha256"]
        == hashlib.sha256(input_path.read_bytes()).hexdigest()
    )
    assert payload["model"]["version"] == "1.2.3"
    assert payload["model"]["artifact_sha256"] == model_sha256
    assert payload["label_filter"] is None
    assert payload["summary"] == {
        "rows": 3,
        "processed_rows": 2,
        "null_rows": 1,
        "entities": 3,
        "labels": {"GPE": 1, "ORG": 1, "PERSON": 1},
    }
    assert [entity["record_id"] for entity in payload["entities"]] == [
        "r1",
        "r1",
        "r3",
    ]
    assert all(
        set(entity)
        == {
            "row_index",
            "record_id",
            "text",
            "label",
            "start_char",
            "end_char",
            "kb_id",
        }
        for entity in payload["entities"]
    )


def test_ner_cli_rejects_unpinned_model_content(
    tmp_path: Path, ner_model: Path
) -> None:
    input_path = tmp_path / "notes.csv"
    input_path.write_text("note\nOpenAI\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "ner",
            str(input_path),
            "--text",
            "note",
            "--model",
            str(ner_model),
            "--expected-model-sha256",
            "0" * 64,
        ],
    )

    assert result.exit_code == 2
    assert "artifact SHA-256" in result.output


@pytest.mark.parametrize(
    ("content", "option", "message"),
    [
        ("id,note\n1,42\n", "--text", "must be string"),
        ("id,note\n1,OpenAI\n1,Pune\n", "--id", "is not unique"),
    ],
)
def test_ner_cli_rejects_ambiguous_columns(
    tmp_path: Path,
    ner_model: Path,
    content: str,
    option: str,
    message: str,
) -> None:
    input_path = tmp_path / "notes.csv"
    input_path.write_text(content, encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "ner",
            str(input_path),
            "--text",
            "note",
            "--id",
            "id",
            "--model",
            str(ner_model),
        ],
    )

    assert result.exit_code == 2
    assert option in result.output or message in result.output
    assert message in result.output


def test_ner_cli_records_normalized_label_filter(
    tmp_path: Path, ner_model: Path
) -> None:
    input_path = tmp_path / "notes.csv"
    output_path = tmp_path / "entities.json"
    input_path.write_text(
        'id,note\nr1,"OpenAI is in Pune"\n',
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "ner",
            str(input_path),
            "--text",
            "note",
            "--model",
            str(ner_model),
            "--label",
            " ORG ",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["label_filter"] == ["ORG"]
    assert [entity["label"] for entity in payload["entities"]] == ["ORG"]


def test_ner_cli_rejects_non_finite_ids_before_serialization(
    tmp_path: Path, ner_model: Path
) -> None:
    input_path = tmp_path / "notes.parquet"
    output_path = tmp_path / "entities.json"
    pl.DataFrame(
        {
            "id": [1.0, float("nan")],
            "note": ["OpenAI", "Pune"],
        }
    ).write_parquet(input_path)

    result = runner.invoke(
        app,
        [
            "ner",
            str(input_path),
            "--text",
            "note",
            "--id",
            "id",
            "--model",
            str(ner_model),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 2
    assert "non-finite values" in result.output
    assert not output_path.exists()
