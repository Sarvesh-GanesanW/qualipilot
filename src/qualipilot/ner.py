"""Named-entity extraction through an explicitly installed spaCy pipeline."""

from __future__ import annotations

import hashlib
import hmac
import string
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class NamedEntity:
    """One labeled entity span using half-open character offsets."""

    text: str
    label: str
    start_char: int
    end_char: int
    kb_id: str | None = None

    def to_dict(self) -> dict[str, str | int | None]:
        """Return a JSON-safe representation."""
        return asdict(self)


class SpacyEntityRecognizer:
    """Load one spaCy pipeline and extract its ``Doc.ents`` spans.

    ``expected_sha256`` pins the installed model's source/data tree, excluding
    Python bytecode caches. It is a drift check, not a code-signing control.
    ``expected_version`` verifies model metadata.
    """

    def __init__(
        self,
        model: str | Path,
        *,
        labels: Iterable[str] | None = None,
        batch_size: int = 1_000,
        n_process: int = 1,
        expected_version: str | None = None,
        expected_sha256: str | None = None,
    ) -> None:
        model_reference = str(model).strip()
        if not model_reference:
            raise ValueError("model must be a spaCy package name or path")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if n_process < 1:
            raise ValueError("n_process must be at least 1")

        selected_labels: frozenset[str] | None = None
        if labels is not None:
            selected_labels = frozenset(label.strip() for label in labels)
            if not selected_labels or "" in selected_labels:
                raise ValueError("labels must contain non-empty values")

        try:
            import spacy
        except ModuleNotFoundError as exc:  # pragma: no cover - env specific
            if exc.name != "spacy":
                raise
            raise RuntimeError(
                "NER requires the optional dependency; "
                "install 'qualipilot[ner]'"
            ) from exc

        artifact_sha256 = _verified_artifact_sha256(
            model_reference, expected_sha256
        )
        try:
            nlp = spacy.load(model_reference)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"unable to load spaCy pipeline {model_reference!r}: {exc}"
            ) from exc

        entity_components = {"ner", "entity_ruler"}
        if entity_components.isdisjoint(nlp.pipe_names):
            raise ValueError(
                f"spaCy pipeline {model_reference!r} has no enabled "
                "'ner' or 'entity_ruler' component"
            )
        _validate_label_filter(nlp, selected_labels, entity_components)

        model_version = _validated_model_version(nlp, expected_version)

        self._nlp: Any = nlp
        self._spacy_version = version("spacy")
        self._model_reference = model_reference
        self._model_version = model_version
        self._artifact_sha256 = artifact_sha256
        self._labels = selected_labels
        self._batch_size = batch_size
        self._n_process = n_process

    @property
    def metadata(self) -> dict[str, str | list[str]]:
        """Return pipeline provenance suitable for an audit record."""
        meta = self._nlp.meta
        return {
            "source": self._model_reference,
            "spacy_version": self._spacy_version,
            "name": str(meta.get("name") or self._model_reference),
            "version": self._model_version,
            "artifact_sha256": self._artifact_sha256,
            "artifact_sha256_scope": (
                "model-tree files excluding __pycache__ and .pyc"
            ),
            "license": str(meta.get("license") or "unknown"),
            "language": str(meta.get("lang") or self._nlp.lang),
            "pipeline": list(self._nlp.pipe_names),
        }

    @property
    def label_filter(self) -> list[str] | None:
        """Return the normalized label filter used for extraction."""
        return sorted(self._labels) if self._labels is not None else None

    def extract(self, text: str) -> tuple[NamedEntity, ...]:
        """Extract entities from one document."""
        return self.extract_many([text])[0]

    def extract_many(
        self, texts: Iterable[str]
    ) -> list[tuple[NamedEntity, ...]]:
        """Extract entities in input order using spaCy's batched pipeline."""
        documents = self._nlp.pipe(
            self._validated_texts(texts),
            batch_size=self._batch_size,
            n_process=self._n_process,
        )
        return [
            tuple(
                NamedEntity(
                    text=entity.text,
                    label=entity.label_,
                    start_char=entity.start_char,
                    end_char=entity.end_char,
                    kb_id=entity.kb_id_ or None,
                )
                for entity in document.ents
                if self._labels is None or entity.label_ in self._labels
            )
            for document in documents
        ]

    def _validated_texts(self, texts: Iterable[str]) -> Iterable[str]:
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(
                    f"text at position {index} must be str, "
                    f"got {type(text).__name__}"
                )
            if len(text) > self._nlp.max_length:
                raise ValueError(
                    f"text at position {index} has {len(text)} characters; "
                    f"pipeline maximum is {self._nlp.max_length}"
                )
            yield text


def _validate_label_filter(
    nlp: Any,
    selected: frozenset[str] | None,
    entity_components: set[str],
) -> None:
    if selected is None:
        return
    available = {
        label
        for component in entity_components
        for label in nlp.pipe_labels.get(component, [])
    }
    unknown = sorted(selected - available)
    if unknown:
        raise ValueError(
            f"labels are not provided by the spaCy pipeline: {unknown}"
        )


def _directory_sha256(root: Path) -> str:
    if not root.is_dir():
        raise ValueError(f"spaCy pipeline path is not a directory: {root}")
    digest = hashlib.sha256()
    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not files:
        raise ValueError(f"spaCy pipeline contains no files: {root}")
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        size = path.stat().st_size
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _validated_model_version(nlp: Any, expected_version: str | None) -> str:
    model_version = str(nlp.meta.get("version") or "unknown")
    if expected_version is not None and model_version != expected_version:
        raise ValueError(
            f"spaCy pipeline version {model_version!r} does not match "
            f"expected version {expected_version!r}"
        )
    return model_version


def _verified_artifact_sha256(
    model_reference: str, expected_sha256: str | None
) -> str:
    artifact_sha256 = _directory_sha256(_model_artifact_root(model_reference))
    if expected_sha256 is None:
        return artifact_sha256
    normalized_sha256 = _normalized_sha256(expected_sha256)
    if not hmac.compare_digest(artifact_sha256, normalized_sha256):
        raise ValueError(
            "spaCy pipeline artifact SHA-256 "
            f"{artifact_sha256} does not match expected "
            f"{normalized_sha256}"
        )
    return artifact_sha256


def _model_artifact_root(model_reference: str) -> Path:
    candidate = Path(model_reference).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    try:
        specification = find_spec(model_reference)
    except (ImportError, ModuleNotFoundError):
        specification = None
    locations = (
        tuple(specification.submodule_search_locations or ())
        if specification is not None
        else ()
    )
    if len(locations) != 1:
        raise ValueError(
            f"unable to locate spaCy pipeline artifact {model_reference!r}"
        )
    return Path(locations[0]).resolve()


def _normalized_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in string.hexdigits for character in normalized
    ):
        raise ValueError("expected_sha256 must be a 64-character hex digest")
    return normalized
