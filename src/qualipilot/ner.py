"""Named-entity extraction through an explicitly installed spaCy pipeline."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
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
    """Load one spaCy pipeline and extract its ``Doc.ents`` spans."""

    def __init__(
        self,
        model: str | Path,
        *,
        labels: Iterable[str] | None = None,
        batch_size: int = 1_000,
        n_process: int = 1,
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

        self._nlp: Any = nlp
        self._model_reference = model_reference
        self._labels = selected_labels
        self._batch_size = batch_size
        self._n_process = n_process

    @property
    def metadata(self) -> dict[str, str | list[str]]:
        """Return pipeline provenance suitable for an audit record."""
        meta = self._nlp.meta
        return {
            "source": self._model_reference,
            "name": str(meta.get("name") or self._model_reference),
            "version": str(meta.get("version") or "unknown"),
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
