"""Tests for non-destructive linkage string normalization."""

from __future__ import annotations

import polars as pl
import pytest
from pydantic import ValidationError

from qualipilot.linking import (
    ConsolidationConfig,
    ExactMatch,
    LinkConfig,
    RecordLinker,
    StringNormalization,
)


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
def test_normalization_applies_before_blocking_and_comparison(
    backend: str,
) -> None:
    if backend == "duckdb":
        pytest.importorskip("duckdb")
    frame = pl.DataFrame(
        {
            "id": [101, 102, 103],
            "bucket": [
                "  \uff27\uff32\uff2f\uff35\uff30\tONE ",
                "group  one",
                "other",
            ],
            "name": [" ALICE\u00a0 SMITH ", "alice smith", "someone"],
        }
    )
    original = frame.clone()
    config = LinkConfig(
        backend=backend,  # type: ignore[arg-type]
        unique_id_column="id",
        comparisons=[ExactMatch(column="name")],
        blocking_rules=[["bucket"]],
        normalization={
            "bucket": StringNormalization(),
            "name": StringNormalization(),
        },
    )

    pairs = RecordLinker(frame, config).run().pairs

    assert pairs.select(["__id_l__", "__id_r__"]).rows() == [(101, 102)]
    assert pairs["level__name"].to_list() == [2]
    assert frame.equals(original)


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
def test_normalized_null_tokens_do_not_block_together(backend: str) -> None:
    if backend == "duckdb":
        pytest.importorskip("duckdb")
    frame = pl.DataFrame(
        {
            "id": [1, 2],
            "bucket": [" MISSING ", "missing"],
            "name": ["same", "same"],
        }
    )
    config = LinkConfig(
        backend=backend,  # type: ignore[arg-type]
        unique_id_column="id",
        comparisons=[ExactMatch(column="name")],
        blocking_rules=[["bucket"]],
        normalization={
            "bucket": StringNormalization(null_tokens=("missing",))
        },
    )

    assert RecordLinker(frame, config).run().pairs.is_empty()


def test_blank_strings_are_missing_after_default_normalization() -> None:
    frame = pl.DataFrame(
        {
            "id": [1, 2],
            "bucket": [" ", "\t"],
            "name": ["same", "same"],
        }
    )
    config = LinkConfig(
        unique_id_column="id",
        comparisons=[ExactMatch(column="name")],
        blocking_rules=[["bucket"]],
        normalization={"bucket": StringNormalization()},
    )

    assert RecordLinker(frame, config).run().pairs.is_empty()


def test_normalization_config_round_trips_as_json() -> None:
    config = LinkConfig(
        unique_id_column="id",
        comparisons=[ExactMatch(column="name")],
        normalization={
            "name": StringNormalization(
                unicode_form="NFC",
                trim=False,
                collapse_whitespace=False,
                lowercase=False,
                null_tokens=("unknown",),
                regex_replacements=((r"[^\w]+", " "),),
            )
        },
    )

    payload = config.model_dump(mode="json")

    assert payload["normalization"]["name"] == {
        "unicode_form": "NFC",
        "trim": False,
        "collapse_whitespace": False,
        "lowercase": False,
        "null_tokens": ["unknown"],
        "regex_replacements": [[r"[^\w]+", " "]],
    }
    assert LinkConfig.model_validate(payload) == config
    with pytest.raises(ValidationError, match="Extra inputs"):
        StringNormalization(unknown_option=True)  # type: ignore[call-arg]


def test_normalization_rejects_identifiers_and_accepts_output_columns() -> (
    None
):
    with pytest.raises(ValueError, match="identifiers must remain unchanged"):
        LinkConfig(
            unique_id_column="id",
            comparisons=[ExactMatch(column="name")],
            normalization={"id": StringNormalization()},
        )

    config = LinkConfig(
        unique_id_column="id",
        comparisons=[ExactMatch(column="name")],
        normalization={"notes": StringNormalization()},
    )
    assert set(config.normalization) == {"notes"}


def test_regex_replacements_standardize_formatted_values() -> None:
    frame = pl.DataFrame(
        {
            "id": [1, 2],
            "phone": ["+1 (555) 123-4567", "+15551234567"],
        }
    )
    config = LinkConfig(
        unique_id_column="id",
        comparisons=[ExactMatch(column="phone")],
        normalization={
            "phone": StringNormalization(regex_replacements=((r"[^\d+]", ""),))
        },
    )

    result = RecordLinker(frame, config).run()

    assert result.pairs["level__phone"].to_list() == [2]
    assert frame["phone"].to_list() == [
        "+1 (555) 123-4567",
        "+15551234567",
    ]

    with pytest.raises(ValidationError, match="must not be empty"):
        StringNormalization(regex_replacements=(("", ""),))

    invalid = config.model_copy(
        update={
            "normalization": {
                "phone": StringNormalization(regex_replacements=(("(", ""),))
            }
        }
    )
    with pytest.raises(ValueError, match="invalid string normalization"):
        RecordLinker(frame, invalid).run()


def test_normalization_rejects_unknown_and_non_string_columns() -> None:
    config = LinkConfig(
        unique_id_column="id",
        comparisons=[ExactMatch(column="name")],
        normalization={"name": StringNormalization()},
    )

    with pytest.raises(ValueError, match=r"missing normalization.*'name'"):
        RecordLinker(
            pl.DataFrame({"id": [1, 2], "other": ["a", "a"]}),
            config,
        ).run()

    with pytest.raises(ValueError, match=r"must be string-like.*'name'"):
        RecordLinker(
            pl.DataFrame({"id": [1, 2], "name": [10, 10]}),
            config,
        ).run()


def test_deduplication_preserves_the_normalized_string_schema() -> None:
    frame = pl.DataFrame(
        {
            "id": [1],
            "name": ["Only record"],
            "segment": pl.Series([" Gold "], dtype=pl.Categorical),
        }
    )
    config = LinkConfig(
        unique_id_column="id",
        comparisons=[ExactMatch(column="name")],
        normalization={"segment": StringNormalization()},
    )

    result = RecordLinker(frame, config).deduplicate(ConsolidationConfig())

    assert frame.schema["segment"] == pl.Categorical
    assert result.consolidation.frame.schema["segment"] == pl.String
    assert result.consolidation.frame["segment"].to_list() == ["gold"]


def test_link_mode_validates_normalization_on_both_frames() -> None:
    config = LinkConfig(
        mode="link",
        unique_id_column="id",
        comparisons=[ExactMatch(column="name")],
        normalization={"name": StringNormalization()},
    )
    left = pl.DataFrame({"id": [1], "name": ["Alice"]})
    right = pl.DataFrame({"id": [2], "name": [10]})

    with pytest.raises(
        ValueError,
        match=r"right normalization columns must be string-like.*'name'",
    ):
        RecordLinker(left, config, right).run()
