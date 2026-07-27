"""Record consolidation contract tests."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import polars as pl
import pytest
from pydantic import ValidationError

import qualipilot.linking.consolidate as consolidate_module
from qualipilot.linking import (
    ConsolidationConfig,
    ExactMatch,
    LinkageResult,
    LinkConfig,
    MergeRule,
    RecordLinker,
    StringNormalization,
    SurvivorSortKey,
    consolidate_records,
)


def test_survivor_ranking_preserves_schema_and_builds_lineage() -> None:
    frame = pl.DataFrame(
        {
            "id": [30, 40, 20, 10],
            "priority": [1, 0, 1, 1],
            "email": [
                "thirty@example.com",
                "solo@example.com",
                None,
                "ten@example.com",
            ],
            "phone": [None, "400", "200", "100"],
        },
        schema_overrides={"priority": pl.UInt8},
    )
    original = frame.clone()
    config = ConsolidationConfig(
        sort_keys=(SurvivorSortKey(column="priority", descending=True),),
        completeness_columns=("email", "phone"),
    )

    result = consolidate_records(
        frame,
        id_column="id",
        clusters={30: 7, 40: 9, 20: 7, 10: 7},
        config=config,
    )

    assert result.frame.to_dicts() == [
        {
            "id": 10,
            "priority": 1,
            "email": "ten@example.com",
            "phone": "100",
        },
        {
            "id": 40,
            "priority": 0,
            "email": "solo@example.com",
            "phone": "400",
        },
    ]
    assert result.frame.schema == frame.schema
    assert frame.equals(original)
    assert result.lineage == {30: 10, 40: 40, 20: 10, 10: 10}
    assert result.removed_count == 2
    assert result.summary() == {
        "input_count": 4,
        "output_count": 2,
        "removed_count": 2,
        "cluster_count": 2,
        "conflict_count": 2,
        "change_count": 0,
    }


def test_merge_rules_fill_replace_and_audit_without_field_values() -> None:
    frame = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "first": [None, " ", "secret-first", "secret-other"],
            "frequency": [
                "secret-old",
                "secret-popular",
                "secret-popular",
                "secret-popular",
            ],
            "latest": [
                "secret-old",
                "secret-middle",
                "secret-new",
                None,
            ],
            "observed": [
                date(2024, 1, 1),
                date(2024, 2, 1),
                date(2024, 3, 1),
                date(2024, 4, 1),
            ],
            "score": [float("nan"), 2.5, 2.5, 3.0],
            "untouched": ["keep", "different", "keep", "keep"],
        }
    )
    config = ConsolidationConfig(
        merge_rules={
            "first": MergeRule(strategy="first_non_null"),
            "frequency": MergeRule(strategy="most_frequent"),
            "latest": MergeRule(strategy="latest", order_by="observed"),
            "score": MergeRule(strategy="first_non_null"),
        }
    )

    result = consolidate_records(
        frame,
        id_column="id",
        clusters={1: 0, 2: 0, 3: 0, 4: 0},
        config=config,
    )

    row = result.frame.row(0, named=True)
    assert row["id"] == 1
    assert row["first"] == "secret-first"
    assert row["frequency"] == "secret-popular"
    assert row["latest"] == "secret-new"
    assert row["score"] == 2.5
    assert row["untouched"] == "keep"
    assert result.frame.schema == frame.schema

    entries = {entry.column: entry for entry in result.audit}
    assert entries["first"].action == "filled"
    assert entries["first"].donor_id == 3
    assert entries["frequency"].action == "replaced"
    assert entries["frequency"].donor_id == 2
    assert entries["latest"].action == "replaced"
    assert entries["latest"].donor_id == 3
    assert entries["score"].action == "filled"
    assert entries["untouched"].action == "retained"
    assert entries["untouched"].distinct_value_count == 2
    assert entries["untouched"].conflicting_source_ids == (1, 2, 3, 4)
    audit_text = repr(result.audit)
    assert "secret-first" not in audit_text
    assert "secret-popular" not in audit_text
    assert "secret-new" not in audit_text


def test_record_linker_deduplicates_normalized_records_in_one_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pl.DataFrame(
        {
            "id": [1, 2],
            "phone": ["+1 (555) 123-4567", "+15551234567"],
            "email": [" N/A ", "ALICE@EXAMPLE.COM"],
        }
    )
    original = frame.clone()
    link_config = LinkConfig(
        unique_id_column="id",
        comparisons=[ExactMatch(column="phone")],
        normalization={
            "phone": StringNormalization(
                regex_replacements=((r"[^\d+]", ""),)
            ),
            "email": StringNormalization(null_tokens=("n/a",)),
        },
    )

    monkeypatch.setattr(
        RecordLinker,
        "run",
        lambda _self: LinkageResult(
            pairs=pl.DataFrame(
                {
                    "id_l": [1],
                    "id_r": [2],
                    "match_probability": [0.99],
                }
            ),
            clusters={1: 0, 2: 0},
            parameters={"threshold": 0.9},
        ),
    )

    result = RecordLinker(frame, link_config).deduplicate(
        ConsolidationConfig(
            completeness_columns=("email",),
            merge_rules={"email": MergeRule(strategy="first_non_null")},
        )
    )

    assert result.consolidation.frame.to_dicts() == [
        {
            "id": 2,
            "phone": "+15551234567",
            "email": "alice@example.com",
        }
    ]
    assert result.consolidation.lineage == {1: 2, 2: 2}
    assert result.linkage.clusters == {1: 0, 2: 0}
    assert frame.equals(original)


def test_deduplicate_validates_consolidation_before_linking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pl.DataFrame({"id": [1], "name": ["Alice"]})

    def fail_if_linked(_self: RecordLinker) -> LinkageResult:
        raise AssertionError("invalid consolidation must fail first")

    monkeypatch.setattr(RecordLinker, "run", fail_if_linked)

    with pytest.raises(ValueError, match="missing configured columns"):
        RecordLinker(
            frame,
            LinkConfig(
                unique_id_column="id",
                comparisons=[ExactMatch(column="name")],
            ),
        ).deduplicate(
            ConsolidationConfig(sort_keys=(SurvivorSortKey(column="missing"),))
        )


def test_singleton_clusters_skip_merge_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pl.DataFrame(
        {
            "id": list(range(5_000)),
            "name": [f"record-{index}" for index in range(5_000)],
        }
    )

    def fail_if_ranked(*args: object, **kwargs: object) -> None:
        raise AssertionError("singletons do not need survivor ranking")

    monkeypatch.setattr(
        consolidate_module,
        "_rank_cluster",
        fail_if_ranked,
    )

    result = consolidate_records(
        frame,
        id_column="id",
        clusters={index: index for index in range(5_000)},
        config=ConsolidationConfig(
            completeness_columns=("name",),
            merge_rules={"name": MergeRule(strategy="most_frequent")},
        ),
    )

    assert result.frame.equals(frame)
    assert result.audit == ()
    assert result.removed_count == 0


def test_singleton_latest_rule_still_requires_order_value() -> None:
    frame = pl.DataFrame(
        {"id": [1], "value": ["present"], "observed": [None]},
        schema_overrides={"observed": pl.Int64},
    )

    with pytest.raises(ValueError, match="requires non-missing"):
        consolidate_records(
            frame,
            id_column="id",
            clusters={1: 0},
            config=ConsolidationConfig(
                merge_rules={
                    "value": MergeRule(
                        strategy="latest",
                        order_by="observed",
                    )
                }
            ),
        )


def test_donor_ties_use_survivor_order_then_unique_id() -> None:
    frame = pl.DataFrame(
        {
            "id": [3, 2, 1],
            "frequency": ["right", "left", "left"],
            "latest": ["right-latest", "left-latest", "winner"],
            "observed": [2, 2, 2],
        }
    )
    config = ConsolidationConfig(
        merge_rules={
            "frequency": MergeRule(strategy="most_frequent"),
            "latest": MergeRule(strategy="latest", order_by="observed"),
        }
    )

    result = consolidate_records(
        frame,
        id_column="id",
        clusters={1: 0, 2: 0, 3: 0},
        config=config,
    )

    assert result.frame.row(0, named=True) == {
        "id": 1,
        "frequency": "left",
        "latest": "winner",
        "observed": 2,
    }


def test_missing_sort_values_rank_after_present_values() -> None:
    frame = pl.DataFrame(
        {
            "id": [1, 2],
            "rank": [float("nan"), 1.0],
            "name": ["missing-rank", "present-rank"],
        }
    )
    config = ConsolidationConfig(
        sort_keys=(SurvivorSortKey(column="rank", descending=True),)
    )

    result = consolidate_records(
        frame,
        id_column="id",
        clusters={1: 0, 2: 0},
        config=config,
    )

    assert result.frame.get_column("id").item() == 2


def test_all_missing_values_remain_in_the_survivor_dtype() -> None:
    frame = pl.DataFrame(
        {
            "id": [1, 2],
            "text": [" ", ""],
            "number": [float("nan"), None],
        },
        schema_overrides={"number": pl.Float32},
    )
    config = ConsolidationConfig(
        merge_rules={
            "text": MergeRule(strategy="first_non_null"),
            "number": MergeRule(strategy="most_frequent"),
        }
    )

    result = consolidate_records(
        frame,
        id_column="id",
        clusters={1: 0, 2: 0},
        config=config,
    )

    assert result.frame.get_column("text").item() == " "
    assert result.frame.get_column("number").is_nan().item()
    assert result.frame.schema == frame.schema
    assert result.audit == ()


def test_merge_preserves_categorical_dtype() -> None:
    frame = pl.DataFrame(
        {
            "id": [1, 2],
            "category": pl.Series([None, "gold"], dtype=pl.Categorical),
        }
    )

    result = consolidate_records(
        frame,
        id_column="id",
        clusters={1: 0, 2: 0},
        config=ConsolidationConfig(
            merge_rules={"category": MergeRule(strategy="first_non_null")}
        ),
    )

    assert result.frame.get_column("category").item() == "gold"
    assert result.frame.schema == frame.schema


def test_most_frequent_scales_across_hashable_decimal_values() -> None:
    size = 5_000
    amounts = [Decimal(value) for value in range(size)]
    amounts[-1] = Decimal("1234.00")
    scores = [float("nan")] * size
    scores[-2:] = [3.0, 3.0]
    frame = pl.DataFrame(
        {
            "id": range(size),
            "amount": amounts,
            "score": scores,
        },
        schema_overrides={"amount": pl.Decimal(10, 2)},
    )

    result = consolidate_records(
        frame,
        id_column="id",
        clusters=dict.fromkeys(range(size), 0),
        config=ConsolidationConfig(
            merge_rules={
                "amount": MergeRule(strategy="most_frequent"),
                "score": MergeRule(strategy="most_frequent"),
            }
        ),
    )

    assert result.frame.get_column("amount").item() == Decimal("1234.00")
    assert result.frame.get_column("score").item() == 3.0
    assert result.frame.schema == frame.schema


def test_latest_requires_an_order_for_every_value_candidate() -> None:
    frame = pl.DataFrame(
        {
            "id": [1, 2],
            "value": ["old", "new"],
            "observed": [1, None],
        }
    )
    config = ConsolidationConfig(
        merge_rules={
            "value": MergeRule(strategy="latest", order_by="observed")
        }
    )

    with pytest.raises(ValueError, match=r"non-missing.*source IDs: \[2\]"):
        consolidate_records(
            frame,
            id_column="id",
            clusters={1: 0, 2: 0},
            config=config,
        )


@pytest.mark.parametrize(
    ("clusters", "message"),
    [
        ({}, "link-mode results"),
        ({1: 0}, "map every input ID exactly once"),
        ({1: 0, 2: 0, 3: 0}, "map every input ID exactly once"),
        ({1: 0, 2: -1}, "non-negative integers"),
        ({1: False, 2: 0}, "non-negative integers"),
    ],
)
def test_cluster_map_must_be_a_complete_dedupe_partition(
    clusters: dict[int, int],
    message: str,
) -> None:
    frame = pl.DataFrame({"id": [1, 2], "value": ["a", "b"]})

    with pytest.raises(ValueError, match=message):
        consolidate_records(
            frame,
            id_column="id",
            clusters=clusters,
            config=ConsolidationConfig(),
        )


def test_configuration_and_scalar_contract_are_strict() -> None:
    with pytest.raises(ValidationError, match="latest merge rules"):
        MergeRule(strategy="latest")
    with pytest.raises(ValidationError, match="Extra inputs"):
        ConsolidationConfig.model_validate({"unknown": True})

    frame = pl.DataFrame(
        {
            "id": [1, 2],
            "value": ["a", "b"],
            "nested": [[1], [2]],
        }
    )
    with pytest.raises(TypeError, match="nested"):
        consolidate_records(
            frame,
            id_column="id",
            clusters={1: 0, 2: 0},
            config=ConsolidationConfig(),
        )

    scalar_frame = frame.drop("nested")
    protected = ConsolidationConfig(
        merge_rules={"id": MergeRule(strategy="most_frequent")}
    )
    with pytest.raises(ValueError, match="ID column is protected"):
        consolidate_records(
            scalar_frame,
            id_column="id",
            clusters={1: 0, 2: 0},
            config=protected,
        )


def test_empty_input_preserves_columns_and_dtypes() -> None:
    frame = pl.DataFrame(
        schema={"id": pl.Int64, "name": pl.String, "score": pl.Float32}
    )

    result = consolidate_records(
        frame,
        id_column="id",
        clusters={},
        config=ConsolidationConfig(),
    )

    assert result.frame.schema == frame.schema
    assert result.lineage == {}
    assert result.audit == ()
    assert result.summary()["removed_count"] == 0
