"""Tests for in-house record linkage."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

from qualipilot.linking import (
    ConsolidationConfig,
    ExactMatch,
    FuzzyString,
    LinkConfig,
    NumericDiff,
    RecordLinker,
)
from qualipilot.linking.blocking import (
    build_candidate_pairs,
    estimate_candidate_pair_upper_bound,
)
from qualipilot.linking.cluster import cluster_from_pairs
from qualipilot.linking.em import (
    build_fit_diagnostics,
    estimate_parameters,
    score_pairs,
)
from qualipilot.linking.linker import LinkageResult


def _unique_names(n: int) -> list[str]:
    rng = np.random.default_rng(0)
    pool = list("abcdefghijklmnopqrstuvwxyz")
    return ["".join(rng.choice(pool, size=12)).strip() for _ in range(n)]


def test_nan_blocking_values_do_not_form_candidates() -> None:
    frame = pl.DataFrame({"id": [1, 2, 3], "block": [float("nan")] * 3})

    assert (
        estimate_candidate_pair_upper_bound(
            frame,
            blocking_rules=[["block"]],
            mode="dedupe",
        )
        == 0
    )
    assert build_candidate_pairs(
        frame,
        id_column="id",
        blocking_rules=[["block"]],
        mode="dedupe",
    ).is_empty()


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
def test_nan_blocking_values_match_across_backends(backend: str) -> None:
    if backend == "duckdb":
        pytest.importorskip("duckdb")
    frame = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "block": [float("nan")] * 3,
            "name": ["same"] * 3,
        }
    )
    config = LinkConfig(
        backend=backend,  # type: ignore[arg-type]
        unique_id_column="id",
        comparisons=[ExactMatch(column="name")],
        blocking_rules=[["block"]],
    )

    assert RecordLinker(frame, config).run().pairs.is_empty()


@pytest.fixture
def synthetic_case() -> tuple[pl.DataFrame, list[tuple[int, int]]]:
    """1000 unique people + 50 near-duplicates with single-char typos."""
    rng = np.random.default_rng(42)
    n = 1000
    names = _unique_names(n)
    df = pl.DataFrame(
        {
            "id": list(range(n)),
            "name": names,
            "postcode": [f"PC{i % 50:03d}" for i in range(n)],
            "dob": rng.integers(1950, 2005, n),
        }
    )
    dupe_ids = rng.choice(n, 50, replace=False).tolist()
    dupe_names = [names[int(i)][:-1] + "q" for i in dupe_ids]
    dupes = pl.DataFrame(
        {
            "id": list(range(n, n + 50)),
            "name": dupe_names,
            "postcode": [df["postcode"][int(i)] for i in dupe_ids],
            "dob": [df["dob"][int(i)] for i in dupe_ids],
        }
    )
    frame = df.vstack(dupes)
    injected_pairs = [
        (n + offset, int(original_id))
        for offset, original_id in enumerate(dupe_ids)
    ]
    return frame, injected_pairs


@pytest.fixture
def synthetic_frame(
    synthetic_case: tuple[pl.DataFrame, list[tuple[int, int]]],
) -> pl.DataFrame:
    return synthetic_case[0]


def test_linker_recovers_injected_duplicates(
    synthetic_case: tuple[pl.DataFrame, list[tuple[int, int]]],
) -> None:
    synthetic_frame, injected_pairs = synthetic_case
    cfg = LinkConfig(
        unique_id_column="id",
        comparisons=[
            FuzzyString(column="name", thresholds=(0.92, 0.75)),
            ExactMatch(column="postcode"),
            NumericDiff(column="dob", thresholds=(0.0, 1.0)),
        ],
        blocking_rules=[["postcode"]],
        match_threshold_probability=0.9,
    )
    linker = RecordLinker(synthetic_frame, cfg)
    result = linker.run()

    summary = result.summary()
    assert summary["candidate_pairs"] > 0
    # at the right threshold, injected dupes should cluster with their
    # originals — we accept >=40 / 50 to tolerate small EM variance
    recalled = sum(
        1
        for duplicate_id, original_id in injected_pairs
        if result.clusters.get(duplicate_id)
        == result.clusters.get(original_id)
    )
    assert recalled >= 40
    expected = {
        tuple(sorted((duplicate_id, original_id)))
        for duplicate_id, original_id in injected_pairs
    }
    predicted = {
        tuple(sorted((left, right)))
        for left, right in result.match_pairs(0.9)
        .select(["__id_l__", "__id_r__"])
        .iter_rows()
    }
    true_positives = len(expected & predicted)
    assert true_positives >= 40
    assert true_positives / len(predicted) >= 0.8


def test_empty_pairs_after_blocking_does_not_crash() -> None:
    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["a", "b", "c"],
            "bucket": ["x", "y", "z"],
        }
    )
    cfg = LinkConfig(
        unique_id_column="id",
        comparisons=[ExactMatch(column="name")],
        blocking_rules=[["bucket"]],  # no two records share a bucket
    )
    result = RecordLinker(df, cfg).run()
    assert result.pairs.height == 0
    assert result.summary()["candidate_pairs"] == 0
    assert result.summary()["clusters"] == 3
    assert set(result.clusters) == {1, 2, 3}
    assert result.pairs.schema["match_probability"] == pl.Float64


def test_medium_frame_stays_within_blocking_bound() -> None:
    n = 5_000
    names = _unique_names(n)
    df = pl.DataFrame(
        {
            "id": list(range(n)),
            "name": names,
            "postcode": [f"PC{i % 100:03d}" for i in range(n)],
        }
    )
    cfg = LinkConfig(
        unique_id_column="id",
        comparisons=[
            FuzzyString(column="name"),
            ExactMatch(column="postcode"),
        ],
        blocking_rules=[["postcode"]],
    )
    result = RecordLinker(df, cfg).run()
    assert result.pairs.height <= 125_000


def test_config_rejects_empty_comparisons() -> None:
    with pytest.raises(ValueError, match="comparison is required"):
        LinkConfig(unique_id_column="id")


def test_blocking_rules_are_canonicalized() -> None:
    config = LinkConfig(
        unique_id_column="id",
        comparisons=[ExactMatch(column="name")],
        blocking_rules=[["z", "a"], ["b"]],
    )

    assert config.blocking_rules == [["a", "z"], ["b"]]


@pytest.mark.parametrize(
    "blocking_rules",
    [
        [["a", "b"], ["b", "a"]],
        [["a"], ["a", "b"]],
        [["a", "b"], ["a"]],
    ],
)
def test_redundant_blocking_rules_are_rejected(
    blocking_rules: list[list[str]],
) -> None:
    with pytest.raises(ValueError, match=r"unique|redundant supersets"):
        LinkConfig(
            unique_id_column="id",
            comparisons=[ExactMatch(column="name")],
            blocking_rules=blocking_rules,
        )


def test_numeric_comparison_levels() -> None:
    pairs = pl.DataFrame(
        {"age_l": [20.0, 30.0, None], "age_r": [21.0, 40.0, 30.0]}
    )
    comp = NumericDiff(column="age", thresholds=(0.5, 5.0))
    levels = comp.assign_levels(pairs)
    # diff 1.0 -> falls into <=5.0 bucket (level 2)
    # diff 10.0 -> "far" (level 1)
    # null -> level 0
    assert levels[0] == 2
    assert levels[1] == 1
    assert levels[2] == 0


def test_exact_comparison_treats_null_as_no_signal() -> None:
    pairs = pl.DataFrame(
        {
            "name_l": ["same", None, "left"],
            "name_r": ["same", "right", None],
        }
    )
    assert ExactMatch(column="name").assign_levels(pairs).tolist() == [2, 0, 0]


@pytest.mark.parametrize(
    ("column", "values"),
    [
        ("missing", [None, None, None, None]),
        ("constant", ["same", "same", "same", "same"]),
    ],
)
def test_non_informative_comparisons_do_not_change_scores(
    column: str,
    values: list[str | None],
) -> None:
    frame = pl.DataFrame(
        {
            "id": range(4),
            "name": ["a", "a", "b", "c"],
            column: values,
        }
    )
    base = RecordLinker(
        frame,
        LinkConfig(
            unique_id_column="id",
            comparisons=[ExactMatch(column="name")],
        ),
    ).run()
    extended = RecordLinker(
        frame,
        LinkConfig(
            unique_id_column="id",
            comparisons=[
                ExactMatch(column="name"),
                ExactMatch(column=column),
            ],
        ),
    ).run()

    np.testing.assert_allclose(
        extended.pairs["match_probability"],
        base.pairs["match_probability"],
    )
    assert extended.parameters["lambda"] == pytest.approx(
        base.parameters["lambda"]
    )
    np.testing.assert_array_equal(
        extended.parameters["m"][1],
        extended.parameters["u"][1],
    )


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
def test_inverted_em_fit_fails_closed(backend: str) -> None:
    if backend == "duckdb":
        pytest.importorskip("duckdb")
    frame = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "block": [1, 1, 2, 2, 3, 3, 4, 4],
            "name": ["a", "a", "b", "b", "c", "c", "d", "X"],
            "email": ["a", "a", "b", "b", "c", "c", "d", "Y"],
        }
    )
    config = LinkConfig(
        backend=backend,  # type: ignore[arg-type]
        unique_id_column="id",
        blocking_rules=[["block"]],
        comparisons=[
            ExactMatch(column="name"),
            ExactMatch(column="email"),
        ],
    )
    linker = RecordLinker(frame, config)

    result = linker.run()

    fit = result.parameters["fit"]
    assert fit["status"] == "rejected"
    assert fit["inverted_comparisons"] == ["name", "email"]
    assert result.pairs["match_probability"].to_list() == [0.0] * 4
    assert result.match_pairs(0.9).is_empty()
    assert len(set(result.clusters.values())) == frame.height
    with pytest.raises(
        ValueError,
        match=r"rejected linkage fit.*agreement ordering inverted",
    ):
        linker.deduplicate(ConsolidationConfig())


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
def test_degenerate_em_fit_fails_closed(backend: str) -> None:
    if backend == "duckdb":
        pytest.importorskip("duckdb")
    frame = pl.DataFrame(
        {
            "id": [1, 2],
            "name": ["same", "same"],
            "email": ["same@example.com", "same@example.com"],
        }
    )
    config = LinkConfig(
        backend=backend,  # type: ignore[arg-type]
        unique_id_column="id",
        comparisons=[
            ExactMatch(column="name"),
            ExactMatch(column="email"),
        ],
    )

    result = RecordLinker(frame, config).run()

    fit = result.parameters["fit"]
    assert fit["status"] == "rejected"
    assert fit["usable_comparisons"] == []
    assert "no comparison" in fit["reason"]
    assert result.pairs["match_probability"].to_list() == [0.0]
    assert len(set(result.clusters.values())) == frame.height


def test_em_smooths_learned_and_scored_probabilities() -> None:
    levels = np.array(
        [
            [1, 1],
            [1, 2],
            [2, 1],
            [2, 2],
        ],
        dtype=np.uint8,
    )
    params = estimate_parameters(
        levels,
        np.array([3, 3], dtype=np.uint8),
        prior=0.1,
    )

    assert 0.0 < params["lambda"] < 1.0
    for table in (params["m"], params["u"]):
        assert np.all(table[:, 1:] > 0.0)
        assert np.all(table[:, 1:] < 1.0)
    extreme_scores = score_pairs(
        np.full((1, 20), 2, dtype=np.uint8),
        np.tile(np.array([[0.0, 1e-12, 1.0]]), (20, 1)),
        np.tile(np.array([[0.0, 1.0, 1e-12]]), (20, 1)),
        0.5,
    )
    assert 0.0 < extreme_scores[0] < 1.0


@pytest.mark.parametrize(
    "joint_counts",
    # Joint comparison-level counts from the deterministic linkage benchmark.
    [
        (
            ((1, 2, 1), 3_864),
            ((1, 2, 2), 147),
            ((1, 2, 3), 72),
            ((2, 2, 3), 3),
            ((3, 2, 3), 47),
        ),
        (
            ((1, 2, 1), 138_947),
            ((1, 2, 2), 5_263),
            ((1, 2, 3), 2_684),
            ((2, 2, 1), 1),
            ((2, 2, 3), 14),
            ((3, 2, 3), 236),
        ),
    ],
    ids=["5k-benchmark", "25k-benchmark"],
)
def test_default_em_budget_converges_on_benchmark_distributions(
    joint_counts: tuple[tuple[tuple[int, int, int], int], ...],
) -> None:
    patterns = np.array(
        [pattern for pattern, _ in joint_counts], dtype=np.uint8
    )
    levels = np.repeat(
        patterns,
        [count for _, count in joint_counts],
        axis=0,
    )
    config = LinkConfig(
        unique_id_column="id",
        comparisons=[
            FuzzyString(column="name", thresholds=(0.92, 0.75)),
            ExactMatch(column="postcode"),
            NumericDiff(column="dob", thresholds=(0.0, 1.0)),
        ],
    )
    n_levels = np.array(
        [comparison.levels for comparison in config.comparisons],
        dtype=np.uint8,
    )

    params = estimate_parameters(
        levels,
        n_levels,
        prior=config.prior_match_probability,
        max_iter=config.em_max_iter,
        tol=config.em_tolerance,
    )
    fit = build_fit_diagnostics(
        params,
        n_levels,
        [comparison.column for comparison in config.comparisons],
        sampled_pair_count=len(levels),
        candidate_pair_count=len(levels),
    )

    assert params["fit_state"]["converged"] is True
    assert 15 < params["fit_state"]["iterations"] <= config.em_max_iter
    assert params["fit_state"]["max_parameter_delta"] < config.em_tolerance
    assert fit["status"] == "ok"


@pytest.mark.parametrize(
    ("comparison", "message"),
    [
        ({"kind": "exact", "column": "name", "levels": 99}, "extra"),
        (
            {"kind": "numeric", "column": "age", "thresholds": [-1]},
            "finite and >= 0",
        ),
        (
            {"kind": "fuzzy", "column": "name", "thresholds": [1.1]},
            r"in \[0, 1\]",
        ),
    ],
)
def test_config_rejects_invalid_comparisons(
    comparison: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        LinkConfig(unique_id_column="id", comparisons=[comparison])  # type: ignore[list-item]


def test_linker_rejects_duplicate_ids() -> None:
    frame = pl.DataFrame({"id": [1, 1], "name": ["a", "a"]})
    config = LinkConfig(
        unique_id_column="id",
        comparisons=[ExactMatch(column="name")],
    )
    with pytest.raises(ValueError, match="unique IDs contain duplicates"):
        RecordLinker(frame, config).run()


@pytest.mark.parametrize("blank_id", ["", " \t "])
def test_linker_and_deduplicator_reject_blank_ids(blank_id: str) -> None:
    frame = pl.DataFrame({"id": [blank_id, "valid"], "name": ["a", "a"]})
    linker = RecordLinker(
        frame,
        LinkConfig(
            unique_id_column="id",
            comparisons=[ExactMatch(column="name")],
        ),
    )

    with pytest.raises(ValueError, match="unique IDs must not be missing"):
        linker.run()
    with pytest.raises(ValueError, match="unique IDs must not be missing"):
        linker.deduplicate(ConsolidationConfig())


def test_pair_cap_is_checked_before_cartesian_materialisation() -> None:
    frame = pl.DataFrame({"id": range(10), "name": ["a"] * 10})
    config = LinkConfig(
        unique_id_column="id",
        comparisons=[ExactMatch(column="name")],
        max_pairs_warning=20,
        max_pairs_hard_cap=20,
    )
    with pytest.raises(MemoryError, match="estimated upper bound 45 pairs"):
        RecordLinker(frame, config).run()


def test_unblocked_pair_budget_uses_estimate_before_materialisation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pl.DataFrame({"id": range(10), "name": ["a"] * 10})
    config = LinkConfig(
        unique_id_column="id",
        comparisons=[ExactMatch(column="name")],
        max_pairs_warning=20,
        max_pairs_hard_cap=100,
    )

    def fail_if_materialised(*args: object, **kwargs: object) -> pl.DataFrame:
        raise AssertionError("candidate pairs were materialised")

    monkeypatch.setattr(
        "qualipilot.linking.linker.build_candidate_pairs",
        fail_if_materialised,
    )

    with pytest.raises(
        MemoryError,
        match=r"unblocked linkage estimated upper bound 45 pairs",
    ):
        RecordLinker(frame, config).run()


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
def test_two_table_link_uses_the_right_frame(backend: str) -> None:
    if backend == "duckdb":
        pytest.importorskip("duckdb")
    left = pl.DataFrame({"id": [1], "name": ["left"], "bucket": ["x"]})
    right = pl.DataFrame({"id": [2], "name": ["right"], "bucket": ["x"]})
    config = LinkConfig(
        backend=backend,  # type: ignore[arg-type]
        mode="link",
        unique_id_column="id",
        comparisons=[ExactMatch(column="name")],
        blocking_rules=[["bucket"]],
    )
    result = RecordLinker(left, config, right).run()
    assert result.pairs.select(["__id_l__", "__id_r__"]).row(0) == (1, 2)


def test_duckdb_blocking_column_need_not_be_compared() -> None:
    pytest.importorskip("duckdb")
    frame = pl.DataFrame(
        {"id": [1, 2], "name": ["a", "a"], "bucket": ["x", "x"]}
    )
    config = LinkConfig(
        backend="duckdb",
        unique_id_column="id",
        comparisons=[ExactMatch(column="name")],
        blocking_rules=[["bucket"]],
    )
    assert RecordLinker(frame, config).run().pairs.height == 1


@pytest.mark.parametrize(
    ("values", "thresholds", "expected"),
    [
        ([0.0, 0.2], (0.5, 5.0), 3),
        ([2**53, 2**53 + 1], (0.0,), 1),
        ([0, 2**53 + 1], (float(2**53),), 1),
        ([-(2**63), 2**63 - 1], (1.0,), 1),
    ],
)
def test_numeric_levels_match_duckdb(
    values: list[float | int],
    thresholds: tuple[float, ...],
    expected: int,
) -> None:
    pytest.importorskip("duckdb")
    frame = pl.DataFrame({"id": [1, 2], "value": values})
    levels = []
    for backend in ("polars", "duckdb"):
        config = LinkConfig(
            backend=backend,  # type: ignore[arg-type]
            unique_id_column="id",
            comparisons=[NumericDiff(column="value", thresholds=thresholds)],
        )
        result = RecordLinker(frame, config).run()
        levels.append(result.pairs["level__value"][0])
    assert levels == [expected, expected]


@pytest.mark.parametrize(
    "threshold",
    [float("nan"), float("inf"), -0.1, 1.1],
)
def test_match_pairs_rejects_invalid_thresholds(threshold: float) -> None:
    result = LinkageResult(pairs=pl.DataFrame({"match_probability": [0.9]}))

    with pytest.raises(ValueError, match="between 0 and 1"):
        result.match_pairs(threshold)


def test_labeled_pair_metrics_include_pairs_omitted_by_blocking() -> None:
    result = LinkageResult(
        pairs=pl.DataFrame(
            {
                "__id_l__": [1, 1, 2],
                "__id_r__": [2, 3, 3],
                "match_probability": [0.9, 0.8, 0.2],
            }
        )
    )
    labels = pl.DataFrame(
        {
            "__id_l__": [1, 1, 2, 1],
            "__id_r__": [2, 3, 3, 4],
            "is_match": [True, False, False, True],
        }
    )

    assert result.evaluate_labeled_pairs(labels, threshold=0.5) == {
        "threshold": 0.5,
        "labeled_pairs": 4,
        "tp": 1,
        "fp": 1,
        "tn": 1,
        "fn": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
def test_labeled_pair_metrics_work_for_both_backends(backend: str) -> None:
    if backend == "duckdb":
        pytest.importorskip("duckdb")
    frame = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "block": ["candidate", "candidate", "left", "right"],
            "name": ["different-a", "different-b", "same", "same"],
        }
    )
    result = RecordLinker(
        frame,
        LinkConfig(
            backend=backend,  # type: ignore[arg-type]
            unique_id_column="id",
            comparisons=[ExactMatch(column="name")],
            blocking_rules=[["block"]],
        ),
    ).run()
    labels = pl.DataFrame(
        {
            "__id_l__": [1, 3],
            "__id_r__": [2, 4],
            "is_match": [False, True],
        }
    )

    metrics = result.evaluate_labeled_pairs(labels, threshold=0.5)

    assert metrics["tn"] == 1
    assert metrics["fn"] == 1
    assert metrics["recall"] == 0.0


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        (
            pl.DataFrame({"__id_l__": [1], "__id_r__": [2]}),
            "missing required columns",
        ),
        (
            pl.DataFrame(
                {
                    "__id_l__": [1],
                    "__id_r__": [2],
                    "is_match": [1],
                }
            ),
            "Boolean dtype",
        ),
        (
            pl.DataFrame(
                {
                    "__id_l__": [1],
                    "__id_r__": [2],
                    "is_match": pl.Series([None], dtype=pl.Boolean),
                }
            ),
            "must not contain nulls",
        ),
        (
            pl.DataFrame(
                {
                    "__id_l__": [1, 1],
                    "__id_r__": [2, 2],
                    "is_match": [True, False],
                }
            ),
            "must be unique",
        ),
    ],
)
def test_labeled_pair_metrics_validate_labels(
    labels: pl.DataFrame,
    message: str,
) -> None:
    result = LinkageResult(
        pairs=pl.DataFrame(
            {
                "__id_l__": [1],
                "__id_r__": [2],
                "match_probability": [0.9],
            }
        )
    )

    with pytest.raises(ValueError, match=message):
        result.evaluate_labeled_pairs(labels, threshold=0.5)


@pytest.mark.parametrize(
    "threshold",
    [float("nan"), float("inf"), -0.1, 1.1],
)
def test_labeled_pair_metrics_reject_invalid_thresholds(
    threshold: float,
) -> None:
    result = LinkageResult(pairs=pl.DataFrame({"match_probability": [0.9]}))

    with pytest.raises(ValueError, match="between 0 and 1"):
        result.evaluate_labeled_pairs(pl.DataFrame(), threshold=threshold)


def test_summary_reports_the_effective_match_threshold() -> None:
    result = LinkageResult(
        pairs=pl.DataFrame({"match_probability": [0.8, 0.7]}),
        parameters={"threshold": 0.75},
    )

    summary = result.summary()

    assert summary["match_threshold_probability"] == 0.75
    assert summary["matched_pairs"] == 1


def test_numeric_comparison_rejects_infinite_values() -> None:
    frame = pl.DataFrame({"id": [1, 2], "value": [float("inf")] * 2})
    config = LinkConfig(
        unique_id_column="id",
        comparisons=[NumericDiff(column="value")],
    )

    with pytest.raises(ValueError, match="infinite values"):
        RecordLinker(frame, config).run()


def test_fuzzy_comparison_rejects_non_string_values() -> None:
    left = pl.DataFrame({"id": [1], "value": [True]})
    right = pl.DataFrame({"id": [2], "value": ["True"]})
    config = LinkConfig(
        mode="link",
        unique_id_column="id",
        comparisons=[FuzzyString(column="value")],
    )

    with pytest.raises(ValueError, match="string-like"):
        RecordLinker(left, config, right).run()


def test_fuzzy_comparison_accepts_categorical_strings() -> None:
    frame = pl.DataFrame(
        {
            "id": [1, 2],
            "name": pl.Series(["same", "same"], dtype=pl.Categorical),
        }
    )
    config = LinkConfig(
        unique_id_column="id",
        comparisons=[FuzzyString(column="name")],
    )

    assert RecordLinker(frame, config).run().pairs["level__name"][0] == 3


def test_empty_fuzzy_strings_match_across_backends() -> None:
    pytest.importorskip("duckdb")
    frame = pl.DataFrame({"id": [1, 2], "name": ["", ""]})
    levels = []
    for backend in ("polars", "duckdb"):
        config = LinkConfig(
            backend=backend,  # type: ignore[arg-type]
            unique_id_column="id",
            comparisons=[FuzzyString(column="name")],
        )
        levels.append(
            RecordLinker(frame, config).run().pairs["level__name"][0]
        )

    assert levels == [3, 3]


def test_fuzzy_threshold_boundaries_match_across_backends() -> None:
    pytest.importorskip("duckdb")
    frame = pl.DataFrame({"id": [1, 2], "name": ["a", "abc"]})
    levels = []
    for backend in ("polars", "duckdb"):
        config = LinkConfig(
            backend=backend,  # type: ignore[arg-type]
            unique_id_column="id",
            comparisons=[FuzzyString(column="name", thresholds=(0.8,))],
        )
        levels.append(
            RecordLinker(frame, config).run().pairs["level__name"][0]
        )

    assert levels == [1, 1]


def test_numeric_threshold_boundaries_match_across_backends() -> None:
    pytest.importorskip("duckdb")
    frame = pl.DataFrame(
        {"id": [1, 2], "value": pl.Series([0.0, 0.1], dtype=pl.Float32)}
    )
    levels = []
    for backend in ("polars", "duckdb"):
        config = LinkConfig(
            backend=backend,  # type: ignore[arg-type]
            unique_id_column="id",
            comparisons=[NumericDiff(column="value", thresholds=(0.1,))],
        )
        levels.append(
            RecordLinker(frame, config).run().pairs["level__value"][0]
        )

    assert levels == [1, 1]


def test_decimal_numeric_differences_preserve_precision() -> None:
    pytest.importorskip("duckdb")
    frame = pl.DataFrame(
        {
            "id": [1, 2],
            "value": pl.Series(
                [
                    Decimal("100000000000000000000"),
                    Decimal("100000000000000000001"),
                ],
                dtype=pl.Decimal(38, 0),
            ),
        }
    )
    levels = []
    for backend in ("polars", "duckdb"):
        config = LinkConfig(
            backend=backend,  # type: ignore[arg-type]
            unique_id_column="id",
            comparisons=[NumericDiff(column="value", thresholds=(0.0,))],
        )
        levels.append(
            RecordLinker(frame, config).run().pairs["level__value"][0]
        )

    assert levels == [1, 1]


def test_opposite_decimal_signs_do_not_overflow() -> None:
    pytest.importorskip("duckdb")
    magnitude = Decimal("9" * 38)
    negative_magnitude = Decimal("-" + "9" * 38)
    frame = pl.DataFrame(
        {
            "id": [1, 2],
            "value": pl.Series(
                [magnitude, negative_magnitude],
                dtype=pl.Decimal(38, 0),
            ),
        }
    )
    levels = []
    for backend in ("polars", "duckdb"):
        config = LinkConfig(
            backend=backend,  # type: ignore[arg-type]
            unique_id_column="id",
            comparisons=[NumericDiff(column="value", thresholds=(1.0,))],
        )
        levels.append(
            RecordLinker(frame, config).run().pairs["level__value"][0]
        )

    assert levels == [1, 1]


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
def test_mixed_decimal_numeric_dtypes_are_rejected(backend: str) -> None:
    if backend == "duckdb":
        pytest.importorskip("duckdb")
    left = pl.DataFrame(
        {
            "id": [1],
            "value": pl.Series(
                [Decimal("9007199254740992")],
                dtype=pl.Decimal(38, 0),
            ),
        }
    )
    right = pl.DataFrame(
        {
            "id": [2],
            "value": pl.Series([9007199254740993], dtype=pl.Int64),
        }
    )
    config = LinkConfig(
        backend=backend,  # type: ignore[arg-type]
        mode="link",
        unique_id_column="id",
        comparisons=[NumericDiff(column="value", thresholds=(0.0,))],
    )

    with pytest.raises(
        ValueError,
        match=r"mixes Decimal and non-Decimal.*cast both sides",
    ):
        RecordLinker(left, config, right).run()


def test_duckdb_fuzzy_rejects_unicode_values() -> None:
    pytest.importorskip("duckdb")
    frame = pl.DataFrame({"id": [1, 2], "name": ["你好", "你号"]})
    config = LinkConfig(
        backend="duckdb",
        unique_id_column="id",
        comparisons=[FuzzyString(column="name")],
    )

    with pytest.raises(ValueError, match=r"ASCII.*Polars"):
        RecordLinker(frame, config).run()


def test_duckdb_linker_quotes_malicious_column_names() -> None:
    pytest.importorskip("duckdb")
    column = 'x") FROM (SELECT 1 AS x); SELECT 999; --'
    frame = pl.DataFrame({"id": [1, 2], column: ["a", "a"]})
    config = LinkConfig(
        backend="duckdb",
        unique_id_column="id",
        comparisons=[ExactMatch(column=column)],
    )
    result = RecordLinker(frame, config).run()
    assert result.pairs[f"level__{column}"].to_list() == [2]


def test_duckdb_linker_rejects_polars_int128() -> None:
    pytest.importorskip("duckdb")
    if not hasattr(pl, "Int128"):
        pytest.skip("Polars does not expose Int128")
    frame = pl.DataFrame(
        {"id": pl.Series([1, 2], dtype=pl.Int128), "name": ["a", "a"]}
    )
    config = LinkConfig(
        backend="duckdb",
        unique_id_column="id",
        comparisons=[ExactMatch(column="name")],
    )

    with pytest.raises(ValueError, match="does not support 128-bit"):
        RecordLinker(frame, config).run()


@pytest.mark.parametrize("role", ["id", "blocking", "comparison"])
def test_duckdb_rejects_float16_linkage_columns(role: str) -> None:
    pytest.importorskip("duckdb")
    if not hasattr(pl, "Float16"):
        pytest.skip("Polars does not expose Float16")
    frame = pl.DataFrame(
        {
            "id": [1, 2],
            "name": ["same", "same"],
            "value": pl.Series([1.0, 2.0], dtype=pl.Float16),
        }
    )
    config = LinkConfig(
        backend="duckdb",
        unique_id_column="value" if role == "id" else "id",
        comparisons=[
            ExactMatch(column="value" if role == "comparison" else "name")
        ],
        blocking_rules=[["value"]] if role == "blocking" else [],
    )

    with pytest.raises(ValueError, match=r"Float16.*Float32 or Float64"):
        RecordLinker(frame, config).run()


@pytest.mark.parametrize("role", ["id", "blocking"])
def test_duckdb_rejects_time_identity_columns(role: str) -> None:
    pytest.importorskip("duckdb")
    frame = pl.DataFrame(
        {
            "id": [1, 2],
            "name": ["same", "same"],
            "value": pl.Series([time(1), time(2)], dtype=pl.Time),
        }
    )
    config = LinkConfig(
        backend="duckdb",
        unique_id_column="value" if role == "id" else "id",
        comparisons=[ExactMatch(column="name")],
        blocking_rules=[["value"]] if role == "blocking" else [],
    )

    with pytest.raises(ValueError, match=r"Time.*Polars backend"):
        RecordLinker(frame, config).run()


def test_duckdb_rejects_duration_unique_ids() -> None:
    pytest.importorskip("duckdb")
    frame = pl.DataFrame(
        {
            "id": pl.Series(
                [timedelta(days=1), timedelta(days=2)],
                dtype=pl.Duration("us"),
            ),
            "name": ["same", "same"],
        }
    )
    config = LinkConfig(
        backend="duckdb",
        unique_id_column="id",
        comparisons=[ExactMatch(column="name")],
    )

    with pytest.raises(ValueError, match=r"Duration unique IDs.*Polars"):
        RecordLinker(frame, config).run()


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
@pytest.mark.parametrize("role", ["blocking", "comparison"])
def test_linker_rejects_object_comparison_or_blocking_columns(
    backend: str,
    role: str,
) -> None:
    if backend == "duckdb":
        pytest.importorskip("duckdb")
    frame = pl.DataFrame(
        {
            "id": [1, 2],
            "name": ["same", "same"],
            "value": pl.Series(["same", "same"], dtype=pl.Object),
        }
    )
    config = LinkConfig(
        backend=backend,  # type: ignore[arg-type]
        unique_id_column="id",
        comparisons=[
            ExactMatch(column="value" if role == "comparison" else "name")
        ],
        blocking_rules=[["value"]] if role == "blocking" else [],
    )

    with pytest.raises(ValueError, match=r"Object.*concrete scalar dtype"):
        RecordLinker(frame, config).run()


def test_polars_numeric_comparison_rejects_int128() -> None:
    if not hasattr(pl, "Int128"):
        pytest.skip("Polars does not expose Int128")
    frame = pl.DataFrame(
        {
            "id": [1, 2],
            "value": pl.Series(
                [-(2**127), 2**127 - 1],
                dtype=pl.Int128,
            ),
        }
    )
    config = LinkConfig(
        unique_id_column="id",
        comparisons=[NumericDiff(column="value")],
    )

    with pytest.raises(ValueError, match="unsupported 128-bit"):
        RecordLinker(frame, config).run()


def test_infinite_unique_ids_are_rejected() -> None:
    frame = pl.DataFrame({"id": [1.0, float("inf")], "name": ["same", "same"]})
    config = LinkConfig(
        unique_id_column="id",
        comparisons=[ExactMatch(column="name")],
    )

    with pytest.raises(ValueError, match="unique IDs must be finite"):
        RecordLinker(frame, config).run()


def test_cluster_labels_do_not_depend_on_input_or_edge_order() -> None:
    forward = cluster_from_pairs(
        np.array([1, 2, 3, 4]),
        np.array([[1, 2], [3, 4]]),
    )
    reversed_order = cluster_from_pairs(
        np.array([4, 3, 2, 1]),
        np.array([[4, 3], [2, 1]]),
    )
    assert forward == reversed_order


def test_date_ids_cluster() -> None:
    ids = pl.Series(
        [date(2020, 1, 1), date(2020, 1, 2)],
        dtype=pl.Date,
    ).to_numpy()

    clusters = cluster_from_pairs(
        ids,
        np.column_stack((ids[:1], ids[1:])),
    )

    assert len(set(clusters.values())) == 1


def test_nanosecond_datetime_ids_keep_datetime_keys() -> None:
    ids = pl.Series(
        [
            datetime(2020, 1, 1),
            datetime(2020, 1, 2),
        ],
        dtype=pl.Datetime("ns"),
    ).to_numpy()

    clusters = cluster_from_pairs(
        ids,
        np.column_stack((ids[:1], ids[1:])),
    )

    assert set(clusters) == {
        datetime(2020, 1, 1),
        datetime(2020, 1, 2),
    }
    assert len(set(clusters.values())) == 1


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
def test_linker_preserves_date_ids_for_rejected_fit(backend: str) -> None:
    if backend == "duckdb":
        pytest.importorskip("duckdb")
    frame = pl.DataFrame(
        {
            "id": pl.Series(
                [date(2020, 1, 1), date(2020, 1, 2)],
                dtype=pl.Date,
            ),
            "first": ["same", "same"],
            "second": ["same", "same"],
        }
    )
    config = LinkConfig(
        backend=backend,  # type: ignore[arg-type]
        unique_id_column="id",
        comparisons=[
            ExactMatch(column="first"),
            ExactMatch(column="second"),
        ],
        prior_match_probability=0.9,
        match_threshold_probability=0.5,
        em_max_iter=1,
    )

    result = RecordLinker(frame, config).run()

    assert set(result.clusters) == set(frame["id"].to_list())
    assert len(set(result.clusters.values())) == 2
    assert result.parameters["fit"]["status"] == "rejected"


@pytest.mark.parametrize("dtype_name", ["Int128", "UInt128"])
def test_polars_linker_preserves_128_bit_cluster_ids(
    dtype_name: str,
) -> None:
    if not hasattr(pl, dtype_name):
        pytest.skip(f"Polars does not expose {dtype_name}")
    dtype = getattr(pl, dtype_name)
    values = [2**100, 2**100 + 1]
    frame = pl.DataFrame(
        {
            "id": pl.Series(values, dtype=dtype),
            "name": ["same", "same"],
            "bucket": ["left", "right"],
        }
    )
    config = LinkConfig(
        unique_id_column="id",
        comparisons=[ExactMatch(column="name")],
        blocking_rules=[["bucket"]],
    )

    result = RecordLinker(frame, config).run()

    assert set(result.clusters) == set(values)


@pytest.mark.parametrize("backend", ["polars", "duckdb"])
def test_linker_preserves_timezone_aware_cluster_ids(backend: str) -> None:
    if backend == "duckdb":
        pytest.importorskip("duckdb")
    timezone = ZoneInfo("Asia/Kolkata")
    ids = [
        datetime(2026, 1, 1, tzinfo=timezone),
        datetime(2026, 1, 2, tzinfo=timezone),
    ]
    frame = pl.DataFrame(
        {
            "id": pl.Series(
                ids,
                dtype=pl.Datetime("us", time_zone="Asia/Kolkata"),
            ),
            "name": ["same", "same"],
            "bucket": ["left", "right"],
        }
    )
    config = LinkConfig(
        backend=backend,  # type: ignore[arg-type]
        unique_id_column="id",
        comparisons=[ExactMatch(column="name")],
        blocking_rules=[["bucket"]],
    )

    result = RecordLinker(frame, config).run()

    assert all(record_id in result.clusters for record_id in ids)
