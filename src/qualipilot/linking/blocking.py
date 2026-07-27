"""Blocking = cheap pre-filter that avoids the N^2 comparison explosion.

A blocking rule is a list of columns; two records block together when
they agree on every column in the list. We run one polars self-join
per rule and union the resulting candidate pairs.

Cost model: a rule on column ``postcode`` where the most common
postcode has K records can produce a quadratic number of pairs.
"""

from __future__ import annotations

import logging

import polars as pl

logger = logging.getLogger(__name__)

_ID_LEFT = "__id_l__"
_ID_RIGHT = "__id_r__"


def build_candidate_pairs(
    df: pl.DataFrame,
    *,
    id_column: str,
    blocking_rules: list[list[str]],
    mode: str,
    df_right: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Return a polars frame of candidate (left, right) record pairs.

    The returned frame has exactly two columns, ``__id_l__`` and
    ``__id_r__``, carrying the unique-id values from each side. Other
    columns are re-joined by the caller once it knows which fields to
    compare.

    Args:
        df: primary dataset (dedupe: only input; link: left side).
        id_column: column whose values are unique within ``df``.
        blocking_rules: list of rules, each a list of column names
            that must agree. Empty list means cartesian product.
        mode: ``"dedupe"`` for self-linkage or ``"link"`` for two
            input tables.
        df_right: right-side dataset when ``mode="link"``.

    Returns:
        Deduplicated polars frame of candidate pairs.
    """
    blocking_columns = list(
        dict.fromkeys(column for rule in blocking_rules for column in rule)
    )
    if mode == "link":
        if df_right is None:
            raise ValueError("df_right is required in link mode")
        left = _blocking_source(
            df,
            id_column=id_column,
            id_alias=_ID_LEFT,
            blocking_columns=blocking_columns,
        )
        right = _blocking_source(
            df_right,
            id_column=id_column,
            id_alias=_ID_RIGHT,
            blocking_columns=blocking_columns,
        )
    else:
        left = _blocking_source(
            df,
            id_column=id_column,
            id_alias=_ID_LEFT,
            blocking_columns=blocking_columns,
        )
        right = _blocking_source(
            df,
            id_column=id_column,
            id_alias=_ID_RIGHT,
            blocking_columns=blocking_columns,
        )

    if not blocking_rules:
        pairs = _cartesian(left, right)
    else:
        per_rule = [
            _join_on_rule(left, right, rule) for rule in blocking_rules
        ]
        pairs = pl.concat(per_rule, how="vertical_relaxed")

    if mode == "dedupe":
        # drop self pairs and keep a single orientation
        pairs = pairs.filter(pl.col(_ID_LEFT) < pl.col(_ID_RIGHT))

    pairs = pairs.unique(
        subset=[_ID_LEFT, _ID_RIGHT],
    ).sort([_ID_LEFT, _ID_RIGHT])

    logger.info(
        "blocking produced %d candidate pairs (mode=%s, rules=%d)",
        pairs.height,
        mode,
        len(blocking_rules),
    )
    return pairs


def _join_on_rule(
    left: pl.DataFrame,
    right: pl.DataFrame,
    rule: list[str],
) -> pl.DataFrame:
    return (
        _valid_blocking_rows(left, rule)
        .join(
            _valid_blocking_rows(right, rule),
            on=rule,
            how="inner",
            suffix="__r",
        )
        .select([_ID_LEFT, _ID_RIGHT])
    )


def _cartesian(left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
    # tiny datasets only — we warn loudly when this path is hit
    logger.warning(
        "no blocking rules supplied; doing full cartesian product (%d x %d)",
        left.height,
        right.height,
    )
    return left.select(_ID_LEFT).join(right.select(_ID_RIGHT), how="cross")


def attach_comparison_columns(
    pairs: pl.DataFrame,
    df: pl.DataFrame,
    id_column: str,
    columns: list[str],
    *,
    df_right: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Decorate candidate pairs with left/right copies of comparison columns.

    Output columns for each comparison column ``c`` are ``c_l`` and
    ``c_r``.
    """
    left_src = df.select(
        pl.col(id_column).alias(_ID_LEFT),
        *[pl.col(column).alias(f"{column}_l") for column in columns],
    )
    right_source = df_right if df_right is not None else df
    right_src = right_source.select(
        pl.col(id_column).alias(_ID_RIGHT),
        *[pl.col(column).alias(f"{column}_r") for column in columns],
    )
    return (
        pairs.join(left_src, on=_ID_LEFT, how="inner")
        .join(right_src, on=_ID_RIGHT, how="inner")
        .sort([_ID_LEFT, _ID_RIGHT])
    )


def estimate_candidate_pair_upper_bound(
    df: pl.DataFrame,
    *,
    blocking_rules: list[list[str]],
    mode: str,
    df_right: pl.DataFrame | None = None,
) -> int:
    """Estimate candidates without materialising the pair join.

    The sum across rules is a conservative union bound because pairs may
    satisfy more than one rule.
    """
    if mode == "link":
        if df_right is None:
            raise ValueError("df_right is required in link mode")
        right = df_right
    else:
        right = df
    if not blocking_rules:
        if mode == "dedupe":
            return df.height * (df.height - 1) // 2
        return df.height * right.height

    total = 0
    for rule in blocking_rules:
        left_counts = _blocking_counts(df, rule, "__left_count__")
        if mode == "dedupe":
            total += sum(
                int(count) * (int(count) - 1) // 2
                for count in left_counts["__left_count__"]
            )
            continue
        right_counts = _blocking_counts(right, rule, "__right_count__")
        joined = left_counts.join(right_counts, on=rule, how="inner")
        total += sum(
            int(left_count) * int(right_count)
            for left_count, right_count in joined.select(
                ["__left_count__", "__right_count__"]
            ).iter_rows()
        )
    # Exact cross-rule overlap requires constructing the pairs that this
    # guard exists to avoid, so retain the safe union upper bound.
    return total


def _blocking_source(
    df: pl.DataFrame,
    *,
    id_column: str,
    id_alias: str,
    blocking_columns: list[str],
) -> pl.DataFrame:
    return df.select(
        pl.col(id_column).alias(id_alias),
        *[pl.col(column) for column in blocking_columns],
    )


def _blocking_counts(
    df: pl.DataFrame,
    rule: list[str],
    count_column: str,
) -> pl.DataFrame:
    return _valid_blocking_rows(df, rule).group_by(rule).len(name=count_column)


def _valid_blocking_rows(
    frame: pl.DataFrame,
    rule: list[str],
) -> pl.DataFrame:
    predicates = []
    for column in rule:
        valid = pl.col(column).is_not_null()
        if frame.schema[column].is_float():
            valid &= pl.col(column).is_not_nan()
        predicates.append(valid)
    return frame.filter(pl.all_horizontal(predicates))
