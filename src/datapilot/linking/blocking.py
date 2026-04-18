"""Blocking = cheap pre-filter that avoids the N^2 comparison explosion.

A blocking rule is a list of columns; two records block together when
they agree on every column in the list. We run one polars self-join
per rule and union the resulting candidate pairs.

Cost model: a rule on column ``postcode`` where the most common
postcode has K records produces up to K^2 pairs. Pick rules that
leave the largest bucket under ~10k records.
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
    if mode == "link":
        if df_right is None:
            raise ValueError("df_right is required in link mode")
        left = df.rename({id_column: _ID_LEFT})
        right = df_right.rename({id_column: _ID_RIGHT})
    else:
        left = df.rename({id_column: _ID_LEFT})
        right = df.rename({id_column: _ID_RIGHT})

    if not blocking_rules:
        pairs = _cartesian(left, right)
    else:
        per_rule = [
            _join_on_rule(left, right, rule)
            for rule in blocking_rules
        ]
        pairs = pl.concat(per_rule, how="vertical_relaxed")

    pairs = pairs.unique(subset=[_ID_LEFT, _ID_RIGHT])

    if mode == "dedupe":
        # drop self pairs and keep a single orientation
        pairs = pairs.filter(pl.col(_ID_LEFT) < pl.col(_ID_RIGHT))

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
    if not rule:
        return _cartesian(left, right)
    return left.join(
        right, on=rule, how="inner", suffix="__r"
    ).select([_ID_LEFT, _ID_RIGHT])


def _cartesian(
    left: pl.DataFrame, right: pl.DataFrame
) -> pl.DataFrame:
    # tiny datasets only — we warn loudly when this path is hit
    logger.warning(
        "no blocking rules supplied; doing full cartesian product "
        "(%d x %d)",
        left.height,
        right.height,
    )
    return left.select(_ID_LEFT).join(
        right.select(_ID_RIGHT), how="cross"
    )


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
    left_src = df.select([id_column, *columns]).rename(
        {id_column: _ID_LEFT, **{c: f"{c}_l" for c in columns}}
    )
    right_source = df_right if df_right is not None else df
    right_src = right_source.select([id_column, *columns]).rename(
        {id_column: _ID_RIGHT, **{c: f"{c}_r" for c in columns}}
    )
    decorated = pairs.join(left_src, on=_ID_LEFT, how="inner").join(
        right_src, on=_ID_RIGHT, how="inner"
    )
    return decorated
