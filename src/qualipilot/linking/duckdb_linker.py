"""DuckDB-powered record linker.

Blocking and comparison-level assignment run in DuckDB. EM and final
probability scoring use the same NumPy implementation as the Polars
backend.
"""

from __future__ import annotations

import logging
import time
from typing import Any, cast

import numpy as np
import polars as pl

from qualipilot.engines._duckdb_sql import quote_identifier
from qualipilot.linking.cluster import cluster_from_pairs
from qualipilot.linking.comparisons import (
    ExactMatch,
    FuzzyString,
    NumericDiff,
)
from qualipilot.linking.config import LinkConfig
from qualipilot.linking.em import (
    build_fit_diagnostics,
    estimate_parameters,
    score_pairs,
)
from qualipilot.linking.linker import (
    LinkageResult,
    _build_scored_pairs,
    _empty_linkage_result,
)

logger = logging.getLogger(__name__)


def run_duckdb_linker(
    df: pl.DataFrame,
    config: LinkConfig,
    df_right: pl.DataFrame | None = None,
) -> LinkageResult:
    """Run the linker end-to-end with DuckDB as the compute engine."""
    import duckdb

    _require_ascii_fuzzy_values(df, config)
    if df_right is not None:
        _require_ascii_fuzzy_values(df_right, config)
    con = duckdb.connect(":memory:")
    try:
        if config.duckdb_threads is not None:
            con.execute("SET threads = ?", [config.duckdb_threads])
        return _run_with_connection(con, df, config, df_right)
    finally:
        con.close()


def _run_with_connection(  # noqa: PLR0915
    con: Any,
    df: pl.DataFrame,
    config: LinkConfig,
    df_right: pl.DataFrame | None,
) -> LinkageResult:
    timings: dict[str, float] = {}
    right = df_right if df_right is not None else df
    con.register("_t_left", df.to_arrow())
    con.register("_t_right", right.to_arrow())

    # ---- blocking + comparison in one SQL pass ----------------------
    t0 = time.perf_counter()
    pair_columns = [c.column for c in config.comparisons]
    blocking_columns = [
        column for rule in config.blocking_rules for column in rule
    ]
    source_columns = list(
        dict.fromkeys(
            [*pair_columns, *blocking_columns, config.unique_id_column]
        )
    )
    left_selects = _renamed_cols(
        source_columns,
        suffix="_l",
        id_col=config.unique_id_column,
        id_alias="__id_l__",
    )
    right_selects = _renamed_cols(
        source_columns,
        suffix="_r",
        id_col=config.unique_id_column,
        id_alias="__id_r__",
    )
    con.execute(f"CREATE VIEW _l AS SELECT {left_selects} FROM _t_left")
    con.execute(f"CREATE VIEW _r AS SELECT {right_selects} FROM _t_right")

    float_blocking_columns = {
        column
        for column in blocking_columns
        if df.schema[column].is_float() or right.schema[column].is_float()
    }
    blocking_sql = _compose_blocking_sql(
        config.blocking_rules,
        float_blocking_columns,
    )
    dedupe_clause = (
        f"AND _l.{quote_identifier('__id_l__')} "
        f"< _r.{quote_identifier('__id_r__')}"
        if config.mode == "dedupe"
        else ""
    )
    level_exprs = [
        _level_expression(
            comparison,
            float_values=(
                df.schema[comparison.column].is_float()
                or right.schema[comparison.column].is_float()
            ),
            integer_values=(
                df.schema[comparison.column].is_integer()
                and right.schema[comparison.column].is_integer()
            ),
            decimal_values=(
                df.schema[comparison.column].is_decimal()
                and right.schema[comparison.column].is_decimal()
            ),
        )
        for comparison in config.comparisons
    ]
    level_selects = ",\n  ".join(level_exprs)

    query = f"""
    CREATE TEMP TABLE pairs AS
    SELECT
        _l.{quote_identifier("__id_l__")},
        _r.{quote_identifier("__id_r__")},
        {level_selects}
    FROM _l JOIN _r
        ON {blocking_sql}
        {dedupe_clause}
    """
    con.execute(query)
    timings["blocking_compare_ms"] = _ms_since(t0)

    count_row = con.execute("SELECT COUNT(*) FROM pairs").fetchone()
    if count_row is None:
        raise RuntimeError("duckdb COUNT returned no row")
    n_pairs_quick = count_row[0]
    if n_pairs_quick > config.max_pairs_hard_cap:
        raise MemoryError(
            f"duckdb blocking produced {n_pairs_quick:,} pairs; "
            f"hard cap is {config.max_pairs_hard_cap:,}. "
            f"Tighten blocking_rules or raise max_pairs_hard_cap."
        )

    # pull the level matrix as a numpy array for the EM step
    t0 = time.perf_counter()
    pair_df = con.execute(
        f"SELECT * FROM pairs ORDER BY "
        f"{quote_identifier('__id_l__')}, {quote_identifier('__id_r__')}"
    ).to_arrow_table()
    timings["fetch_ms"] = _ms_since(t0)

    n_pairs = pair_df.num_rows
    if n_pairs == 0:
        empty_pairs = cast(pl.DataFrame, pl.from_arrow(pair_df)).select(
            ["__id_l__", "__id_r__"]
        )
        return _empty_linkage_result(
            df,
            empty_pairs,
            config,
            timings,
        )

    level_cols = [f"level__{comp.column}" for comp in config.comparisons]
    levels = np.column_stack(
        [
            pair_df.column(c).to_numpy(zero_copy_only=False).astype(np.uint8)
            for c in level_cols
        ]
    )
    n_levels = np.array(
        [comp.levels for comp in config.comparisons],
        dtype=np.uint8,
    )

    # ---- EM (numpy, sampled if huge) -------------------------------
    t0 = time.perf_counter()
    sample_size = config.em_sample_size
    em_levels = (
        levels[
            np.random.default_rng(config.em_random_seed).choice(
                n_pairs, sample_size, replace=False
            )
        ]
        if n_pairs > sample_size
        else levels
    )
    params = estimate_parameters(
        em_levels,
        n_levels,
        prior=config.prior_match_probability,
        max_iter=config.em_max_iter,
        tol=config.em_tolerance,
    )
    fit = build_fit_diagnostics(
        params,
        n_levels,
        [comparison.column for comparison in config.comparisons],
        sampled_pair_count=em_levels.shape[0],
        candidate_pair_count=n_pairs,
    )
    timings["em_ms"] = _ms_since(t0)

    # ---- score + cluster -------------------------------------------
    t0 = time.perf_counter()
    if fit["status"] != "rejected":
        probs = score_pairs(
            levels,
            params["m"],
            params["u"],
            params["lambda"],
        )
    else:
        logger.error("unsafe linkage fit rejected: %s", fit["reason"])
        probs = np.zeros(levels.shape[0], dtype=np.float32)
    timings["score_ms"] = _ms_since(t0)

    decorated = cast(pl.DataFrame, pl.from_arrow(pair_df))
    scored = _build_scored_pairs(
        decorated,
        levels,
        probs,
        config,
    )

    t0 = time.perf_counter()
    clusters: dict[Any, int] = {}
    if config.mode == "dedupe":
        confident = scored.filter(
            pl.col("match_probability") >= config.match_threshold_probability
        )
        ids = df[config.unique_id_column].to_list()
        if confident.height == 0:
            clusters = cluster_from_pairs(
                ids,
                (),
            )
        else:
            edges = confident.select(["__id_l__", "__id_r__"]).iter_rows()
            clusters = cluster_from_pairs(ids, edges)
    timings["cluster_ms"] = _ms_since(t0)

    return LinkageResult(
        pairs=scored,
        clusters=clusters,
        parameters={
            "m": params["m"],
            "u": params["u"],
            "lambda": params["lambda"],
            "threshold": config.match_threshold_probability,
            "fit": fit,
        },
        timings_ms=timings,
    )


# ---- helpers --------------------------------------------------------


def _renamed_cols(
    cols: list[str], *, suffix: str, id_col: str, id_alias: str
) -> str:
    out = []
    for c in cols:
        if c == id_col:
            out.append(
                f"{quote_identifier(c)} AS {quote_identifier(id_alias)}"
            )
        else:
            out.append(
                f"{quote_identifier(c)} AS {quote_identifier(c + suffix)}"
            )
    return ", ".join(out)


def _compose_blocking_sql(
    rules: list[list[str]],
    float_columns: set[str],
) -> str:
    if not rules:
        return "TRUE"
    ors = []
    for rule in rules:
        comparisons = []
        for column in rule:
            left = f"_l.{quote_identifier(f'{column}_l')}"
            right = f"_r.{quote_identifier(f'{column}_r')}"
            comparison = f"{left} = {right}"
            if column in float_columns:
                comparison = (
                    f"NOT isnan({left}) AND NOT isnan({right}) "
                    f"AND {comparison}"
                )
            comparisons.append(comparison)
        ands = " AND ".join(comparisons)
        ors.append(f"({ands})")
    return "(" + " OR ".join(ors) + ")"


def _level_expression(
    comp: Any,
    *,
    float_values: bool = False,
    integer_values: bool = False,
    decimal_values: bool = False,
) -> str:
    """Translate a python Comparison into a DuckDB CASE expression."""
    col = comp.column
    alias = f"level__{col}"
    lcol = f"_l.{quote_identifier(f'{col}_l')}"
    rcol = f"_r.{quote_identifier(f'{col}_r')}"
    quoted_alias = quote_identifier(alias)
    missing = f"{lcol} IS NULL OR {rcol} IS NULL"
    if float_values:
        missing += (
            f" OR isnan(CAST({lcol} AS DOUBLE))"
            f" OR isnan(CAST({rcol} AS DOUBLE))"
        )

    if isinstance(comp, ExactMatch):
        return (
            f"CASE "
            f"WHEN {missing} THEN 0 "
            f"WHEN {lcol} = {rcol} THEN 2 "
            f"ELSE 1 END AS {quoted_alias}"
        )
    if isinstance(comp, FuzzyString):
        thresholds = sorted(comp.thresholds)
        similarity = (
            f"CASE WHEN CAST({lcol} AS VARCHAR) = '' "
            f"AND CAST({rcol} AS VARCHAR) = '' THEN 1.0 "
            f"ELSE jaro_winkler_similarity("
            f"CAST({lcol} AS VARCHAR), CAST({rcol} AS VARCHAR)) END"
        )
        expr_parts: list[str] = [f"WHEN {missing} THEN 0"]
        for rank, t in enumerate(reversed(thresholds), start=2):
            level = 2 + len(thresholds) - (rank - 2) - 1
            expr_parts.append(f"WHEN {similarity} >= {t} THEN {level}")
        expr_parts.append("ELSE 1")
        return "CASE " + " ".join(expr_parts) + f" END AS {quoted_alias}"
    if isinstance(comp, NumericDiff):
        thresholds = sorted(comp.thresholds)
        parts: list[str] = [f"WHEN {missing} THEN 0"]
        if decimal_values:
            same_sign = (
                f"(({lcol} >= 0 AND {rcol} >= 0) "
                f"OR ({lcol} < 0 AND {rcol} < 0))"
            )
            decimal_diff = (
                "CAST(ABS("
                f"CASE WHEN {same_sign} THEN {lcol} ELSE 0 END - "
                f"CASE WHEN {same_sign} THEN {rcol} ELSE 0 END"
                ") AS DOUBLE)"
            )
            float_diff = (
                f"ABS(CAST({lcol} AS DOUBLE) - CAST({rcol} AS DOUBLE))"
            )
            difference = (
                f"CASE WHEN {same_sign} THEN {decimal_diff} "
                f"ELSE {float_diff} END"
            )
        else:
            cast_type = "HUGEINT" if integer_values else "DOUBLE"
            left_value = f"CAST({lcol} AS {cast_type})"
            right_value = f"CAST({rcol} AS {cast_type})"
            difference = f"ABS({left_value} - {right_value})"
        for index, threshold in enumerate(thresholds):
            level = 1 + len(thresholds) - index
            parts.append(f"WHEN {difference} <= {threshold!r} THEN {level}")
        parts.append("ELSE 1")
        return "CASE " + " ".join(parts) + f" END AS {quoted_alias}"
    raise TypeError(f"unsupported comparison: {type(comp).__name__}")


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


def _require_ascii_fuzzy_values(
    frame: pl.DataFrame,
    config: LinkConfig,
) -> None:
    for comparison in config.comparisons:
        if not isinstance(comparison, FuzzyString):
            continue
        values = frame[comparison.column].cast(pl.String)
        if values.str.contains(r"[^\x00-\x7f]").fill_null(False).any():
            raise ValueError(
                "DuckDB fuzzy matching supports ASCII values only; "
                "use the Polars backend for Unicode data"
            )
