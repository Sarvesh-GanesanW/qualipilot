"""DuckDB-powered record linker.

Runs blocking + comparison + scoring inside a single DuckDB query
plan. EM still happens in numpy because the M-step is too
stateful to express cleanly as a CTE chain, but the heavy pair
generation and per-pair level assignment stay in SQL.

In practice this beats the polars+numpy path by ~2-3x on datasets
with >1M candidate pairs because:
    * DuckDB SIMD-vectorises the blocking join
    * ``jaro_winkler_similarity`` in DuckDB is SIMD-accelerated
    * pair level assignment runs in one expression tree, no round
      trips to Python per comparison
"""

from __future__ import annotations

import logging
import time
from typing import Any, cast

import numpy as np
import polars as pl

from qualipilot.linking.cluster import cluster_from_pairs
from qualipilot.linking.comparisons import (
    ExactMatch,
    FuzzyString,
    NumericDiff,
)
from qualipilot.linking.config import LinkConfig
from qualipilot.linking.em import estimate_parameters, score_pairs
from qualipilot.linking.linker import LinkageResult

logger = logging.getLogger(__name__)


def run_duckdb_linker(  # noqa: PLR0915
    df: pl.DataFrame, config: LinkConfig
) -> LinkageResult:
    """Run the linker end-to-end with DuckDB as the compute engine."""
    import duckdb

    timings: dict[str, float] = {}
    con = duckdb.connect(":memory:")
    con.execute("PRAGMA threads=8")
    con.register("_t", df.to_arrow())

    # ---- blocking + comparison in one SQL pass ----------------------
    t0 = time.perf_counter()
    pair_columns = [c.column for c in config.comparisons]
    left_selects = _renamed_cols(
        [*pair_columns, config.unique_id_column],
        suffix="_l",
        id_col=config.unique_id_column,
        id_alias="__id_l__",
    )
    right_selects = _renamed_cols(
        [*pair_columns, config.unique_id_column],
        suffix="_r",
        id_col=config.unique_id_column,
        id_alias="__id_r__",
    )
    con.execute(f"CREATE VIEW _l AS SELECT {left_selects} FROM _t")
    con.execute(f"CREATE VIEW _r AS SELECT {right_selects} FROM _t")

    blocking_sql = _compose_blocking_sql(config.blocking_rules)
    dedupe_clause = (
        "AND _l.__id_l__ < _r.__id_r__" if config.mode == "dedupe" else ""
    )
    level_exprs = [_level_expression(comp) for comp in config.comparisons]
    level_selects = ",\n  ".join(level_exprs)

    query = f"""
    CREATE TEMP TABLE pairs AS
    SELECT DISTINCT
        _l.__id_l__,
        _r.__id_r__,
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
    pair_df = con.execute("SELECT * FROM pairs").fetch_arrow_table()
    timings["fetch_ms"] = _ms_since(t0)

    n_pairs = pair_df.num_rows
    if n_pairs == 0:
        return LinkageResult(
            pairs=cast(pl.DataFrame, pl.from_arrow(pair_df)),
            clusters={},
            parameters={"lambda": 0.0, "threshold": 0.0},
            timings_ms=timings,
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
    timings["em_ms"] = _ms_since(t0)

    # ---- score + cluster -------------------------------------------
    t0 = time.perf_counter()
    probs = score_pairs(levels, params["m"], params["u"], params["lambda"])
    timings["score_ms"] = _ms_since(t0)

    scored = cast(pl.DataFrame, pl.from_arrow(pair_df)).with_columns(
        pl.Series("match_probability", probs.astype(np.float64))
    )

    t0 = time.perf_counter()
    clusters: dict[Any, int] = {}
    if config.mode == "dedupe":
        confident = scored.filter(
            pl.col("match_probability") >= config.match_threshold_probability
        )
        ids = df[config.unique_id_column].to_numpy()
        if confident.height == 0:
            clusters = {rid: int(i) for i, rid in enumerate(ids)}
        else:
            edges = np.column_stack(
                (
                    confident["__id_l__"].to_numpy(),
                    confident["__id_r__"].to_numpy(),
                )
            )
            clusters = cluster_from_pairs(ids, edges)
    timings["cluster_ms"] = _ms_since(t0)

    return LinkageResult(
        pairs=scored,
        clusters=clusters,
        parameters={
            **params,
            "threshold": config.match_threshold_probability,
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
            out.append(f'"{c}" AS {id_alias}')
        else:
            out.append(f'"{c}" AS "{c}{suffix}"')
    return ", ".join(out)


def _compose_blocking_sql(rules: list[list[str]]) -> str:
    if not rules:
        return "TRUE"
    ors = []
    for rule in rules:
        if not rule:
            ors.append("TRUE")
            continue
        ands = " AND ".join(f'_l."{c}_l" = _r."{c}_r"' for c in rule)
        ors.append(f"({ands})")
    return "(" + " OR ".join(ors) + ")"


def _level_expression(comp: Any) -> str:
    """Translate a python Comparison into a DuckDB CASE expression."""
    col = comp.column
    alias = f"level__{col}"
    lcol = f'_l."{col}_l"'
    rcol = f'_r."{col}_r"'

    if isinstance(comp, ExactMatch):
        return (
            f"CASE "
            f"WHEN {lcol} IS NULL OR {rcol} IS NULL THEN 0 "
            f"WHEN {lcol} = {rcol} THEN 2 "
            f"ELSE 1 END AS {alias}"
        )
    if isinstance(comp, FuzzyString):
        thresholds = sorted(comp.thresholds)
        # build a nested CASE starting from the strictest threshold
        # (highest level) going down to "below lowest -> level 1"
        expr_parts: list[str] = [
            f"WHEN {lcol} IS NULL OR {rcol} IS NULL THEN 0"
        ]
        for rank, t in enumerate(reversed(thresholds), start=2):
            level = 2 + len(thresholds) - (rank - 2) - 1
            expr_parts.append(
                f"WHEN jaro_winkler_similarity("
                f"CAST({lcol} AS VARCHAR), CAST({rcol} AS VARCHAR)"
                f") >= {t} THEN {level}"
            )
        expr_parts.append("ELSE 1")
        return "CASE " + " ".join(expr_parts) + f" END AS {alias}"
    if isinstance(comp, NumericDiff):
        thresholds = sorted(comp.thresholds, reverse=True)
        parts: list[str] = [f"WHEN {lcol} IS NULL OR {rcol} IS NULL THEN 0"]
        for rank, t in enumerate(thresholds, start=2):
            parts.append(f"WHEN ABS({lcol} - {rcol}) <= {t} THEN {rank}")
        parts.append("ELSE 1")
        return "CASE " + " ".join(parts) + f" END AS {alias}"
    raise TypeError(f"unsupported comparison: {type(comp).__name__}")


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)
