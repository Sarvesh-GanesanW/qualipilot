"""Top-level orchestrator for the linking pipeline.

Stitches blocking, comparison, EM parameter fitting, and clustering.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from datapilot.linking.blocking import (
    attach_comparison_columns,
    build_candidate_pairs,
)
from datapilot.linking.cluster import cluster_from_pairs
from datapilot.linking.config import LinkConfig
from datapilot.linking.em import estimate_parameters, score_pairs

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LinkageResult:
    """Outcome of a ``RecordLinker`` run."""

    pairs: pl.DataFrame  # id_l, id_r, match_probability, levels per comp
    clusters: dict[object, int] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)

    def match_pairs(self, threshold: float) -> pl.DataFrame:
        """Return only pairs above the given probability threshold."""
        return self.pairs.filter(pl.col("match_probability") >= threshold)

    def summary(self) -> dict[str, Any]:
        total = self.pairs.height
        matches = int(
            self.pairs.get_column("match_probability")
            .ge(self.parameters.get("threshold", 0.9))
            .sum()
        )
        cluster_count = (
            len(set(self.clusters.values())) if self.clusters else 0
        )
        return {
            "candidate_pairs": total,
            "matched_pairs": matches,
            "clusters": cluster_count,
            "timings_ms": self.timings_ms,
            "lambda": float(self.parameters.get("lambda", 0.0)),
        }


class RecordLinker:
    """Fast Fellegi-Sunter record linker.

    Typical usage::

        from datapilot.linking import (
            RecordLinker, LinkConfig, ExactMatch, FuzzyString,
        )

        cfg = LinkConfig(
            unique_id_column="id",
            comparisons=[
                ExactMatch(column="email"),
                FuzzyString(column="name", thresholds=(0.92, 0.8)),
            ],
            blocking_rules=[["postcode"]],
        )
        linker = RecordLinker(df, cfg)
        result = linker.run()
        print(result.summary())
    """

    def __init__(
        self,
        df: pl.DataFrame,
        config: LinkConfig,
        df_right: pl.DataFrame | None = None,
    ) -> None:
        self._df = _ensure_polars(df)
        self._df_right = (
            _ensure_polars(df_right) if df_right is not None else None
        )
        self.config = config

    def run(self) -> LinkageResult:
        """Block, compare, learn, score, and cluster in one call."""
        if self.config.backend == "duckdb":
            # deferred import so duckdb stays an optional extra
            from datapilot.linking.duckdb_linker import (
                run_duckdb_linker,
            )

            return run_duckdb_linker(self._df, self.config)

        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        pairs = build_candidate_pairs(
            self._df,
            id_column=self.config.unique_id_column,
            blocking_rules=self.config.blocking_rules,
            mode=self.config.mode,
            df_right=self._df_right,
        )
        timings["blocking_ms"] = _ms_since(t0)

        if pairs.height == 0:
            logger.warning("no candidate pairs after blocking")
            return LinkageResult(
                pairs=pairs,
                clusters={},
                parameters={"lambda": 0.0, "threshold": 0.0},
                timings_ms=timings,
            )

        if pairs.height > self.config.max_pairs_hard_cap:
            raise MemoryError(
                f"blocking produced {pairs.height:,} pairs; "
                f"hard cap is {self.config.max_pairs_hard_cap:,}. "
                f"Tighten blocking_rules or raise max_pairs_hard_cap."
            )
        if pairs.height > self.config.max_pairs_warning:
            logger.warning(
                "blocking produced %d pairs; consider tighter rules",
                pairs.height,
            )

        t0 = time.perf_counter()
        compare_columns = [c.column for c in self.config.comparisons]
        decorated = attach_comparison_columns(
            pairs,
            self._df,
            self.config.unique_id_column,
            compare_columns,
            df_right=self._df_right,
        )
        timings["decorate_ms"] = _ms_since(t0)

        t0 = time.perf_counter()
        levels, n_levels = _assign_all_levels(
            decorated, self.config.comparisons
        )
        timings["compare_ms"] = _ms_since(t0)

        t0 = time.perf_counter()
        sample_size = self.config.em_sample_size
        if levels.shape[0] > sample_size:
            # fit on a random subsample so EM stays cheap; score all
            # pairs with the learned parameters after
            rng = np.random.default_rng(0)
            idx = rng.choice(levels.shape[0], size=sample_size, replace=False)
            em_levels = levels[idx]
            logger.info(
                "em fitting on %d/%d sampled pairs",
                sample_size,
                levels.shape[0],
            )
        else:
            em_levels = levels
        params = estimate_parameters(
            em_levels,
            n_levels,
            prior=self.config.prior_match_probability,
            max_iter=self.config.em_max_iter,
            tol=self.config.em_tolerance,
        )
        timings["em_ms"] = _ms_since(t0)

        t0 = time.perf_counter()
        probs = score_pairs(levels, params["m"], params["u"], params["lambda"])
        timings["score_ms"] = _ms_since(t0)

        scored = decorated.select(
            [pl.col("__id_l__"), pl.col("__id_r__")]
        ).with_columns(
            pl.Series("match_probability", probs.astype(np.float64))
        )
        # bolt level columns on for debuggability
        for i, comp in enumerate(self.config.comparisons):
            scored = scored.with_columns(
                pl.Series(
                    f"level__{comp.column}",
                    levels[:, i].astype(np.int8),
                )
            )

        t0 = time.perf_counter()
        clusters = _cluster_if_dedupe(
            self._df,
            scored,
            config=self.config,
        )
        timings["cluster_ms"] = _ms_since(t0)

        return LinkageResult(
            pairs=scored,
            clusters=clusters,
            parameters={
                **params,
                "threshold": self.config.match_threshold_probability,
            },
            timings_ms=timings,
        )


def _ensure_polars(df: Any) -> pl.DataFrame:
    if isinstance(df, pl.DataFrame):
        return df
    if type(df).__module__.startswith("pandas"):
        return pl.from_pandas(df)
    raise TypeError(f"unsupported frame type: {type(df).__name__}")


def _assign_all_levels(
    decorated: pl.DataFrame, comparisons: list[Any]
) -> tuple[np.ndarray, np.ndarray]:
    n = decorated.height
    c = len(comparisons)
    out = np.zeros((n, c), dtype=np.uint8)
    sizes = np.zeros(c, dtype=np.uint8)
    for i, comp in enumerate(comparisons):
        out[:, i] = comp.assign_levels(decorated)
        sizes[i] = comp.levels
    return out, sizes


def _cluster_if_dedupe(
    df: pl.DataFrame,
    scored: pl.DataFrame,
    *,
    config: LinkConfig,
) -> dict[object, int]:
    if config.mode != "dedupe":
        return {}
    threshold = config.match_threshold_probability
    confident = scored.filter(pl.col("match_probability") >= threshold)
    if confident.height == 0:
        # everyone lives in their own singleton cluster
        ids = df[config.unique_id_column].to_numpy()
        return {rid: int(i) for i, rid in enumerate(ids)}

    ids = df[config.unique_id_column].to_numpy()
    edges = np.column_stack(
        (
            confident["__id_l__"].to_numpy(),
            confident["__id_r__"].to_numpy(),
        )
    )
    return cluster_from_pairs(ids, edges)


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)
