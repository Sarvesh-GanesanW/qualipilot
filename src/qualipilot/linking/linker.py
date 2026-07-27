"""Top-level orchestrator for the linking pipeline.

Stitches blocking, comparison, EM parameter fitting, and clustering.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from qualipilot.linking.blocking import (
    attach_comparison_columns,
    build_candidate_pairs,
    estimate_candidate_pair_upper_bound,
)
from qualipilot.linking.cluster import cluster_from_pairs
from qualipilot.linking.comparisons import ExactMatch, FuzzyString, NumericDiff
from qualipilot.linking.config import LinkConfig, StringNormalization
from qualipilot.linking.consolidate import (
    ConsolidationConfig,
    ConsolidationResult,
    _validate_consolidation_frame,
    consolidate_records,
)
from qualipilot.linking.em import estimate_parameters, score_pairs

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
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be finite and between 0 and 1")
        return self.pairs.filter(pl.col("match_probability") >= threshold)

    def summary(self) -> dict[str, Any]:
        total = self.pairs.height
        threshold = float(self.parameters.get("threshold", 0.9))
        matches = int(
            self.pairs.get_column("match_probability").ge(threshold).sum()
        )
        cluster_count = (
            len(set(self.clusters.values())) if self.clusters else 0
        )
        return {
            "candidate_pairs": total,
            "matched_pairs": matches,
            "match_threshold_probability": threshold,
            "clusters": cluster_count,
            "timings_ms": self.timings_ms,
            "lambda": float(self.parameters.get("lambda", 0.0)),
        }


@dataclass(frozen=True, slots=True)
class DeduplicationResult:
    """Linkage evidence and its consolidated record set."""

    linkage: LinkageResult
    consolidation: ConsolidationResult


class RecordLinker:
    """Fellegi-Sunter record linker.

    Typical usage::

        from qualipilot.linking import (
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
        _validate_link_inputs(
            self._df,
            self.config,
            self._df_right,
        )
        df = normalize_records(self._df, self.config)
        df_right = (
            normalize_records(self._df_right, self.config)
            if self._df_right is not None
            else None
        )
        estimated_pairs = estimate_candidate_pair_upper_bound(
            df,
            blocking_rules=self.config.blocking_rules,
            mode=self.config.mode,
            df_right=df_right,
        )
        _guard_pair_count(estimated_pairs, self.config, estimated=True)

        if self.config.backend == "duckdb":
            # deferred import so duckdb stays an optional extra
            from qualipilot.linking.duckdb_linker import (
                run_duckdb_linker,
            )

            return run_duckdb_linker(
                df,
                self.config,
                df_right,
            )

        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        pairs = build_candidate_pairs(
            df,
            id_column=self.config.unique_id_column,
            blocking_rules=self.config.blocking_rules,
            mode=self.config.mode,
            df_right=df_right,
        )
        timings["blocking_ms"] = _ms_since(t0)

        if pairs.height == 0:
            logger.warning("no candidate pairs after blocking")
            return _empty_linkage_result(
                self._df,
                pairs,
                self.config,
                timings,
            )

        _guard_pair_count(pairs.height, self.config)

        t0 = time.perf_counter()
        compare_columns = [c.column for c in self.config.comparisons]
        decorated = attach_comparison_columns(
            pairs,
            df,
            self.config.unique_id_column,
            compare_columns,
            df_right=df_right,
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
            rng = np.random.default_rng(self.config.em_random_seed)
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

        scored = _build_scored_pairs(
            decorated,
            levels,
            probs,
            self.config,
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

    def deduplicate(
        self,
        consolidation: ConsolidationConfig,
    ) -> DeduplicationResult:
        """Match, normalize, merge, and remove redundant records."""
        if self.config.mode != "dedupe" or self._df_right is not None:
            raise ValueError("deduplicate() requires dedupe mode")
        _validate_link_inputs(self._df, self.config, None)
        normalized = normalize_records(self._df, self.config)
        _validate_consolidation_frame(
            normalized,
            self.config.unique_id_column,
            consolidation,
        )
        linkage = self.run()
        return DeduplicationResult(
            linkage=linkage,
            consolidation=consolidate_records(
                normalized,
                id_column=self.config.unique_id_column,
                clusters=linkage.clusters,
                config=consolidation,
            ),
        )


def _ensure_polars(df: Any) -> pl.DataFrame:
    if isinstance(df, pl.DataFrame):
        return df
    if type(df).__module__.startswith("pandas"):
        return pl.from_pandas(df)
    raise TypeError(f"unsupported frame type: {type(df).__name__}")


def _validate_link_inputs(  # noqa: PLR0912
    df: pl.DataFrame,
    config: LinkConfig,
    df_right: pl.DataFrame | None,
) -> None:
    if config.mode == "link":
        if df_right is None:
            raise ValueError("df_right is required in link mode")
        frames = [("left", df), ("right", df_right)]
    else:
        if df_right is not None:
            raise ValueError("df_right is only valid in link mode")
        frames = [("input", df)]

    blocking_columns = {
        column for rule in config.blocking_rules for column in rule
    }
    comparison_columns = {
        comparison.column for comparison in config.comparisons
    }
    _validate_normalization_inputs(frames, config)
    required = {
        config.unique_id_column,
        *comparison_columns,
        *blocking_columns,
    }
    for label, frame in frames:
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(
                f"{label} frame is missing required columns: {missing}"
            )
        _validate_linkage_dtypes(frame, config, label=label)
        _validate_unique_ids(
            frame,
            config.unique_id_column,
            label=label,
        )
        for comparison in config.comparisons:
            if isinstance(comparison, NumericDiff) and str(
                frame.schema[comparison.column]
            ) in {"Int128", "UInt128"}:
                raise ValueError(
                    f"{label} column {comparison.column!r} uses an "
                    "unsupported 128-bit numeric type"
                )
            if (
                isinstance(comparison, NumericDiff)
                and not frame.schema[comparison.column].is_numeric()
            ):
                raise ValueError(
                    f"{label} column {comparison.column!r} must be numeric"
                )
            if (
                isinstance(comparison, NumericDiff)
                and frame.schema[comparison.column].is_float()
            ):
                has_infinite = frame.select(
                    pl.col(comparison.column).is_infinite().any()
                ).item()
                if has_infinite:
                    raise ValueError(
                        f"{label} column {comparison.column!r} "
                        "must not contain infinite values"
                    )
            if isinstance(comparison, FuzzyString) and frame.schema[
                comparison.column
            ].base_type() not in {pl.String, pl.Categorical, pl.Enum}:
                raise ValueError(
                    f"{label} column {comparison.column!r} must be string-like"
                )

    if df_right is not None:
        for rule in config.blocking_rules:
            for column in rule:
                if (
                    column not in config.normalization
                    and df.schema[column] != df_right.schema[column]
                ):
                    raise ValueError(
                        f"blocking column {column!r} has incompatible "
                        "left/right dtypes"
                    )
        for comparison in config.comparisons:
            if (
                isinstance(comparison, ExactMatch)
                and comparison.column not in config.normalization
                and df.schema[comparison.column]
                != df_right.schema[comparison.column]
            ):
                raise ValueError(
                    f"exact-match column {comparison.column!r} has "
                    "incompatible left/right dtypes"
                )
            if isinstance(comparison, NumericDiff):
                left_dtype = df.schema[comparison.column]
                right_dtype = df_right.schema[comparison.column]
                if left_dtype.is_decimal() != right_dtype.is_decimal():
                    raise ValueError(
                        f"numeric comparison column {comparison.column!r} "
                        "mixes Decimal and non-Decimal dtypes; cast both "
                        "sides to compatible dtypes before linking"
                    )

    generated = {"__id_l__", "__id_r__", "__left_count__", "__right_count__"}
    conflict = sorted(generated & blocking_columns)
    if conflict:
        raise ValueError(f"blocking columns use reserved names: {conflict}")
    comparison_aliases = [
        alias
        for comparison in config.comparisons
        for alias in (comparison.left_col(), comparison.right_col())
    ]
    if len(set(comparison_aliases)) != len(comparison_aliases) or (
        generated & set(comparison_aliases)
    ):
        raise ValueError("comparison columns produce reserved output names")


def _validate_normalization_inputs(
    frames: list[tuple[str, pl.DataFrame]],
    config: LinkConfig,
) -> None:
    columns = set(config.normalization)
    if not columns:
        return
    string_types = {pl.String, pl.Categorical, pl.Enum}
    for label, frame in frames:
        missing = sorted(columns - set(frame.columns))
        if missing:
            raise ValueError(
                f"{label} frame is missing normalization columns: {missing}"
            )
        non_string = sorted(
            column
            for column in columns
            if frame.schema[column].base_type() not in string_types
        )
        if non_string:
            raise ValueError(
                f"{label} normalization columns must be string-like: "
                f"{non_string}"
            )


def normalize_records(
    frame: pl.DataFrame,
    config: LinkConfig,
) -> pl.DataFrame:
    """Return a normalized copy using the linkage configuration."""
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("frame must be a Polars DataFrame")
    if not isinstance(config, LinkConfig):
        raise TypeError("config must be a LinkConfig")
    _validate_normalization_inputs([("input", frame)], config)
    if not config.normalization:
        return frame
    try:
        expressions: list[pl.Expr] = []
        for column, options in config.normalization.items():
            expression = _string_normalization_expr(column, options)
            if options.null_tokens:
                tokens = _normalize_null_tokens(options)
                expression = (
                    pl.when(expression.is_in(tokens))
                    .then(pl.lit(None, dtype=pl.String))
                    .otherwise(expression)
                )
            expressions.append(expression.alias(column))
        return frame.with_columns(expressions)
    except Exception as exc:
        raise ValueError("invalid string normalization rule") from exc


def _string_normalization_expr(
    column: str,
    options: StringNormalization,
) -> pl.Expr:
    expression = pl.col(column).cast(pl.String)
    if options.unicode_form is not None:
        expression = expression.str.normalize(options.unicode_form)
    for pattern, replacement in options.regex_replacements:
        expression = expression.str.replace_all(pattern, replacement)
    if options.trim:
        expression = expression.str.strip_chars()
    if options.collapse_whitespace:
        expression = expression.str.replace_all(r"\s+", " ")
    if options.lowercase:
        expression = expression.str.to_lowercase()
    return expression


def _normalize_null_tokens(options: StringNormalization) -> list[str]:
    token_column = "__normalization_token__"
    normalized = pl.DataFrame(
        {token_column: options.null_tokens},
        schema={token_column: pl.String},
    ).select(_string_normalization_expr(token_column, options))
    return list(dict.fromkeys(normalized[token_column].to_list()))


def _validate_linkage_dtypes(
    frame: pl.DataFrame,
    config: LinkConfig,
    *,
    label: str,
) -> None:
    blocking_columns = {
        column for rule in config.blocking_rules for column in rule
    }
    comparison_columns = {
        comparison.column for comparison in config.comparisons
    }
    object_columns = sorted(
        column
        for column in comparison_columns | blocking_columns
        if frame.schema[column] == pl.Object
    )
    if object_columns:
        raise ValueError(
            f"{label} frame has unsupported Polars Object comparison "
            f"or blocking columns: {object_columns}; cast them to a "
            "concrete scalar dtype before linking"
        )
    if config.backend != "duckdb":
        return

    required = {
        config.unique_id_column,
        *comparison_columns,
        *blocking_columns,
    }
    unsupported_128 = sorted(
        column
        for column in required
        if str(frame.schema[column]) in {"Int128", "UInt128"}
    )
    if unsupported_128:
        raise ValueError(
            "DuckDB linkage does not support 128-bit Polars "
            f"columns: {unsupported_128}"
        )
    float16 = getattr(pl, "Float16", None)
    float16_columns = sorted(
        column for column in required if frame.schema[column] == float16
    )
    if float16_columns:
        raise ValueError(
            "DuckDB linkage does not support Float16 columns: "
            f"{float16_columns}; cast them to Float32 or Float64"
        )
    time_columns = sorted(
        column
        for column in blocking_columns | {config.unique_id_column}
        if frame.schema[column] == pl.Time
    )
    if time_columns:
        raise ValueError(
            "DuckDB linkage does not support Time columns as unique IDs or "
            f"blocking columns: {time_columns}; cast them to String or use "
            "the Polars backend"
        )
    if frame.schema[config.unique_id_column].base_type() == pl.Duration:
        raise ValueError(
            "DuckDB linkage does not support Duration unique IDs; cast them "
            "to Int64 or String, or use the Polars backend"
        )


def _validate_unique_ids(
    df: pl.DataFrame,
    column: str,
    *,
    label: str,
) -> None:
    values = df[column]
    dtype = values.dtype
    if dtype.is_nested() or dtype == pl.Object:
        raise ValueError(f"{label} unique IDs must be scalar values")
    missing = values.null_count()
    if dtype.is_float():
        missing += int(values.is_nan().sum())
        if values.is_infinite().any():
            raise ValueError(f"{label} unique IDs must be finite")
    elif dtype.base_type() in {pl.String, pl.Categorical, pl.Enum}:
        missing += int(values.cast(pl.String).str.strip_chars().eq("").sum())
    if missing:
        raise ValueError(f"{label} unique IDs must not be missing")
    if values.n_unique() != values.len():
        raise ValueError(f"{label} unique IDs contain duplicates")
    try:
        values.sort()
    except Exception as exc:
        raise ValueError(f"{label} unique IDs must be orderable") from exc


def _guard_pair_count(
    pair_count: int,
    config: LinkConfig,
    *,
    estimated: bool = False,
) -> None:
    qualifier = "estimated upper bound" if estimated else "blocking produced"
    if pair_count > config.max_pairs_hard_cap:
        raise MemoryError(
            f"{qualifier} {pair_count:,} pairs; "
            f"hard cap is {config.max_pairs_hard_cap:,}. "
            "Tighten blocking_rules or raise max_pairs_hard_cap."
        )
    if not config.blocking_rules and pair_count > config.max_pairs_warning:
        raise MemoryError(
            f"unblocked linkage {qualifier} {pair_count:,} pairs; "
            f"max_pairs_warning is {config.max_pairs_warning:,}. "
            "Add blocking_rules or explicitly raise max_pairs_warning."
        )
    if estimated and pair_count > config.max_pairs_warning:
        logger.warning(
            "blocking may produce up to %d pairs; consider tighter rules",
            pair_count,
        )


def _build_scored_pairs(
    decorated: pl.DataFrame,
    levels: np.ndarray,
    probabilities: np.ndarray,
    config: LinkConfig,
) -> pl.DataFrame:
    scored = decorated.select(["__id_l__", "__id_r__"]).with_columns(
        pl.Series(
            "match_probability",
            probabilities.astype(np.float64),
        )
    )
    return scored.with_columns(
        *[
            pl.Series(
                f"level__{comparison.column}",
                levels[:, index],
                dtype=pl.UInt8,
            )
            for index, comparison in enumerate(config.comparisons)
        ]
    )


def _empty_linkage_result(
    df: pl.DataFrame,
    pairs: pl.DataFrame,
    config: LinkConfig,
    timings: dict[str, float],
) -> LinkageResult:
    levels = np.empty((0, len(config.comparisons)), dtype=np.uint8)
    scored = _build_scored_pairs(
        pairs,
        levels,
        np.empty(0, dtype=np.float64),
        config,
    )
    clusters: dict[object, int] = {}
    if config.mode == "dedupe":
        ids = df[config.unique_id_column].to_list()
        clusters = cluster_from_pairs(
            ids,
            (),
        )
    return LinkageResult(
        pairs=scored,
        clusters=clusters,
        parameters={
            "lambda": 0.0,
            "threshold": config.match_threshold_probability,
        },
        timings_ms=timings,
    )


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
        ids = df[config.unique_id_column].to_list()
        return cluster_from_pairs(
            ids,
            (),
        )

    ids = df[config.unique_id_column].to_list()
    edges = confident.select(["__id_l__", "__id_r__"]).iter_rows()
    return cluster_from_pairs(ids, edges)


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)
