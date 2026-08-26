"""Fellegi-Sunter parameter learning via Expectation-Maximisation.

Notation used below:
    N  number of candidate pairs
    C  number of comparisons
    L  max number of levels across comparisons (short arrays padded)

    levels[N, C]  integer level assigned to each (pair, comparison)
    m[C, L]       P(level=l for comp c | pair is a true match)
    u[C, L]       P(level=l for comp c | pair is NOT a match)
    lam           prior P(pair is a match) after blocking

The pair-wise probability work is NumPy-vectorised. The M-step uses
one ``numpy.bincount`` call per comparison.
"""

from __future__ import annotations

import logging
from typing import TypedDict, cast

import numpy as np

logger = logging.getLogger(__name__)

_TINY = 1e-12
_PSEUDOCOUNT = 0.5
_ORDER_TOLERANCE = 1e-6
_NEUTRAL_WEIGHT_SPAN = 1e-6
_MIN_PROBABILITY = np.nextafter(np.float32(0), np.float32(1))
_MAX_PROBABILITY = np.nextafter(np.float32(1), np.float32(0))


class _EMFitState(TypedDict):
    converged: bool
    iterations: int
    max_parameter_delta: float
    informative: list[bool]
    smoothing_pseudocount: float


EMParams = TypedDict(
    "EMParams",
    {
        "m": np.ndarray,
        "u": np.ndarray,
        "lambda": float,
        "fit_state": _EMFitState,
    },
)


def estimate_parameters(
    levels: np.ndarray,
    n_levels_per_comp: np.ndarray,
    *,
    prior: float,
    max_iter: int = 25,
    tol: float = 1e-4,
) -> EMParams:
    """Return smoothed ``m``, ``u`` and ``lambda`` via EM.

    Args:
        levels: uint8 array of shape ``(N, C)``.
        n_levels_per_comp: uint8 array of shape ``(C,)`` giving the
            number of valid levels for each comparison.
        prior: starting value of lambda.
        max_iter: hard cap on iterations.
        tol: stop once the max-abs change in m/u drops below this.

    Returns:
        Dict with ``m``, ``u`` (shape ``(C, L)``), ``lambda``, and the
        convergence state used by the linkage safety check.
    """
    n_pairs = levels.shape[0]
    if n_pairs == 0:
        raise ValueError("no candidate pairs supplied to EM")

    max_levels = int(n_levels_per_comp.max())
    informative = np.array(
        [np.unique(column[column != 0]).size >= 2 for column in levels.T],
        dtype=bool,
    )
    # Use float32 for the working probability tables to bound memory.
    m_f64, u_f64 = _initialise(
        levels,
        n_levels_per_comp,
        max_levels,
        informative,
    )
    m = m_f64.astype(np.float32)
    u = u_f64.astype(np.float32)
    lam = float(prior)

    level_mask = _build_level_mask(n_levels_per_comp, max_levels).astype(
        np.float32
    )
    smoothing_mask = level_mask.copy()
    smoothing_mask[:, 0] = 0.0

    prev_m = m.copy()
    prev_u = u.copy()
    prev_lam = lam

    # levels.T is (C, N) — shape needed by take_along_axis over axis=1.
    # keep the transpose once outside the loop and gather from log
    # tables directly to skip the big np.log on the per-pair matrix.
    levels_t = levels.T.astype(np.int64)
    observed = (levels != 0) & informative[None, :]

    converged = False
    iterations = 0
    delta = float("inf")
    for step in range(max_iter):
        responsibilities = _expectation(
            levels_t,
            observed,
            m,
            u,
            lam,
        )
        lam = float(
            (responsibilities.sum(dtype=np.float64) + _PSEUDOCOUNT)
            / (n_pairs + 2 * _PSEUDOCOUNT)
        )
        not_r = 1.0 - responsibilities

        # vector m-step: bincount per comparison is still the cleanest
        # path, and np.bincount is already C-optimised
        for c in np.flatnonzero(informative):
            observed_rows = observed[:, c]
            counts_m = np.bincount(
                levels[observed_rows, c],
                weights=responsibilities[observed_rows],
                minlength=max_levels,
            )
            counts_u = np.bincount(
                levels[observed_rows, c],
                weights=not_r[observed_rows],
                minlength=max_levels,
            )
            counts_m += _PSEUDOCOUNT * smoothing_mask[c]
            counts_u += _PSEUDOCOUNT * smoothing_mask[c]
            m[c, :] = counts_m / (counts_m.sum() + _TINY)
            u[c, :] = counts_u / (counts_u.sum() + _TINY)

        m = _renormalise(m * level_mask)
        u = _renormalise(u * level_mask)

        delta = max(
            float(np.abs(m - prev_m).max()),
            float(np.abs(u - prev_u).max()),
            abs(lam - prev_lam),
        )
        prev_m = m.copy()
        prev_u = u.copy()
        prev_lam = lam
        iterations = step + 1

        logger.debug("em step %d  lambda=%.6f  delta=%.6e", step, lam, delta)
        if delta < tol:
            converged = True
            logger.info(
                "em converged in %d iterations (lambda=%.6f)",
                step + 1,
                lam,
            )
            break
    else:
        logger.info("em stopped at max_iter=%d (lambda=%.6f)", max_iter, lam)

    return cast(
        EMParams,
        {
            "m": m,
            "u": u,
            "lambda": lam,
            "fit_state": _EMFitState(
                converged=converged,
                iterations=iterations,
                max_parameter_delta=delta,
                informative=informative.tolist(),
                smoothing_pseudocount=_PSEUDOCOUNT,
            ),
        },
    )


def _expectation(
    levels_t: np.ndarray,
    observed: np.ndarray,
    m: np.ndarray,
    u: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Return per-pair match responsibilities in log space."""
    log_m = np.log(m + _TINY, dtype=np.float32)
    log_u = np.log(u + _TINY, dtype=np.float32)
    log_m_product = np.where(
        observed,
        np.take_along_axis(log_m, levels_t, axis=1).T,
        0.0,
    ).sum(axis=1)
    log_u_product = np.where(
        observed,
        np.take_along_axis(log_u, levels_t, axis=1).T,
        0.0,
    ).sum(axis=1)
    num = float(np.log(lam + _TINY)) + log_m_product
    den_other = float(np.log(1.0 - lam + _TINY)) + log_u_product
    max_ab = np.maximum(num, den_other)
    log_denom = max_ab + np.log(
        np.exp(num - max_ab, dtype=np.float32)
        + np.exp(den_other - max_ab, dtype=np.float32)
    )
    return cast(
        np.ndarray,
        np.exp(num - log_denom, dtype=np.float32),
    )


def build_fit_diagnostics(
    params: EMParams,
    n_levels_per_comp: np.ndarray,
    comparison_names: list[str],
    *,
    sampled_pair_count: int,
    candidate_pair_count: int,
) -> dict[str, object]:
    """Describe whether learned parameters are safe enough to score."""
    if len(comparison_names) != len(n_levels_per_comp):
        raise ValueError("comparison names do not match learned parameters")

    m = params["m"]
    u = params["u"]
    fit_state = params["fit_state"]
    informative_names: list[str] = []
    neutral_names: list[str] = []
    usable_names: list[str] = []
    inverted_names: list[str] = []

    for index, name in enumerate(comparison_names):
        if not fit_state["informative"][index]:
            neutral_names.append(name)
            continue
        informative_names.append(name)
        level_count = int(n_levels_per_comp[index])
        weights = np.log(
            (m[index, 1:level_count] + _TINY)
            / (u[index, 1:level_count] + _TINY)
        )
        if float(np.ptp(weights)) <= _NEUTRAL_WEIGHT_SPAN:
            neutral_names.append(name)
            continue
        usable_names.append(name)
        if np.any(np.diff(weights) < -_ORDER_TOLERANCE):
            inverted_names.append(name)

    warnings: list[str] = []
    if not fit_state["converged"]:
        warnings.append(
            f"EM did not converge within {fit_state['iterations']} iterations"
        )
    reasons: list[str] = []
    if inverted_names:
        reasons.append(
            "agreement ordering inverted for " + ", ".join(inverted_names)
        )
    if not usable_names:
        reasons.append(
            "no comparison has both varied observed levels and a learned "
            "non-neutral weight"
        )

    status = "rejected" if reasons else "warning" if warnings else "ok"
    return {
        "status": status,
        "reason": "; ".join(reasons) if reasons else None,
        "warnings": warnings,
        "converged": fit_state["converged"],
        "iterations": fit_state["iterations"],
        "max_parameter_delta": fit_state["max_parameter_delta"],
        "candidate_pair_count": candidate_pair_count,
        "sampled_pair_count": sampled_pair_count,
        "smoothing_pseudocount": fit_state["smoothing_pseudocount"],
        "informative_comparisons": informative_names,
        "usable_comparisons": usable_names,
        "neutral_comparisons": neutral_names,
        "inverted_comparisons": inverted_names,
    }


def score_pairs(
    levels: np.ndarray,
    m: np.ndarray,
    u: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Return match probability per candidate pair."""
    levels_t = levels.T.astype(np.int64)
    log_m = np.log(m + _TINY, dtype=np.float32)
    log_u = np.log(u + _TINY, dtype=np.float32)

    # Exactly neutral comparisons carry no evidence. Excluding them also
    # avoids a harmless shared log term changing scores through rounding.
    usable = np.any(m != u, axis=1)
    observed = (levels != 0) & usable[None, :]
    log_m_product = np.where(
        observed,
        np.take_along_axis(log_m, levels_t, axis=1).T,
        0.0,
    ).sum(axis=1)
    log_u_product = np.where(
        observed,
        np.take_along_axis(log_u, levels_t, axis=1).T,
        0.0,
    ).sum(axis=1)

    log_lam = float(np.log(lam + _TINY))
    log_1mlam = float(np.log(1.0 - lam + _TINY))

    num = log_lam + log_m_product
    den_other = log_1mlam + log_u_product
    max_ab = np.maximum(num, den_other)
    log_denom = max_ab + np.log(
        np.exp(num - max_ab, dtype=np.float32)
        + np.exp(den_other - max_ab, dtype=np.float32)
    )
    probabilities = np.exp(num - log_denom, dtype=np.float32)
    return cast(
        np.ndarray,
        np.clip(
            probabilities,
            _MIN_PROBABILITY,
            _MAX_PROBABILITY,
        ),
    )


def _initialise(
    levels: np.ndarray,
    n_levels_per_comp: np.ndarray,
    max_levels: int,
    informative: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Seed m/u with sensible priors.

    m gets most of its mass on the highest level (exact match is
    typical for real matches). u follows the empirical non-null
    frequency over the candidate set.
    """
    n_comps = levels.shape[1]
    m = np.zeros((n_comps, max_levels), dtype=np.float64)
    u = np.zeros((n_comps, max_levels), dtype=np.float64)

    for c in range(n_comps):
        top = int(n_levels_per_comp[c]) - 1
        counts = np.bincount(levels[:, c], minlength=max_levels).astype(
            np.float64
        )
        counts[0] = 0
        if not informative[c]:
            counts[1 : top + 1] += _PSEUDOCOUNT
            m[c, :] = counts
            u[c, :] = counts
            continue

        # m seed: 0.7 on top level, linearly decaying for lower ones
        m[c, top] = 0.7
        m[c, 1:top] = 0.25 / max(top - 1, 1)
        counts[1 : top + 1] += _PSEUDOCOUNT
        u[c, :] = counts

    mask = _build_level_mask(n_levels_per_comp, max_levels)
    m *= mask
    u *= mask
    return _renormalise(m), _renormalise(u)


def _build_level_mask(
    n_levels_per_comp: np.ndarray, max_levels: int
) -> np.ndarray:
    """Boolean-ish mask for levels actually in use per comparison."""
    idx = np.arange(max_levels)[None, :]
    return (idx < n_levels_per_comp[:, None]).astype(np.float64)


def _renormalise(arr: np.ndarray) -> np.ndarray:
    totals = arr.sum(axis=1, keepdims=True) + _TINY
    return cast(np.ndarray, arr / totals)
