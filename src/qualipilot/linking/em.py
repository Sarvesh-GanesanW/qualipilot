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


EMParams = TypedDict(
    "EMParams",
    {"m": np.ndarray, "u": np.ndarray, "lambda": float},
)


def estimate_parameters(
    levels: np.ndarray,
    n_levels_per_comp: np.ndarray,
    *,
    prior: float,
    max_iter: int = 25,
    tol: float = 1e-4,
) -> EMParams:
    """Return learned ``m``, ``u`` and ``lambda`` via EM.

    Args:
        levels: uint8 array of shape ``(N, C)``.
        n_levels_per_comp: uint8 array of shape ``(C,)`` giving the
            number of valid levels for each comparison.
        prior: starting value of lambda.
        max_iter: hard cap on iterations.
        tol: stop once the max-abs change in m/u drops below this.

    Returns:
        Dict with keys ``m``, ``u`` (shape ``(C, L)``) and ``lambda``.
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

    prev_m = m.copy()
    prev_u = u.copy()
    prev_lam = lam

    # levels.T is (C, N) — shape needed by take_along_axis over axis=1.
    # keep the transpose once outside the loop and gather from log
    # tables directly to skip the big np.log on the per-pair matrix.
    levels_t = levels.T.astype(np.int64)
    observed = levels != 0

    for step in range(max_iter):
        log_m = np.log(m + _TINY, dtype=np.float32)
        log_u = np.log(u + _TINY, dtype=np.float32)

        # gather in log-space: N*C table lookups, no per-pair log calls
        log_m_pair = np.where(
            observed,
            np.take_along_axis(log_m, levels_t, axis=1).T,
            0.0,
        )
        log_u_pair = np.where(
            observed,
            np.take_along_axis(log_u, levels_t, axis=1).T,
            0.0,
        )
        log_m_product = log_m_pair.sum(axis=1)
        log_u_product = log_u_pair.sum(axis=1)

        log_lam = float(np.log(lam + _TINY))
        log_1mlam = float(np.log(1.0 - lam + _TINY))

        num = log_lam + log_m_product
        den_other = log_1mlam + log_u_product
        # log-space softmax for numerical stability
        max_ab = np.maximum(num, den_other)
        log_denom = max_ab + np.log(
            np.exp(num - max_ab, dtype=np.float32)
            + np.exp(den_other - max_ab, dtype=np.float32)
        )
        responsibilities = np.exp(num - log_denom, dtype=np.float32)

        lam = float(responsibilities.mean())
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

        logger.debug("em step %d  lambda=%.6f  delta=%.6e", step, lam, delta)
        if delta < tol:
            logger.info(
                "em converged in %d iterations (lambda=%.6f)",
                step + 1,
                lam,
            )
            break
    else:
        logger.info("em stopped at max_iter=%d (lambda=%.6f)", max_iter, lam)

    return cast(EMParams, {"m": m, "u": u, "lambda": lam})


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

    observed = levels != 0
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
    return cast(np.ndarray, np.exp(num - log_denom, dtype=np.float32))


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
            if counts.sum() == 0:
                counts[1 : top + 1] = 1
            m[c, :] = counts
            u[c, :] = counts
            continue

        # m seed: 0.7 on top level, linearly decaying for lower ones
        m[c, top] = 0.7
        m[c, 1:top] = 0.25 / max(top - 1, 1)
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
