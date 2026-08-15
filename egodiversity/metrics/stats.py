"""Statistical confidence for diversity comparisons.

Wraps any scalar diversity metric (``vendi_score``, ``hill_number``, or
``composite_index``) with bootstrap confidence intervals and a
permutation test, so a subset comparison reports a confidence interval
and a p-value rather than a single unqualified number.
"""

from typing import Callable

import numpy as np

ScoreFn = Callable[[np.ndarray], float]


def bootstrap_ci(
    score_fn: ScoreFn,
    items: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Estimate a bootstrap confidence interval for ``score_fn(items)``.

    Resamples ``items`` with replacement ``n_boot`` times (rows for a 2-D
    embedding array, elements for a 1-D array), recomputes ``score_fn``
    on each resample, and returns ``(mean, lower, upper)`` where
    ``lower``/``upper`` are the ``(1 - ci) / 2`` and
    ``1 - (1 - ci) / 2`` percentiles of the bootstrap distribution.

    ``seed`` makes the resampling reproducible; pass ``None`` for
    non-deterministic sampling.
    """
    x = np.asarray(items)
    n_items = x.shape[0]
    if n_items == 0:
        return 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)
    boot_scores = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        resample_idx = rng.integers(0, n_items, size=n_items)
        boot_scores[i] = score_fn(x[resample_idx])

    alpha = (1.0 - ci) / 2.0
    lower, upper = np.quantile(boot_scores, [alpha, 1.0 - alpha])
    return float(boot_scores.mean()), float(lower), float(upper)


def permutation_test(
    score_fn: ScoreFn,
    a_items: np.ndarray,
    b_items: np.ndarray,
    n_perm: int = 2000,
    seed: int | None = None,
) -> tuple[float, float]:
    """One-sided permutation test for "subset A is more diverse than B".

    Pools ``a_items`` and ``b_items``, then ``n_perm`` times shuffles the
    pooled items and re-splits them into groups of the original sizes
    (``len(a_items)``, ``len(b_items)``) to build a null distribution of
    ``score_fn(A') - score_fn(B')`` under the assumption that group
    membership is uninformative. Returns
    ``(observed_diff, p_value)`` where ``observed_diff`` is
    ``score_fn(a_items) - score_fn(b_items)`` and ``p_value`` is the
    fraction of permuted diffs at least as large as ``observed_diff``
    (Laplace-smoothed with a ``+1`` correction so ``p_value`` is never
    exactly 0), i.e. the probability of seeing this large a gap in favor
    of A by chance alone if A and B were equally diverse.
    """
    a = np.asarray(a_items)
    b = np.asarray(b_items)
    n_a, n_b = a.shape[0], b.shape[0]
    pooled = np.concatenate([a, b], axis=0)

    observed_diff = score_fn(a) - score_fn(b)

    rng = np.random.default_rng(seed)
    permuted_diffs = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        shuffled = pooled[rng.permutation(n_a + n_b)]
        permuted_diffs[i] = score_fn(shuffled[:n_a]) - score_fn(shuffled[n_a:])

    at_least_as_extreme = np.sum(permuted_diffs >= observed_diff)
    p_value = (at_least_as_extreme + 1) / (n_perm + 1)
    return float(observed_diff), float(p_value)
