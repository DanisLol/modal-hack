"""Vendi Score and Hill numbers: two instances of one Renyi-diversity core.

Both metrics measure the "effective number of distinct things" in a
population from a probability distribution: for the Vendi Score
(Friedman & Dieng, 2023) that distribution is the eigenvalue spectrum of
a trace-normalized item-similarity kernel; for Hill numbers (ecology's
"effective number of species") it is the frequency distribution over a
categorical variable (task, scene, object, demonstrator, ...). Sharing
one implementation of the underlying order-``q`` Renyi diversity keeps
the two metrics numerically and conceptually consistent.
"""

import numpy as np

_EPS = 1e-12


def _renyi_diversity(probabilities: np.ndarray, q: float) -> float:
    """Compute the order-``q`` Hill/Renyi diversity of a distribution.

    ``probabilities`` must be a 1-D array of non-negative values summing
    to 1 (a probability distribution over "species", whether those are
    eigenvalues of a similarity kernel or categories of metadata). For
    ``q == 1`` this is ``exp(Shannon entropy)``; for other ``q`` it is
    ``(sum(p_i ** q)) ** (1 / (1 - q))``, the standard Hill number
    formula. Zero-probability entries are dropped before the entropy sum
    so they contribute nothing (matching the ``0 * log(0) := 0``
    convention).

    Returns a scalar diversity value in ``[1, len(probabilities)]``: 1
    when all mass sits on one category (no diversity), and the number of
    non-zero categories when mass is spread uniformly (maximum
    diversity).
    """
    p = np.asarray(probabilities, dtype=np.float64)
    p = p[p > _EPS]
    if p.size == 0:
        return 0.0
    p = p / p.sum()

    if np.isclose(q, 1.0):
        shannon_entropy = -np.sum(p * np.log(p))
        return float(np.exp(shannon_entropy))

    return float(np.power(np.sum(np.power(p, q)), 1.0 / (1.0 - q)))


def vendi_score(embeddings: np.ndarray, q: float = 1.0) -> float:
    """Compute the Vendi Score of a set of item embeddings.

    Builds a cosine-similarity kernel ``K`` over ``embeddings`` (shape
    ``(n_items, dim)``), trace-normalizes it to ``K / n_items`` so its
    eigenvalues sum to 1, and returns the order-``q`` Renyi diversity of
    that eigenvalue spectrum: ``VS_q = exp(-sum(lambda_i * log(lambda_i)))``
    for ``q == 1`` (Friedman & Dieng, 2023), or the generalized Hill-number
    form for other ``q``. The result lies in ``[1, n_items]``: 1 if every
    embedding is identical, ``n_items`` if every embedding is mutually
    orthogonal (maximally diverse under the kernel).

    ``embeddings`` rows with zero norm are not supported (cosine
    similarity is undefined); callers should filter or fall back to a
    zero vector's own small epsilon norm upstream.
    """
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"embeddings must be 2-D (n_items, dim), got shape {x.shape}")
    n_items = x.shape[0]
    if n_items == 0:
        return 0.0
    if n_items == 1:
        return 1.0

    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms > _EPS, norms, _EPS)
    unit_x = x / norms

    kernel = unit_x @ unit_x.T
    eigenvalues = np.linalg.eigvalsh(kernel) / n_items
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    return _renyi_diversity(eigenvalues, q)


def hill_number(category_counts: dict | np.ndarray, q: float = 1.0) -> float:
    """Compute the order-``q`` Hill number of a categorical distribution.

    ``category_counts`` is either a mapping from category label to raw
    count/frequency (e.g. ``{"fold_clothes": 40, "pour_water": 12}``) or
    a 1-D array of non-negative counts. Counts are normalized into
    proportions internally, so raw counts, frequencies, or already-
    normalized proportions all give the same result. Used for
    task/scene/object/demonstrator coverage, where the "species" are
    metadata categories rather than kernel eigenvalues.

    Returns the "effective number of categories": 1 if every item falls
    in a single category, and the number of distinct categories present
    if items are spread evenly across them.
    """
    if isinstance(category_counts, dict):
        counts = np.array(list(category_counts.values()), dtype=np.float64)
    else:
        counts = np.asarray(category_counts, dtype=np.float64)

    if counts.size == 0 or counts.sum() <= 0:
        return 0.0

    return _renyi_diversity(counts / counts.sum(), q)
