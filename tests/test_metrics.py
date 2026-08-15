"""Unit tests for egodiversity.metrics.

Exercises the pure-numpy diversity math (Vendi Score, Hill numbers,
bootstrap CI, permutation test, composite index) against synthetic
data, independent of any real EgoVerse embeddings or metadata.
"""

import numpy as np
import pytest

from egodiversity.metrics import bootstrap_ci, composite_index, hill_number, permutation_test, vendi_score


def test_vendi_score_identical_items_is_one() -> None:
    """Identical embeddings carry no diversity: score collapses to 1."""
    identical = np.tile([1.0, 2.0, 3.0], (10, 1))
    assert vendi_score(identical) == pytest.approx(1.0, abs=1e-6)


def test_vendi_score_orthogonal_items_is_n() -> None:
    """Mutually orthogonal embeddings are maximally diverse: score equals n."""
    orthogonal = np.eye(5)
    assert vendi_score(orthogonal) == pytest.approx(5.0, abs=1e-6)


def test_vendi_score_random_items_between_one_and_n() -> None:
    """A generic random set falls strictly between the two extremes."""
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(20, 16))
    score = vendi_score(embeddings)
    assert 1.0 < score < 20.0


def test_vendi_score_single_item_is_one() -> None:
    """A single-item set has no internal spread to measure."""
    assert vendi_score(np.array([[1.0, 2.0, 3.0]])) == pytest.approx(1.0)


def test_vendi_score_rejects_non_2d_input() -> None:
    """Embeddings must be (n_items, dim); anything else is a usage error."""
    with pytest.raises(ValueError):
        vendi_score(np.array([1.0, 2.0, 3.0]))


def test_hill_number_single_category_is_one() -> None:
    """All items in one category: no coverage diversity."""
    assert hill_number({"fold_clothes": 42}) == pytest.approx(1.0)


def test_hill_number_uniform_categories_equals_category_count() -> None:
    """Even spread across k categories yields an effective count of k."""
    counts = {"a": 10, "b": 10, "c": 10, "d": 10}
    assert hill_number(counts) == pytest.approx(4.0, abs=1e-6)


def test_hill_number_dict_and_array_agree() -> None:
    """Dict-of-counts and raw-count-array inputs must give the same answer."""
    counts_dict = {"a": 3, "b": 1, "c": 6}
    counts_array = np.array([3, 1, 6])
    assert hill_number(counts_dict) == pytest.approx(hill_number(counts_array))


def test_hill_number_is_scale_invariant() -> None:
    """Raw counts and normalized proportions must give the same answer."""
    counts = np.array([4.0, 4.0, 2.0])
    proportions = counts / counts.sum()
    assert hill_number(counts) == pytest.approx(hill_number(proportions))


def test_hill_number_empty_is_zero() -> None:
    """No items observed: diversity is undefined/zero, not an error."""
    assert hill_number({}) == 0.0


@pytest.mark.parametrize("q", [0.0, 0.5, 2.0, 3.0])
def test_hill_number_matches_shannon_at_q_near_one(q: float) -> None:
    """Every Renyi order q agrees at the uniform-distribution extreme."""
    counts = {"a": 5, "b": 5, "c": 5}
    assert hill_number(counts, q=q) == pytest.approx(3.0, abs=1e-6)


def test_bootstrap_ci_brackets_the_point_estimate() -> None:
    """The point estimate should fall within its own resampled interval."""
    rng = np.random.default_rng(1)
    embeddings = rng.normal(size=(30, 8))
    point_estimate = vendi_score(embeddings)

    mean, lower, upper = bootstrap_ci(vendi_score, embeddings, n_boot=200, seed=1)

    assert lower <= mean <= upper
    assert lower - 1.0 <= point_estimate <= upper + 1.0


def test_bootstrap_ci_empty_items_returns_zeros() -> None:
    """No items to resample: report degenerate zeros rather than crashing."""
    assert bootstrap_ci(vendi_score, np.empty((0, 4))) == (0.0, 0.0, 0.0)


def test_permutation_test_detects_a_more_diverse_than_b() -> None:
    """A clearly-more-diverse-than-B setup should yield a small p-value."""
    diverse = np.eye(10)
    collapsed = np.tile([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (10, 1))

    observed_diff, p_value = permutation_test(vendi_score, diverse, collapsed, n_perm=500, seed=2)

    assert observed_diff > 0
    assert p_value < 0.05


def test_permutation_test_p_value_in_valid_range() -> None:
    """p-values must always lie in (0, 1], regardless of the input."""
    rng = np.random.default_rng(3)
    a = rng.normal(size=(15, 6))
    b = rng.normal(size=(15, 6))

    _, p_value = permutation_test(vendi_score, a, b, n_perm=300, seed=3)

    assert 0.0 < p_value <= 1.0


def test_composite_index_all_axes_maximal_is_one() -> None:
    """When every axis hits its theoretical max (n), the index saturates at 1."""
    n = 10
    result = composite_index(
        visual_vs=n,
        motion_vs=n,
        task_hill=n,
        scene_hill=n,
        object_hill=n,
        demographic_hill=n,
        n=n,
    )
    assert result.index == pytest.approx(1.0, abs=1e-6)
    assert result.percent_of_max == pytest.approx(100.0, abs=1e-4)
    assert all(v == pytest.approx(1.0) for v in result.components.values())


def test_composite_index_rewards_balance_over_a_single_spike() -> None:
    """Geometric mean should rank a balanced subset above a spiky one."""
    n = 10
    balanced = composite_index(
        visual_vs=6, motion_vs=6, task_hill=6, scene_hill=6, object_hill=6, demographic_hill=6, n=n
    )
    spiky = composite_index(
        visual_vs=10, motion_vs=1, task_hill=1, scene_hill=1, object_hill=1, demographic_hill=1, n=n
    )
    assert balanced.index > spiky.index


def test_composite_index_rejects_non_positive_n() -> None:
    """n is a normalizer, not a score: it must be positive."""
    with pytest.raises(ValueError):
        composite_index(1, 1, 1, 1, 1, 1, n=0)


def test_composite_index_respects_custom_weights() -> None:
    """Zeroing out every axis but one reduces the index to that axis alone."""
    n = 10
    result = composite_index(
        visual_vs=10,
        motion_vs=1,
        task_hill=1,
        scene_hill=1,
        object_hill=1,
        demographic_hill=1,
        n=n,
        weights={"motion": 0.0, "task": 0.0, "scene": 0.0, "object": 0.0, "demographic": 0.0},
    )
    assert result.index == pytest.approx(1.0, abs=1e-6)
