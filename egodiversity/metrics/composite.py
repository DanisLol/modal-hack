"""Composite EgoDiversity Index: combine the six per-axis diversity scores.

Visual Vendi Score, motion Vendi Score, and the four metadata Hill
numbers (task/scene/object/demonstrator) are each measured on their own
scale with their own theoretical maximum (the sample size ``n``, since
both a Vendi Score and a Hill number over ``n`` items saturate at
``n``). Normalizing every axis by ``n`` before combining makes the
composite comparable across subsets of different sizes, and reporting
each normalized axis alongside the composite keeps the score
explainable instead of a single opaque number.
"""

import math
from dataclasses import dataclass, field

_EPS = 1e-12

DEFAULT_WEIGHTS: dict[str, float] = {
    "visual": 1.0,
    "motion": 1.0,
    "task": 1.0,
    "scene": 1.0,
    "object": 1.0,
    "demographic": 1.0,
}


@dataclass
class CompositeIndexResult:
    """Aggregate EgoDiversity Index plus its per-axis breakdown.

    ``index`` is the weighted geometric mean of the six axis fractions,
    in ``[0, 1]``. ``percent_of_max`` is the same value scaled to a
    percentage for display ("% of maximum attainable diversity").
    ``components`` holds each axis's own fraction of its theoretical
    max, before combining, for the dashboard's axis-breakdown chart.
    """

    index: float
    percent_of_max: float
    components: dict[str, float] = field(default_factory=dict)


def composite_index(
    visual_vs: float,
    motion_vs: float,
    task_hill: float,
    scene_hill: float,
    object_hill: float,
    demographic_hill: float,
    n: int,
    weights: dict[str, float] | None = None,
) -> CompositeIndexResult:
    """Combine six per-axis diversity scores into one EgoDiversity Index.

    Each raw score (two Vendi Scores, four Hill numbers) is divided by
    ``n`` (the number of items the score was computed over) to get a
    fraction of its theoretical maximum, then the fractions are combined
    via a weighted geometric mean: ``exp(sum(w_i * log(f_i)) / sum(w_i))``.
    The geometric mean (rather than an arithmetic mean) means a subset
    that is diverse on every axis outranks one that is very diverse on
    one axis and collapsed on another, matching the intent of measuring
    diversity across all three axes at once rather than letting one
    dominate.

    ``weights`` maps axis name (``"visual"``, ``"motion"``, ``"task"``,
    ``"scene"``, ``"object"``, ``"demographic"``) to a non-negative
    weight; omitted axes fall back to ``DEFAULT_WEIGHTS`` (equal
    weighting). Raises ``ValueError`` if ``n <= 0``.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    raw_scores = {
        "visual": visual_vs,
        "motion": motion_vs,
        "task": task_hill,
        "scene": scene_hill,
        "object": object_hill,
        "demographic": demographic_hill,
    }
    merged_weights = {**DEFAULT_WEIGHTS, **(weights or {})}

    components = {axis: min(max(score / n, 0.0), 1.0) for axis, score in raw_scores.items()}

    total_weight = sum(merged_weights[axis] for axis in raw_scores)
    if total_weight <= 0:
        raise ValueError("sum of weights must be positive")

    weighted_log_sum = sum(
        merged_weights[axis] * math.log(max(components[axis], _EPS)) for axis in raw_scores
    )
    index = math.exp(weighted_log_sum / total_weight)

    return CompositeIndexResult(index=index, percent_of_max=index * 100.0, components=components)
