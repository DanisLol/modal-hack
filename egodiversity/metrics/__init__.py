"""Reference-free diversity metrics for EgoDiversity.

Pure-numpy implementations that run identically inside a Modal function
or in a local Python process: no learned judge, no network calls. See
``PLAN.md`` section 4 for the mathematical definitions this module
implements.
"""

from egodiversity.metrics.composite import composite_index
from egodiversity.metrics.diversity import hill_number, vendi_score
from egodiversity.metrics.stats import bootstrap_ci, permutation_test

__all__ = [
    "bootstrap_ci",
    "composite_index",
    "hill_number",
    "permutation_test",
    "vendi_score",
]
