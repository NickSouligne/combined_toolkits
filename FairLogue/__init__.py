# FairLogue/__init__.py

"""
FairLogue: intersectional fairness evaluation tools.
"""

from .Component1 import (
    FairnessResults,
    GroupRates,
    cross_validate_intersectional_fairness,
    evaluate_intersectional_fairness,
)

from .Component3 import FairnessPipeline

__all__ = [
    "FairnessResults",
    "GroupRates",
    "cross_validate_intersectional_fairness",
    "evaluate_intersectional_fairness",
    "FairnessPipeline",
]

__version__ = "0.1.0"
