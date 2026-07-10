# src/iftoolkit/component2/__init__.py
from .outcome_models import build_outcome_models_and_scores
from .estimation_functions import get_bs_rescaled, get_defs_analysis, get_defs_from_rates
from .Fairness import FairnessPipeline
from .helpers import _as_str_groups, make_outcome_estimator, _add_group_dummies, ProbaEstimator, choose_threshold_youden, _clip_probs
from .model import ensure_probabilistic_estimator, Model
from .plots import annotate_plot, get_plots

__all__ = [
    "build_outcome_models_and_scores",
    "get_bs_rescaled",
    "get_defs_analysis",
    "get_defs_from_rates",
    "FairnessPipeline",
    "_as_str_groups",
    "make_outcome_estimator",
    "_add_group_dummies",
    "ProbaEstimator",
    "choose_threshold_youden",
    "_clip_probs",
    "ensure_probabilistic_estimator",
    "Model",
    "annotate_plot",
    "get_plots",
]