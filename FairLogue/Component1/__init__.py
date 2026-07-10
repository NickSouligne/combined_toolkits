# FairLogue/Component1/__init__.py

from .containers import FairnessResults, GroupRates

from .intersectional_metrics import (
    bootstrap_fairness_metrics,
    cross_validate_intersectional_fairness,
    evaluate_intersectional_fairness,
)

from .plots import (
    _plot_bar,
    _plot_bar_series_by_group,
    _plot_fairness_matrix,
    _plot_grouped_eods_components,
)

from .utilities import (
    _as_prob,
    _compute_group_rates,
    _get_model,
    _make_ohe,
    _maybe_balanced,
    confusion_by_group,
    filter_intersectional_groups,
)

__all__ = [
    "GroupRates",
    "FairnessResults",
    "evaluate_intersectional_fairness",
    "cross_validate_intersectional_fairness",
    "bootstrap_fairness_metrics",
    "filter_intersectional_groups",
    "confusion_by_group",
]
