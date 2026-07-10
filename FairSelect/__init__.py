"""
Convenience package exports for the `code` package.

This module re-exports the top-level functions, classes and constants
defined across the package modules so users can import them directly
from `code` (e.g. `from code import build_estimator`).
"""

# Core
from .core import (
    RunResult,
    build_estimator,
    build_preprocessor,
    split_data,
    evaluate_run,
)

# Dependencies / flags
from .deps import (
    AIF360_OK,
    SKLEARN_OK,
    IMBLEARN_OK,
    FAIRLEARN_OK,
    XGB_OK,
    LGBM_OK,
)

# GUI / entrypoint
from .gui import FairnessToolGUI
from .main import main


# Params
from .params import PARAM_SPECS, AVAILABLE_MODELS, _p

# Techniques: combined, in, post
from .techniques_combined import run_combined_pipeline
from .techniques_in import (
    run_compositional_models,
    run_prejudice_remover,
    run_group_balanced_ensemble,
    run_multicalibration,
    run_reductions_meta,
    run_baseline,
    fit_isotonic_by_group,
    apply_isotonic_by_group,
)
from .techniques_post import (
    group_thresholds_youden,
    predict_with_group_thresholds,
    run_group_youden_postproc,
    run_multiaccuracy_boost,
    run_reject_option_shift,
    run_input_repair,
)

# Utilities
from .utils import (
    estimator_accepts_sample_weight,
    fit_with_optional_sample_weight,
    _fmt,
    _fmt_delta,
    coerce_value,
    eval_tuple,
    to_proba,
    ece_bin,
    group_key,
    safe_auroc,
    safe_auprc,
    youden_threshold,
    confusion_rates,
    metrics_block,
    macro_gaps,
    group_balanced_bootstrap_indices,
    input_repair_standardize_by_group,
)

# Runner
from .runner import (
    run_pipeline,
    PipelineConfig,
    _normalize_features,
    _load_df,
    _selected_dict,
)

__all__ = [
    # core
    "RunResult",
    "build_estimator",
    "build_preprocessor",
    "split_data",
    "evaluate_run",

    # deps
    "AIF360_OK",
    "SKLEARN_OK",
    "IMBLEARN_OK",
    "FAIRLEARN_OK",
    "XGB_OK",
    "LGBM_OK",

    # gui / entry
    "FairnessToolGUI",
    "main",

    # params
    "PARAM_SPECS",
    "AVAILABLE_MODELS",
    "_p",

    # techniques
    "run_combined_pipeline",
    "run_compositional_models",
    "run_prejudice_remover",
    "run_group_balanced_ensemble",
    "run_multicalibration",
    "run_reductions_meta",
    "run_baseline",
    "fit_isotonic_by_group",
    "apply_isotonic_by_group",
    "group_thresholds_youden",
    "predict_with_group_thresholds",
    "run_group_youden_postproc",
    "run_multiaccuracy_boost",
    "run_reject_option_shift",
    "run_input_repair",

    # utils
    "estimator_accepts_sample_weight",
    "fit_with_optional_sample_weight",
    "_fmt",
    "_fmt_delta",
    "coerce_value",
    "eval_tuple",
    "to_proba",
    "ece_bin",
    "group_key",
    "safe_auroc",
    "safe_auprc",
    "youden_threshold",
    "confusion_rates",
    "metrics_block",
    "macro_gaps",
    "group_balanced_bootstrap_indices",
    "input_repair_standardize_by_group",

    #runner
    "run_pipeline",
    "PipelineConfig",
    "_normalize_features",
    "_load_df",
    "_selected_dict",


]
