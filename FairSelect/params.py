from typing import Dict, List, Any
from .deps import SKLEARN_OK, XGB_OK, LGBM_OK


"""
This file contains the parameter specifications for various machine learning models.
"""

PARAM_SPECS: Dict[str, List[Dict[str, Any]]] = {}


def _p(name, ptype, required=False, default=None, help="", choices=None):
    """Helper to define parameter spec dict."""
    return {
        "name": name,
        "type": ptype,
        "required": required,
        "default": default,
        "help": help,
        "choices": choices or [],
    }

#Logistic Regression
PARAM_SPECS["Logistic Regression"] = [
    _p("penalty", "choice", False, "l2", "Regularization penalty.", ["l1", "l2", "elasticnet", "none"]),
    _p("C", float, False, 1.0, "Inverse reg strength."),
    _p("solver", "choice", False, "lbfgs", "Optimizer.", ["lbfgs", "liblinear", "saga", "newton-cg", "newton-cholesky"]),
    _p("max_iter", int, False, 200, "Max iterations."),
    _p("class_weight", "choice", False, "None", "Class weights.", ["None", "balanced"]),
    _p("random_state", int, False, None, "Random seed."),
]

#Neural Network
PARAM_SPECS["Neural Network"] = [
    _p("hidden_layer_sizes", str, False, "(100,)", "Tuple-like string, e.g. '(128,64)'."),
    _p("activation", "choice", False, "relu", "Activation.", ["identity", "logistic", "tanh", "relu"]),
    _p("solver", "choice", False, "adam", "Optimizer.", ["lbfgs", "sgd", "adam"]),
    _p("alpha", float, False, 0.0001, "L2 term."),
    _p("learning_rate", "choice", False, "constant", "LR schedule.", ["constant", "invscaling", "adaptive"]),
    _p("max_iter", int, False, 200, "Max iterations."),
    _p("random_state", int, False, None, "Random seed."),
]

#Random Forest
PARAM_SPECS["Random Forest"] = [
    _p("n_estimators", int, False, 200, "Trees."),
    _p("max_depth", int, False, None, "Max depth."),
    _p("min_samples_split", int, False, 2, "Min split."),
    _p("min_samples_leaf", int, False, 1, "Min leaf."),
    _p("max_features", "choice", False, "sqrt", "Features/split.", ["sqrt", "log2", "None"]),
    _p("class_weight", "choice", False, "None", "Weights.", ["None", "balanced", "balanced_subsample"]),
    _p("random_state", int, False, None, "Random seed."),
]

#Decision Tree
PARAM_SPECS["Decision Tree"] = [
    _p("criterion", "choice", False, "gini", "Criterion.", ["gini", "entropy", "log_loss"]),
    _p("max_depth", int, False, None, "Max depth."),
    _p("min_samples_split", int, False, 2, "Min split."),
    _p("min_samples_leaf", int, False, 1, "Min leaf."),
    _p("max_features", "choice", False, "None", "Features.", ["None", "sqrt", "log2"]),
    _p("random_state", int, False, None, "Random seed."),
]

#SVM
PARAM_SPECS["SVM"] = [
    _p("kernel", "choice", False, "rbf", "Kernel.", ["linear", "poly", "rbf", "sigmoid"]),
    _p("C", float, False, 1.0, "C parameter."),
    _p("gamma", "choice", False, "scale", "Gamma.", ["scale", "auto"]),
    _p("degree", int, False, 3, "Poly degree."),
    _p("probability", bool, False, True, "Enable probability."),
]

from .deps import XGB_OK, LGBM_OK

if XGB_OK:
    PARAM_SPECS["XGBoost"] = [
        _p("n_estimators", int, False, 300, "Rounds."),
        _p("max_depth", int, False, 6, "Depth."),
        _p("learning_rate", float, False, 0.1, "Eta."),
        _p("subsample", float, False, 1.0, "Row sample."),
        _p("colsample_bytree", float, False, 1.0, "Feature sample."),
        _p("reg_alpha", float, False, 0.0, "L1."),
        _p("reg_lambda", float, False, 1.0, "L2."),
        _p("random_state", int, False, None, "Seed."),
    ]

if LGBM_OK:
    PARAM_SPECS["LightGBM"] = [
        _p("n_estimators", int, False, 300, "Rounds."),
        _p("max_depth", int, False, -1, "Depth (-1 none)."),
        _p("learning_rate", float, False, 0.05, "LR."),
        _p("num_leaves", int, False, 31, "Leaves."),
        _p("subsample", float, False, 1.0, "Bagging."),
        _p("colsample_bytree", float, False, 1.0, "Feature frac."),
        _p("reg_alpha", float, False, 0.0, "L1."),
        _p("reg_lambda", float, False, 0.0, "L2."),
        _p("random_state", int, False, None, "Seed."),
    ]

AVAILABLE_MODELS: List[str] = []
if SKLEARN_OK:
    AVAILABLE_MODELS += ["Logistic Regression", "Neural Network", "Random Forest", "Decision Tree", "SVM"]
if XGB_OK:
    AVAILABLE_MODELS.append("XGBoost")
if LGBM_OK:
    AVAILABLE_MODELS.append("LightGBM")
