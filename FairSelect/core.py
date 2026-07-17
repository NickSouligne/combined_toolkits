from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union, Sequence
import numpy as np
import pandas as pd
from .deps import (
    ColumnTransformer, StandardScaler, OneHotEncoder,
    LogisticRegression, MLPClassifier, RandomForestClassifier,
    DecisionTreeClassifier, SVC, XGB_OK, LGBM_OK, XGBClassifier, LGBMClassifier,
    train_test_split,
)
from .utils import (
    group_key, metrics_block, confusion_rates, ece_bin, macro_gaps,
)
from .params import PARAM_SPECS


"""
This file contains the core logic for building and training the models
"""


@dataclass(frozen=True)
class FairSelectPlanSpec:
    """
    Complete, serializable specification for refitting one FairSelect run.

    plan_type
        One of:
            "baseline"
            "single"
            "combined"

    technique
        The selected FairSelect technique key for a single-technique run.
        Must be None for baseline and combined runs.

    selected
        Complete selected-technique mapping for a combined run. Stored as
        a tuple so the specification remains immutable.
    """

    plan_type: str

    model_name: str
    model_params: Dict[str, Any]

    target: str
    protected: Tuple[str, ...]
    features: Tuple[str, ...]

    include_protected_features: bool = False

    technique: Optional[str] = None
    selected: Tuple[Tuple[str, bool], ...] = field(
        default_factory=tuple
    )

    train_index: Tuple[Any, ...] = field(
        default_factory=tuple
    )
    validation_index: Tuple[Any, ...] = field(
        default_factory=tuple
    )
    test_index: Tuple[Any, ...] = field(
        default_factory=tuple
    )

    random_state: int = 42

    def selected_dict(self) -> Dict[str, bool]:
        return dict(self.selected)

@dataclass
class RunResult:
    name: str
    overall: Dict[str, float]
    group_stats: pd.DataFrame
    notes: str = ""
    fairlogue: dict | None = None
    fair_model: object | None = None

    test_index: list | None = None
    y_test: object | None = None
    A_test: object | None = None
    y_prob_test: object | None = None
    y_pred_test: object | None = None

    refit_spec: FairSelectPlanSpec | None = None

#---Core training runners---
def build_estimator(model_name: str, params: Dict[str, Any]):
    '''
    Reads the user inputs to the GUI and constructs the appropriate model object
    '''
def build_estimator(model_name, params=None):
    """
    Build a classifier using a flexible model-name alias.

    Examples
    --------
    Logistic regression:
        "Logistic Regression", "logistic", "logreg", "lr"

    Neural network:
        "Neural Network", "neural", "nn", "mlp"

    Random forest:
        "Random Forest", "random_forest", "rf"

    Decision tree:
        "Decision Tree", "decision_tree", "dt"

    SVM:
        "SVM", "SVC", "support vector machine"

    XGBoost:
        "XGBoost", "xgb"

    LightGBM:
        "LightGBM", "lgbm", "light gbm"
    """
    params = dict(params or {})

    # Normalize capitalization, spaces, hyphens, and underscores.
    normalized_name = (
        str(model_name)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    # Collapse repeated underscores.
    while "__" in normalized_name:
        normalized_name = normalized_name.replace("__", "_")

    model_aliases = {
        # Logistic regression
        "logistic_regression": "logistic_regression",
        "logistic": "logistic_regression",
        "logreg": "logistic_regression",
        "lr": "logistic_regression",

        # Neural network
        "neural_network": "neural_network",
        "neural": "neural_network",
        "nn": "neural_network",
        "mlp": "neural_network",
        "mlpclassifier": "neural_network",

        # Random forest
        "random_forest": "random_forest",
        "randomforest": "random_forest",
        "rf": "random_forest",

        # Decision tree
        "decision_tree": "decision_tree",
        "decisiontree": "decision_tree",
        "dt": "decision_tree",

        # Support vector machine
        "svm": "svm",
        "svc": "svm",
        "support_vector_machine": "svm",
        "support_vector_classifier": "svm",

        # XGBoost
        "xgboost": "xgboost",
        "xgb": "xgboost",
        "xgbclassifier": "xgboost",

        # LightGBM
        "lightgbm": "lightgbm",
        "light_gbm": "lightgbm",
        "lgbm": "lightgbm",
        "lgbmclassifier": "lightgbm",
    }

    canonical_name = model_aliases.get(normalized_name)

    if canonical_name is None:
        supported_aliases = ", ".join(sorted(model_aliases))

        raise ValueError(
            f"Unknown model name: {model_name!r}. "
            f"Supported names and aliases include: {supported_aliases}"
        )

    if canonical_name == "logistic_regression":
        class_weight = params.get("class_weight")

        if isinstance(class_weight, str) and class_weight.lower() == "none":
            class_weight = None

        C_value = params.get("C", 1.0)
        if C_value is None:
            C_value = 1.0

        l1_ratio = params.get("l1_ratio", 0.0)
        if l1_ratio is None:
            l1_ratio = 0.0

        return LogisticRegression(
            C=float(C_value),
            l1_ratio=float(l1_ratio),
            solver=params.get("solver", "lbfgs"),
            max_iter=int(params.get("max_iter", 200) or 200),
            class_weight=class_weight,
            random_state=params.get("random_state"),
        )

    if canonical_name == "neural_network":
        hidden_layer_sizes = params.get("hidden_layer_sizes")

        if hidden_layer_sizes is None:
            hidden_layer_sizes = (100,)
        elif isinstance(hidden_layer_sizes, list):
            hidden_layer_sizes = tuple(hidden_layer_sizes)
        elif isinstance(hidden_layer_sizes, int):
            hidden_layer_sizes = (hidden_layer_sizes,)

        alpha = params.get("alpha", 0.0001)
        if alpha is None:
            alpha = 0.0001

        max_iter = params.get("max_iter", 200)
        if max_iter is None:
            max_iter = 200

        return MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=params.get("activation", "relu"),
            solver=params.get("solver", "adam"),
            alpha=float(alpha),
            learning_rate=params.get(
                "learning_rate",
                "constant",
            ),
            max_iter=int(max_iter),
            random_state=params.get("random_state"),
        )

    if canonical_name == "random_forest":
        max_features = params.get("max_features", "sqrt")

        if (
            isinstance(max_features, str)
            and max_features.lower() == "none"
        ):
            max_features = None

        class_weight = params.get("class_weight")

        if (
            isinstance(class_weight, str)
            and class_weight.lower() == "none"
        ):
            class_weight = None

        return RandomForestClassifier(
            n_estimators=int(
                params.get("n_estimators", 200) or 200
            ),
            max_depth=params.get("max_depth"),
            min_samples_split=int(
                params.get("min_samples_split", 2) or 2
            ),
            min_samples_leaf=int(
                params.get("min_samples_leaf", 1) or 1
            ),
            max_features=max_features,
            class_weight=class_weight,
            random_state=params.get("random_state"),
            n_jobs=params.get("n_jobs", -1),
        )

    if canonical_name == "decision_tree":
        max_features = params.get("max_features")

        if (
            isinstance(max_features, str)
            and max_features.lower() == "none"
        ):
            max_features = None

        return DecisionTreeClassifier(
            criterion=params.get("criterion", "gini"),
            max_depth=params.get("max_depth"),
            min_samples_split=int(
                params.get("min_samples_split", 2) or 2
            ),
            min_samples_leaf=int(
                params.get("min_samples_leaf", 1) or 1
            ),
            max_features=max_features,
            class_weight=(
                None
                if str(
                    params.get("class_weight", "none")
                ).lower() == "none"
                else params.get("class_weight")
            ),
            random_state=params.get("random_state"),
        )

    if canonical_name == "svm":
        C_value = params.get("C", 1.0)
        if C_value is None:
            C_value = 1.0

        degree = params.get("degree", 3)
        if degree is None:
            degree = 3

        class_weight = params.get("class_weight")

        if (
            isinstance(class_weight, str)
            and class_weight.lower() == "none"
        ):
            class_weight = None

        probability = params.get("probability", True)

        # Avoid bool("False") evaluating to True.
        if isinstance(probability, str):
            probability = probability.strip().lower() in {
                "true",
                "1",
                "yes",
                "y",
            }

        return SVC(
            kernel=params.get("kernel", "rbf"),
            C=float(C_value),
            gamma=params.get("gamma", "scale"),
            degree=int(degree),
            probability=bool(probability),
            class_weight=class_weight,
            random_state=params.get("random_state"),
        )

    if canonical_name == "xgboost":
        if not XGB_OK:
            raise ImportError(
                "XGBoost was requested, but xgboost is not installed "
                "or could not be imported."
            )

        return XGBClassifier(
            n_estimators=int(
                params.get("n_estimators", 300) or 300
            ),
            max_depth=int(
                params.get("max_depth", 6)
                if params.get("max_depth", 6) is not None
                else 6
            ),
            learning_rate=float(
                params.get("learning_rate", 0.1)
                if params.get("learning_rate", 0.1) is not None
                else 0.1
            ),
            subsample=float(
                params.get("subsample", 1.0)
                if params.get("subsample", 1.0) is not None
                else 1.0
            ),
            colsample_bytree=float(
                params.get("colsample_bytree", 1.0)
                if params.get("colsample_bytree", 1.0) is not None
                else 1.0
            ),
            reg_alpha=float(
                params.get("reg_alpha", 0.0)
                if params.get("reg_alpha", 0.0) is not None
                else 0.0
            ),
            reg_lambda=float(
                params.get("reg_lambda", 1.0)
                if params.get("reg_lambda", 1.0) is not None
                else 1.0
            ),
            random_state=params.get("random_state"),
            n_jobs=params.get("n_jobs", -1),
            eval_metric=params.get(
                "eval_metric",
                "logloss",
            ),
        )

    if canonical_name == "lightgbm":
        if not LGBM_OK:
            raise ImportError(
                "LightGBM was requested, but lightgbm is not installed "
                "or could not be imported."
            )

        return LGBMClassifier(
            n_estimators=int(
                params.get("n_estimators", 300) or 300
            ),
            max_depth=int(
                params.get("max_depth", -1)
                if params.get("max_depth", -1) is not None
                else -1
            ),
            learning_rate=float(
                params.get("learning_rate", 0.05)
                if params.get("learning_rate", 0.05) is not None
                else 0.05
            ),
            num_leaves=int(
                params.get("num_leaves", 31)
                if params.get("num_leaves", 31) is not None
                else 31
            ),
            subsample=float(
                params.get("subsample", 1.0)
                if params.get("subsample", 1.0) is not None
                else 1.0
            ),
            colsample_bytree=float(
                params.get("colsample_bytree", 1.0)
                if params.get("colsample_bytree", 1.0) is not None
                else 1.0
            ),
            reg_alpha=float(
                params.get("reg_alpha", 0.0)
                if params.get("reg_alpha", 0.0) is not None
                else 0.0
            ),
            reg_lambda=float(
                params.get("reg_lambda", 0.0)
                if params.get("reg_lambda", 0.0) is not None
                else 0.0
            ),
            class_weight=(
                None
                if str(
                    params.get("class_weight", "none")
                ).lower() == "none"
                else params.get("class_weight")
            ),
            random_state=params.get("random_state"),
            n_jobs=params.get("n_jobs", -1),
            verbosity=int(
                params.get("verbosity", -1)
                if params.get("verbosity", -1) is not None
                else -1
            ),
        )

    # This should be unreachable because aliases are validated above.
    raise RuntimeError(
        f"No estimator builder exists for {canonical_name!r}."
    )


def build_preprocessor(X: pd.DataFrame, drop_cols: List[str] | None = None, *, include_protected_features: bool = False) -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer that:
      - standardizes numeric features (z-score via StandardScaler)
      - one-hot encodes categorical features (OneHotEncoder)
      - optionally passes everything through unchanged if no transformers are needed.

    Build preprocessing for the columns already present in X.

    Feature inclusion and exclusion are controlled by run_pipeline().
    The drop_cols argument is retained for backward compatibility.

    Returns
    -------
    ColumnTransformer
        A fitted-structure (not yet fit on data) that defines how to transform
        numeric vs categorical columns.
    """
    feat_cols = list(X.columns)
    X_sub = X[feat_cols]

    num_cols = X_sub.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [column for column in feat_cols if column not in num_cols]

    transformers = []

    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))

    if cat_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            )
        )

    if not transformers:
        transformers.append(("num", "passthrough", feat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def split_data(df: pd.DataFrame, target: str, protected: List[str],
               features: List[str], test_size=0.25, val_size=0.2,
               random_state=42):
    """
    Split a dataset into train, validation, and test sets while preserving:
      - the outcome (target) column
      - protected group labels (possibly intersectional via group_key)
      - specified feature columns

    The split is done in two stages:
      1) Train vs Test
      2) Train vs Validation (within the original train set)

    Both splits are stratified on the target to preserve class balance.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame containing all columns.
    target : str
        Name of the target column (binary outcome).
    protected : List[str]
        List of protected attribute column names used to derive group labels.
    features : List[str]
        List of feature column names used as inputs to the model.
    test_size : float
        Proportion of the full data to allocate to the test set.
    val_size : float
        Proportion of the *training portion* (after test split) to allocate to validation.
    random_state : int
        Seed for reproducible shuffling and splitting.

    Returns
    -------
    X_tr, X_va, X_te : pd.DataFrame
        Feature matrices for train, validation, and test sets.
    y_tr, y_va, y_te : pd.Series
        Target vectors for train, validation, and test sets.
    A_tr, A_va, A_te : pd.Series
        Group labels (possibly intersectional) for train, validation, and test sets.
    """
    #Make a copy to avoid modifying original DataFrame
    df = df.copy()
    #Create the intersectional group labels
    A = group_key(df, protected)

    y = df[target].astype(int) #Extract target column and force to integer
    X = df[features].copy() #Extract feature columns

    #Split into train / test datasets
    X_tr, X_te, y_tr, y_te, A_tr, A_te = train_test_split(
        X, y, A, test_size=test_size, stratify=y, random_state=random_state
    )

    #Split into train / validation datasets
    X_tr, X_va, y_tr, y_va, A_tr, A_va = train_test_split(
        X_tr, y_tr, A_tr, test_size=val_size, stratify=y_tr, random_state=random_state
    )
    return X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te


def evaluate_run(
    name: str,
    y_true,
    p,
    yhat,
    groups,
    *,
    fair_model=None,
    notes: str = "",
    run_fairlogue: bool = True,
    test_index=None,
) -> RunResult:
    """
    Compute overall performance and fairness metrics for a single model run.

    The evaluation proceeds in three layers:
      1) Convert inputs to NumPy arrays for consistent downstream operations.
      2) Compute global (overall) metrics over the entire test set.
      3) Compute per-group metrics (TPR, FPR, etc.), then derive macro gaps
         (e.g., max TPR difference across groups) and merge them into the overall summary.

    Parameters
    ----------
    name : str
        Label for this run (e.g., "Baseline", "Pre: Reweight").
    y_true : array-like
        True binary labels for the test set.
    p : array-like
        Predicted probabilities or scores for the positive class on the test set.
    yhat : array-like
        Hard predictions (0/1) for the test set.
    groups : array-like
        Group labels for each instance in the test set (e.g., intersectional categories).

    Returns
    -------
    RunResult
        Dataclass holding:
        - name: run label
        - overall: dict of scalar metrics (ACC, AUROC, EO_diff, etc.)
        - group_stats: DataFrame of per-group metrics
    """
        # Preserve the held-out labels and group assignments for later auditing
    y_test_saved = (
        y_true.copy()
        if hasattr(y_true, "copy")
        else np.asarray(y_true).copy()
    )

    A_test_saved = (
        groups.copy()
        if hasattr(groups, "copy")
        else np.asarray(groups).copy()
    )

    if test_index is not None:
        test_index_saved = list(test_index)
    elif hasattr(y_true, "index"):
        test_index_saved = list(y_true.index)
    elif hasattr(groups, "index"):
        test_index_saved = list(groups.index)
    else:
        test_index_saved = None

    # Existing conversions
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    yhat = np.asarray(yhat)

    #Ensures the inputs are arrays for reliable downstream processing
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    yhat = np.asarray(yhat)

    #Compute overall model metrics
    overall = metrics_block(y_true, p, yhat)
    rows = [] #List of per-group metrics

    #Iterate over each unique group to compute group-specific metrics
    for g in pd.Series(groups).unique():
        #Create a boolean mask to select all instances belonging to group g
        m = (groups == g).to_numpy()

        #Skip groups with no members
        if m.sum() == 0:
            continue

        #Compute confusion matrix metrics (TPR, FPR, etc.) for group g
        rates = confusion_rates(y_true[m], yhat[m])
        #Compute Expected Calibration Error (ECE) within group g
        rates["ECE"] = ece_bin(y_true[m], p[m], 10)
        #Append the group label and its metrics to the rows list
        rows.append({"group": str(g), **rates})
    
    #Compile group-level metrics into a DataFrame and sort by group label
    group_df = pd.DataFrame(rows).sort_values("group").reset_index(drop=True)

    #Compute fairness disparities across groups and merge into overall metrics
    overall.update(macro_gaps(group_df))

    fairlogue_results = None

    if run_fairlogue:
        from .fairlogue_bridge import run_fairlogue_observational

        fairlogue_results = run_fairlogue_observational(
            y_true=y_true,
            y_prob=p,
            y_pred=yhat,
            groups=groups,
            run_name=name,
        )

    return RunResult(
        name=name,
        overall=overall,
        group_stats=group_df,
        notes=notes,
        fairlogue=fairlogue_results,
        fair_model=fair_model,
        test_index=test_index_saved,
        y_test=y_test_saved,
        A_test=A_test_saved,
        y_prob_test=np.asarray(p).copy(),
        y_pred_test=np.asarray(yhat).copy(),
    )
