from __future__ import annotations
from typing import List, Protocol, Any, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve



#---- Utilities & type helpers ----

def _as_str_groups(s: pd.Series) -> pd.Series:
    """
    Return group labels as strings (stable downstream ops).
    Ensures consistent dtype for joins/indexing. See docs/helpers.md#_as_str_groups
    """
    return s.astype(str)

def _clip_probs(arr: np.ndarray, lo: float = 1e-6, hi: float = 1 - 1e-6) -> np.ndarray:
    """
    Numerically clip probabilities to (lo, hi) for safety.
    Guards against divide-by-zero and log(0). See docs/helpers.md#_clip_probs
    """
    return np.clip(arr, lo, hi)


def choose_threshold_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Select global τ via Youden’s J (TPR − FPR) on pooled OOF preds.
    Used as a single operating point across groups. See docs/helpers.md#choose_threshold_youden
    """
    fpr, tpr, thresh = roc_curve(y_true.astype(int), y_prob)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thresh[idx])


def _add_group_dummies(
    X: pd.DataFrame,
    group_vals: pd.Series,
    groups: List[str],
    prefix: str = "G__"
) -> pd.DataFrame:
    """
    Add fixed-order one-hot columns for group labels to X.
    Uses an explicit groups list for fold-stable columns. See docs/helpers.md#_add_group_dummies
    """
    X = X.copy()
    cols = [f"{prefix}{g}" for g in groups]
    dummies = pd.DataFrame(0, index=X.index, columns=cols, dtype=int)
    g_str = _as_str_groups(group_vals)
    for g in groups:
        dummies.loc[g_str == g, f"{prefix}{g}"] = 1
    return pd.concat([X, dummies], axis=1)

def _init_group_dummy_frame(
    index,
    groups: List[str],
    prefix: str = "G__",
    dtype=np.uint8,
):
    cols = [f"{prefix}{g}" for g in groups]
    return pd.DataFrame(0, index=index, columns=cols, dtype=dtype)


#---- Estimator Protocols ----

class ProbaEstimator(Protocol):
    """
    Protocol for estimators used in outcome modeling.
    Must implement fit(X,y) and predict_proba(X)->(n,2) for binary tasks.
    See docs/helpers.md#ProbaEstimator
    """
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Any: ...
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray: ...


def make_outcome_estimator(model_type: str, random_state: int = 42) -> ProbaEstimator:
    """
    Factory for baseline outcome models ('rf' or 'glm').
    Returns a calibrated, class-weighted classifier. See docs/helpers.md#make_outcome_estimator
    """
    model_type = model_type.lower()
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=700,
            max_depth=3,
            random_state=random_state,
            class_weight="balanced",
        )
    if model_type == "glm":
        #LogisticRegression = GLM logit with L2; stable & fast
        return LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=random_state,
        )
    raise ValueError("Unknown estimator kind; use 'rf' or 'glm'.")