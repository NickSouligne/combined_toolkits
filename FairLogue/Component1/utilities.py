from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from .containers import GroupRates


#Helper function to create OneHotEncoder with sparse or dense output
def _make_ohe(dense: bool = False) -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=not dense)

def _maybe_balanced(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    p = dict(params or {})
    p.setdefault("class_weight", "balanced")
    return p

#Helper function to get model based on string input and parameters
#Currently supports: logistic regression, random forest, decision tree, lightgbm, neural network
def _get_model(model_type: str, model_params: Optional[Dict[str, Any]]):
    mt = model_type.lower()
    params = model_params or {}
    if mt in ("logreg", "logistic", "lr"):
        return LogisticRegression(max_iter = params.pop("max_iter", 1000), **_maybe_balanced(params))
    if mt in ("lr_cv", "log_cv", "logreg_cv", "logreg_crossval"):
        return LogisticRegressionCV(max_iter = params.pop("max_iter", 1000), **_maybe_balanced(params))
    if mt in ("rf", "random_forest", "randomforest"):
        return RandomForestClassifier(**_maybe_balanced(params))
    if mt in ("dt", "decision_tree", "decisiontree"):
        return DecisionTreeClassifier(**_maybe_balanced(params))
    if mt in ("nn", "mlp", "neural", "neural_network"):
        default_params = dict(hidden_layer_sizes=(64, 32), activation="relu", max_iter=300)
        default_params.update(params)
        return MLPClassifier(**default_params)
    if mt in ("lgbm", "lightgbm"):
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params)
    raise ValueError("Unsupported model_type: use one of: logreg, logreg_cv, rf, dt, lgbm, nn")

#Helper function to get probabilities from a fitted model
def _as_prob(estimator, X):
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        return proba[:, 1] if proba.shape[1] == 2 else proba.max(axis=1)
    if hasattr(estimator, "decision_function"):
        df = estimator.decision_function(X).astype(float)
        return (df - df.min()) / (df.max() - df.min() + 1e-12)
    return estimator.predict(X).astype(float)

#Helper function to filter small intersectional groups from dataframe (default: min size 20)
def filter_intersectional_groups(
    df,
    target: str,
    protected_1: str,
    protected_2: str,
    *,
    positive_label=1,
    min_group_size: int = 20
):
    """
    Returns: df_filtered
    Drops groups with total count < min_group_size.
    """
    g = df[protected_1].astype(str) + "|" + df[protected_2].astype(str)
    counts = g.value_counts()
    keep_groups = counts[counts >= min_group_size].index
    df_filtered = df[g.isin(keep_groups)].copy()
    return df_filtered

#Helper function to compute confusion matrix components and TP/FP rates by group
def confusion_by_group(y_true, y_pred, groups):
    df = pd.DataFrame({"y": y_true, "yhat": y_pred, "group": groups})
    def agg(g):
        y = g["y"].values
        yh = g["yhat"].values
        n = len(y)
        pos = int((y == 1).sum()); neg = n - pos
        tp = int(((y == 1) & (yh == 1)).sum())
        fp = int(((y == 0) & (yh == 1)).sum())
        tn = int(((y == 0) & (yh == 0)).sum())
        fn = int(((y == 1) & (yh == 0)).sum())
        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        p_hat = (yh == 1).mean()
        return pd.Series({
            "n_test": n, "pos_test": pos, "neg_test": neg,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "TPR": tpr, "FPR": fpr, "P_hat": p_hat
        })
    return df.groupby("group", sort=False).apply(agg).reset_index()

#Helper function to compute group rates utilizing GroupRates dataclass
def _compute_group_rates(y_true, y_pred, groups: pd.Series) -> List[GroupRates]:
    out: List[GroupRates] = []
    for g, idx in groups.groupby(groups).groups.items():
        idx = np.asarray(list(idx))
        y_g = y_true[idx]
        yhat_g = y_pred[idx]

        #confusion components
        if np.unique(y_g).size == 1:
            #handle single-class edge case explicitly
            tn = fp = fn = tp = 0
            if y_g[0] == 1:
                tp = int((yhat_g == 1).sum()); fn = int((yhat_g == 0).sum())
            else:
                tn = int((yhat_g == 0).sum()); fp = int((yhat_g == 1).sum())
        else:
            tn, fp, fn, tp = confusion_matrix(y_g, yhat_g, labels=[0, 1]).ravel()

        n = len(y_g)
        pos_true = int((y_g == 1).sum())
        neg_true = n - pos_true
        positive_rate = float((yhat_g == 1).mean()) if n > 0 else 0.0
        tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else np.nan

        out.append(
            GroupRates(
                group=str(g), n=n, positive_rate=positive_rate, tpr=tpr, fpr=fpr,
                pos_true=pos_true, neg_true=neg_true, TP=tp, FP=fp, TN=tn, FN=fn
            )
        )
    return out
