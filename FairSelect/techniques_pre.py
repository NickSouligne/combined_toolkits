import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .deps import IMBLEARN_OK
from .core import RunResult, build_estimator, build_preprocessor, evaluate_run
from .utils import to_proba, fit_with_optional_sample_weight, youden_threshold
from .FairModel_helper import make_standard_fair_model
from FairModel import FairModel

def compute_reweights(y: pd.Series, a: pd.Series) -> np.ndarray:
    """
    Compute Kamiran–Calders style reweighting factors based on (y, a).

    Intuition:
      - We want to adjust for imbalances between outcome y and group a.
      - For each (y_i, a_i), we compute a weight:
            w(y_i, a_i) ~= P(y_i) * P(a_i) / P(y_i, a_i)
        so that, under the reweighted distribution, y and a are rendered
        (approximately) independent.

    Parameters
    ----------
    y : pd.Series
        Binary labels for each training example (e.g., 0/1 outcomes).
    a : pd.Series
        Protected/group attribute for each example (can be categorical or
        intersectional labels).

    Returns
    -------
    np.ndarray
        Array of sample weights, one per row in (y, a), to be passed as
        sample_weight when fitting a classifier.
    """
    #Combine y and a into a single dataframe for joint and marginal distributions
    df = pd.DataFrame({"y": y, "a": a})

    #P(y): marginal distribution of the label y (normalized frequencies)
    py = df["y"].value_counts(normalize=True)
    #P(a): marginal distribution of the group attribute a
    pa = df["a"].value_counts(normalize=True)
    #P(y,a): joint distribution of (y, a)
    pya = df.value_counts(normalize=True)
    weights = [] #Computed weights

    #iterate over rows to compute weights (y_i, a_i) = (label, group)
    for yi, ai in zip(df["y"], df["a"]):
        #Reweighting formula from docstring
        w = (py.get(yi, 0) * pa.get(ai, 0)) / max(pya.get((yi, ai), 1e-12), 1e-12)
        weights.append(w)
    return np.asarray(weights, dtype=float)



def local_massaging_fit_flip(y_train: pd.Series, scores: np.ndarray, a_train: pd.Series) -> np.ndarray:
    """
    Perform local label massaging to equalize positive rates across groups.

    High-level idea:
      - Compute the overall positive rate p+ across all training data.
      - For each group g:
          * Compute its current positive rate p_g.
          * If p_g < p+, we need more positives in that group:
                - Flip some negatives (y=0) to positives (y=1).
          * If p_g > p+, we need fewer positives:
                - Flip some positives (y=1) to negatives (y=0).
          * To avoid drastic changes, we only flip those instances
            whose predicted probability (score) is closest to 0.5, i.e.,
            those "closest to the decision boundary" and most ambiguous.
      - Continue until each group's positive rate is approximately p+.

    Parameters
    ----------
    y_train : pd.Series
        Original training labels (0/1).
    scores : np.ndarray
        Predicted probabilities for the training data from a temporary model
        (used to identify which points are closest to the boundary).
    a_train : pd.Series
        Group labels for the training data.

    Returns
    -------
    np.ndarray
        New label array after massaging (flips applied), same length as y_train.
    """
    #Create local copy to avoid modifying original labels
    y = y_train.copy().to_numpy().astype(int)
    a = a_train.to_numpy() #Convert labels to numpy arrays

    
    p_overall = y.mean() #Overall positive rate across all training examples
    groups = pd.Series(a).unique() #Unique groups 
    d = np.abs(scores - 0.5) #distance to boundary

    # Iterate over each group to adjust positive rates
    for g in groups:
        #Boolean mask for current group
        idx = np.where(a==g)[0]
        #Skip groups with no members
        if len(idx)==0: continue
        #Current positive rate for this group
        grp_rate = y[idx].mean()
        target = p_overall #Ideal positive rate
 
        #If already at target, skip
        if math.isclose(grp_rate, target, abs_tol=1e-6): continue

        #Compute number of label flips needed
        need = int(round(target*len(idx) - y[idx].sum()))
        if need>0:
            #We need to *increase* the number of positives in this group
            cand = idx[(y[idx]==0)] #candidates: negatives in group
            order = cand[np.argsort(d[cand])] #sort by distance to boundary
            flip = order[:need] #select top 'need' to flip
            y[flip] = 1 #Apply flips
        elif need<0:
            #We need to *decrease* the number of positives (too many positives)
            cand = idx[(y[idx]==1)] #candidates: positives in group
            order = cand[np.argsort(d[cand])] #sort by distance to boundary
            flip = order[:abs(need)] #select top 'need' to flip
            y[flip] = 0 #Apply flips
    return y


def run_reweighting(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected_cols, all_df_train):
    """
    Pre-processing: reweighting by (y, a) via compute_reweights.

    Workflow:
      1. Build a shared preprocessor on train+val and fit on that union.
      2. Transform X_tr through the preprocessor.
      3. Compute sample weights using (y_tr, A_tr) via compute_reweights.
      4. Fit the chosen estimator using fit_with_optional_sample_weight so that
         if the estimator doesn't support sample_weight, a fallback is used.
      5. Evaluate on X_te with the trained model.

    This method aims to decorrelate labels y and group A in the weighted
    training distribution.
    """
    #Build and fit preprocessor on train+val
    prep = build_preprocessor(pd.concat([X_tr,X_va]), protected_cols)
    Xt = prep.fit_transform(X_tr)
    #Build base estimator
    clf = build_estimator(model_name, params)
    #Compute reweighting factors
    w = compute_reweights(y_tr, A_tr)
    #Fit the base estimator using reweights as the sample weight
    fit_with_optional_sample_weight(clf, Xt, y_tr, sample_weight=w)
    #Compute predicted probabilities on the test set
    p = to_proba(clf, prep.transform(X_te))
    yhat = (p >= 0.5).astype(int) #Hard predictions at 0.5 threshold
    fair_model = make_standard_fair_model(
        name="Pre: Reweight (y,a)",
        features=X_tr.columns,
        protected_cols=protected_cols,
        preprocessor=prep,
        estimator=clf,
        threshold=0.5,
        outcome_col=None,
        metadata={
            "source": "FairSelect",
            "technique": "Pre:Reweight (y,a)",
            "model_name": model_name,
            "model_params": params,
            "uses_sample_weight": True,
            },
        )

    return evaluate_run(
        "Pre: Reweight (y,a)",
        y_te.to_numpy(),
        p,
        yhat,
        A_te,
        fair_model=fair_model,
        test_index=X_te.index,
    )

def run_smote_or_ros(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected_cols, all_df_train):
    """
    Pre-processing: class-balancing via SMOTE or oversampling.

    Workflow:
      - Always:
          1. Fit a shared preprocessor on train+val and transform X_tr.
          2. Let yt be the training labels.

      - If imblearn is available:
          3. Try to construct a SMOTE sampler. If that fails, fall back to
             imblearn's RandomOverSampler.
          4. Use sampler.fit_resample(Xt, yt) to create a balanced training set.

      - If imblearn is *not* available:
          3. Manually oversample the minority class with NumPy to match the size
             of the majority class.

      - Then:
          4. Train the estimator on balanced data (Xs, ys).
          5. Evaluate on the original test set.

    Returns a RunResult labeled according to which path was used:
      - "Pre: SMOTE"
      - "Pre: SMOTE (approx)" (fallback labeling if IMBLEARN_OK but SMOTE fails)
      - "Pre: Oversample (fallback)" (if no imblearn support)
    """
    #Build and fit preprocessor on train+val
    prep = build_preprocessor(X_tr, protected_cols)
    Xt = prep.fit_transform(X_tr)
    yt = y_tr.to_numpy()
    
    #Use IMBLEARN if available
    if IMBLEARN_OK:
        try:
            from imblearn.over_sampling import SMOTE, RandomOverSampler
            sampler = SMOTE(random_state=42)
        except Exception:
            sampler = RandomOverSampler(random_state=42)
    else:
        #simple numpy oversample
        print("IMBLEARN not available, using simple oversampling fallback.")
        #Identify pos and neg indices
        pos = np.where(yt==1)[0]; neg = np.where(yt==0)[0]

        if len(pos)==0 or len(neg)==0: #nothing to rebalance, return baseline
            Xs, ys = Xt, yt
        else:
            nmax = max(len(pos), len(neg)) #Find target size for each class as maximum count
            #Oversample both positives and negatives to size nmax
            pos_up = np.random.choice(pos, size=nmax, replace=True)
            neg_up = np.random.choice(neg, size=nmax, replace=True)
            #Concatenate oversampled indices
            idx = np.concatenate([pos_up, neg_up])
            #Balanced feature and label matrix
            Xs, ys = Xt[idx], yt[idx]
        #Build and fit estimator on balanced data
        clf = build_estimator(model_name, params)
        clf.fit(Xs, ys)
        #Predict on test set
        p = to_proba(clf, prep.transform(X_te))
        yhat = (p >= 0.5).astype(int) #Hard predictions at 0.5 threshold
        tag = "Pre: Oversample (fallback)" if not IMBLEARN_OK else "Pre: SMOTE (approx)"
        return evaluate_run(tag, y_te.to_numpy(), p, yhat, A_te, test_index=X_te.index, fair_model=make_standard_fair_model(
            name=tag,
            features=X_tr.columns,
            protected_cols=protected_cols,
            preprocessor=prep,
            estimator=clf,
            threshold=0.5,
            outcome_col=None,
            metadata={
                "source": "FairSelect",
                "technique": tag,
                "model_name": model_name,
                "model_params": params,
                "uses_sample_weight": False,
            },
        ))

    #Use imblearn sampler to fit_resample
    #Obtain a balanced training set
    Xs, ys = sampler.fit_resample(Xt, yt)
    #Build and fit estimator on balanced data
    clf = build_estimator(model_name, params)
    clf.fit(Xs, ys)
    #Predict on test set
    p = to_proba(clf, prep.transform(X_te))
    yhat = (p >= 0.5).astype(int) #Hard predictions at 0.5 threshold
    fair_model = make_standard_fair_model(
        name="Pre: SMOTE / Oversample",
        features=X_tr.columns,
        protected_cols=protected_cols,
        preprocessor=prep,
        estimator=clf,
        threshold=0.5,
        outcome_col=None,
        metadata={
            "source": "FairSelect",
            "technique": "Pre:SMOTE / Oversample",
            "model_name": model_name,
            "model_params": params,
            "resampling": "SMOTE_or_ROS",
        },
    )

    return evaluate_run(
        "Pre: SMOTE / Oversample",
        y_te.to_numpy(),
        p,
        yhat,
        A_te,
        fair_model=fair_model,
        test_index=X_te.index,
    )

def run_local_massaging(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected_cols, all_df_train):
    """
    Pre-processing: label massaging based on boundary scores.

    Workflow:
      1. Build a preprocessor on train+val, fit it, and transform X_tr.
      2. Train a temporary baseline classifier on (Xt, y_tr).
      3. Use this baseline to compute training probabilities `scores`.
      4. Apply local_massaging_fit_flip(y_tr, scores, A_tr) to get new labels
         y_mass that equalize positive rates across groups.
      5. Train a *final* classifier on (Xt, y_mass).
      6. Evaluate on the original (unmodified) test set X_te.

    The massaging happens only on labels in the training data, not on features.
    """
    #Build and fit preprocessor on train+val
    prep = build_preprocessor(pd.concat([X_tr,X_va]), protected_cols)
    Xt = prep.fit_transform(X_tr)
    #Build and fit temporary baseline on training data
    base = build_estimator(model_name, params)
    base.fit(Xt, y_tr)
    #Get training probabilities from baseline
    scores = to_proba(base, Xt)
    #Apply local massaging to get new training labels
    y_mass = local_massaging_fit_flip(y_tr, scores, A_tr)
    #Build and fit final estimator on massaged labels
    clf = build_estimator(model_name, params)
    clf.fit(Xt, y_mass)
    #Predict on test set
    p = to_proba(clf, prep.transform(X_te))
    yhat = (p >= 0.5).astype(int) #Hard predictions at 0.5 threshold
    fair_model = make_standard_fair_model(
        name="Pre: Local Massaging",
        features=X_tr.columns,
        protected_cols=protected_cols,
        preprocessor=prep,
        estimator=clf,
        threshold=0.5,
        outcome_col=None,
        metadata={
            "source": "FairSelect",
            "technique": "Pre:Local Massaging",
            "model_name": model_name,
            "model_params": params,
        },
        )

    return evaluate_run(
        "Pre: Local Massaging",
        y_te.to_numpy(),
        p,
        yhat,
        A_te,
        fair_model=fair_model,
        test_index=X_te.index,
    )