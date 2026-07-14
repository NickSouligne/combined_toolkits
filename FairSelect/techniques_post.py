from typing import Dict

import numpy as np
import pandas as pd

from .core import build_estimator, build_preprocessor, evaluate_run
from .utils import (
    youden_threshold, to_proba, confusion_rates,input_repair_standardize_by_group, apply_multiaccuracy_boost, _build_auditor, _logit, _make_group_onehot, _sigmoid
)
from .FairModel_helper import InputRepairPredictor, make_predictor_fair_model, StandardPredictor, MultiaccuracyBoostPredictor, RejectOptionShiftPredictor, KamiranRejectOptionPredictor
from FairModel import FairModel

def group_thresholds_youden(groups: pd.Series, y_val: np.ndarray, p_val: np.ndarray) -> Dict[str, float]:
    """
    Compute group-specific decision thresholds using Youden's J statistic.

    For each unique group label in `groups`, this function:
      1. Selects the validation examples belonging to that group.
      2. Calls `youden_threshold` on that group's labels and probabilities to
         find the probability cutoff that maximizes:
             J = TPR - FPR
      3. Stores this optimal threshold in a dictionary keyed by the group label
         (as a string).

    Parameters
    ----------
    groups : pd.Series
        Group labels for each validation example (e.g., intersectional A_va).
    y_val : np.ndarray
        Binary ground-truth labels (0/1) for the validation set.
    p_val : np.ndarray
        Predicted probabilities for the validation set (same order as y_val).

    Returns
    -------
    Dict[str, float]
        Mapping from group label (stringified) to its Youden-optimal threshold.
    """
    #Dictionary to hold group-specific thresholds
    th = {}

    #Iterate over each unique group
    for g in np.unique(groups):
        #Boolean mask for this group
        m = (groups==g)
        #Skip if no members in this group
        if m.sum()==0: continue
        #Compute and store Youden-optimal threshold for this group
        th[str(g)] = youden_threshold(y_val[m], p_val[m])
    return th

def predict_with_group_thresholds(groups: pd.Series, p: np.ndarray, thresholds: Dict[str,float], default: float=0.5) -> np.ndarray:
    """
    Convert probabilities into predictions using group-specific thresholds.

    For each group:
      - Look up its threshold from `thresholds` (if missing, use `default`).
      - Apply that threshold to all probability scores for examples in that group.

    Parameters
    ----------
    groups : pd.Series
        Group labels for each example in the set to be predicted (e.g., A_te).
    p : np.ndarray
        Predicted probabilities for the corresponding examples.
    thresholds : Dict[str, float]
        Mapping from group label (as string) to its decision threshold.
    default : float, optional
        Default threshold to use if a group's threshold is not found in the
        dictionary, by default 0.5.

    Returns
    -------
    np.ndarray
        Binary predictions (0/1) of the same shape as `p`, after applying
        group-specific thresholds.
    """
    #Initialize prediction array
    yhat = np.zeros_like(p, dtype=int)
    #Iterate over each group
    for g in np.unique(groups):
        #Get this group's threshold (or default if missing)
        t = thresholds.get(str(g), default)
        #Boolean mask for this group
        m = (groups==g)
        #Apply threshold to this group's probabilities
        yhat[m] = (p[m] >= t).astype(int)
    return yhat


def run_group_youden_postproc(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected_cols, all_df_train):
    '''
    Post-processing: per-group Youden-optimal thresholds.

    Workflow:
      1. Train a base classifier on the training set with a shared preprocessor.
      2. On the validation set, get predicted probabilities p_val.
      3. For each group, compute the Youden-optimal threshold on (y_val, p_val).
      4. On the test set, get predicted probabilities p_test from the same model.
      5. Convert p_test to predictions using the group-specific thresholds.

    This does not alter the model itself, only the decision thresholds per group.
    '''
    #Build preprocessor on train/val
    prep = build_preprocessor(pd.concat([X_tr,X_va]), protected_cols)
    #Fit preprocessor on train/val
    prep.fit(pd.concat([X_tr,X_va]))
    #Create the base estimator
    clf = build_estimator(model_name, params)
    #Fit base estimator on training data
    clf.fit(prep.transform(X_tr), y_tr)
    #Compute predicted probabilities on validation set
    p_val = to_proba(clf, prep.transform(X_va))
    #Compute group-wise Youden-optimal thresholds on validation set
    th = group_thresholds_youden(A_va, y_va.to_numpy(), p_val)
    #Compute predicted probabilities on test set
    p_test = to_proba(clf, prep.transform(X_te))
    #Apply group-specific thresholds to test predictions to get final predictions
    yhat = predict_with_group_thresholds(A_te, p_test, th, default=0.5)
    predictor = StandardPredictor(
        features=X_tr.columns,
        protected_cols=protected_cols,
        preprocessor=prep,
        estimator=clf,
        threshold=0.5,
        group_thresholds=th,
    )

    fair_model = make_predictor_fair_model(
        name="Post: Youden per group",
        features=X_tr.columns,
        protected_cols=protected_cols,
        predictor=predictor,
        threshold=0.5,
        outcome_col=None,
        metadata={
            "source": "FairSelect",
            "technique": "Post:Youden per group",
            "model_name": model_name,
            "model_params": params,
            "group_thresholds": th,
        },
    )

    return evaluate_run(
        "Post: Youden per group",
        y_te.to_numpy(),
        p_test,
        yhat,
        A_te,
        fair_model=fair_model,
    )






def run_multiaccuracy_boost(
    model_name,
    params,
    X_tr, X_va, X_te,
    y_tr, y_va, y_te,
    A_tr, A_va, A_te,
    protected_cols,
    all_df_train=None,
    *,
    alpha=0.02,
    eta=None,
    max_iters=25,
    auditor_type="ridge",
    random_state=0,
    eps=1e-6,
    include_group_in_auditor=True,
):
    """
    Kim et al. (2018)-style Multiaccuracy Boost (practical implementation).

    Structure:
      1) Train base model on train.
      2) Use validation as the "audit" set.
      3) Fix partitions based on base model f0:
           X  : all points
           X0 : points where f0(x) <= 0.5
           X1 : points where f0(x) >  0.5
      4) Iterate:
           - compute residual r_t = p_t - y on audit set
           - train an auditor h(x) to predict r_t on each partition
           - pick partition with max correlation score E[h(x)*r_t] on that partition
           - if max score <= alpha: stop
           - update logits on that partition: logit(p) <- logit(p) - eta*h(x)
         Apply the same learned auditor update to BOTH validation and test logits.

    Notes:
      - This is iterative residual-auditing + updates (boosting-style),
        not a single logistic meta-model on [p, group_onehot].
      - The auditor can use richer features than just group membership.
      - We update in logit space to keep probabilities in (0,1) stably.

    Returns:
      evaluate_run(...) output for test set, using final adjusted probabilities.
    """
    if eta is None:
        eta = alpha  # Kim suggests eta = O(alpha); alpha is a reasonable default.

    prep = build_preprocessor(pd.concat([X_tr, X_va]), protected_cols)
    prep.fit(pd.concat([X_tr, X_va]))

    clf = build_estimator(model_name, params)
    clf.fit(prep.transform(X_tr), y_tr)

    # -------------------------------------------------------
    # 2. Base probabilities on validation and test sets
    # -------------------------------------------------------
    p_val = to_proba(clf, prep.transform(X_va))
    p_test = to_proba(clf, prep.transform(X_te))

    # -------------------------------------------------------
    # 3. Apply multiaccuracy boost to test probabilities
    # -------------------------------------------------------
    p_adj = apply_multiaccuracy_boost(
        X_va=X_va,
        X_te=X_te,
        y_va=y_va,
        A_va=A_va,
        A_te=A_te,
        p_val=p_val,
        p_test=p_test,
        prep=prep,
        alpha=0.02,
        eta=eta,
        max_iters=25,
        auditor_type="ridge",
        random_state=params.get("random_state", 0),
        include_group_in_auditor=True,
    )

    yhat = (p_adj >= 0.5).astype(int)

    # -------------------------------------------------------
    # 4. Build FairModel-compatible predictor
    # -------------------------------------------------------
    ma_predictor = MultiaccuracyBoostPredictor(
        features=list(X_tr.columns),
        protected_cols=list(protected_cols),
        preprocessor=prep,
        estimator=clf,
        X_val=X_va,
        y_val=y_va,
        A_val=A_va,
        p_val=p_val,
        threshold=0.5,
        alpha=0.02,
        eta=eta,
        max_iters=25,
        auditor_type="ridge",
        random_state=params.get("random_state", 0),
        include_group_in_auditor=True,
    )

    fair_model = FairModel(
        name="Post: Multiaccuracy Boost",
        features=list(X_tr.columns),
        protected_cols=list(protected_cols),
        predictor=ma_predictor,
        threshold=0.5,
        outcome_col=None,
        positive_label=1,
        metadata={
            "source": "FairSelect",
            "technique": "Post:Multiaccuracy Boost",
            "model_name": model_name,
            "model_params": params,
            "postprocessing": "multiaccuracy_boost",
            "alpha": 0.02,
            "eta": eta,
            "max_iters": 25,
            "auditor_type": "ridge",
            "include_group_in_auditor": True,
        },
    )

    return evaluate_run(
        "Post: Multiaccuracy Boost",
        y_te.to_numpy(),
        p_adj,
        yhat,
        A_te,
        fair_model=fair_model,
    )


def run_reject_option_shift(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected_cols, all_df_train):
    """
    Post-processing: simple reject-option-style threshold shifting.
    Loosely based on Hardt et al. (2016) Equality of Opportunity in supervised learning

    Workflow:
      1. Train a base classifier on the training set.
      2. Get probabilities p_val on the validation set.
      3. At a fixed threshold 0.5, compute group-wise TPRs on validation.
      4. Identify:
           - group with lowest TPR ("worst")
           - group with highest TPR ("best")
      5. On test:
           - Start with a threshold of 0.5 for all groups.
           - Decrease threshold for worst group by 0.05 (down to a minimum of 0.0).
           - Increase threshold for best group by 0.05 (up to a maximum of 1.0).
           - Use these group-specific thresholds for final predictions.

    Intuition:
      - Groups with lower TPR (more false negatives) get a slightly more lenient
        threshold (more positives).
      - Groups with higher TPR get a slightly stricter threshold.
    """
    
    #simple boundary widening in favor of group with lowest TPR on val
    #build and fit preprocessor on train/val
    prep = build_preprocessor(pd.concat([X_tr,X_va]), protected_cols)
    prep.fit(pd.concat([X_tr,X_va]))
    
    #Build base estimator
    clf = build_estimator(model_name, params)
    #Fit base estimator on training data
    clf.fit(prep.transform(X_tr), y_tr)
    #Predict probabilities on validation set
    p_val = to_proba(clf, prep.transform(X_va))

    #Compute group-wise TPRs at global threshold 0.5
    yhat_val = (p_val>=0.5).astype(int) #Hard predictions at 0.5
    groups = pd.Series(A_va).unique() #Find unique groups in validation set
    tprs = {} #Dict to map TPRs to group

    #Iterate over each group to compute TPR
    for g in groups:
        #Boolean mask for this group
        m = (A_va==g).to_numpy()
        #Compute confusion rates (TPR, FPR, etc.)
        cr = confusion_rates(y_va.to_numpy()[m], yhat_val[m])
        #Store TPR for this group
        tprs[str(g)] = cr["TPR"]
    #Identify the worst group by TPR
    worst = min(tprs, key=lambda k: (tprs[k] if not np.isnan(tprs[k]) else -1))
    #Predict probabilities on test set
    p_test = to_proba(clf, prep.transform(X_te))
    #Identify the best group by TPR
    best = max(tprs, key=lambda k: (tprs[k] if not np.isnan(tprs[k]) else -1))
    #Start with global threshold of 0.5
    th = {g:0.5 for g in groups}
    #Adjust thresholds: lower for worst TPR group, raise for best TPR group
    th[str(worst)] = max(0.0, th[str(worst)]-0.05)
    th[str(best)]  = min(1.0, th[str(best)]+0.05)
    #Apply adjusted thresholds to test set
    yhat = predict_with_group_thresholds(A_te, p_test, th, default=0.5)
    reject_predictor = RejectOptionShiftPredictor(
        features=list(X_tr.columns),
        protected_cols=list(protected_cols),
        preprocessor=prep,
        estimator=clf,
        thresholds=th,
        default_threshold=0.5,
    )

    fair_model = FairModel(
        name="Post: Reject-Option Shift",
        features=list(X_tr.columns),
        protected_cols=list(protected_cols),
        predictor=reject_predictor,
        threshold=0.5,
        group_thresholds=th,
        outcome_col=None,
        positive_label=1,
        metadata={
            "source": "FairSelect",
            "technique": "Post:Reject-Option Shift",
            "model_name": model_name,
            "model_params": params,
            "postprocessing": "simple_threshold_shift",
            "thresholds": th,
            "worst_tpr_group": str(worst),
            "best_tpr_group": str(best),
            "validation_tprs": tprs,
            "base_threshold": 0.5,
            "shift": 0.05,
        },
    )

    return evaluate_run(
        "Post: Reject-Option Shift",
        y_te.to_numpy(),
        p_test,
        yhat,
        A_te,
        fair_model=fair_model,
    )

def run_input_repair(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected_cols, all_df_train):
    '''
    Post-processing: input repair via per-group standardization alignment.

    Idea:
      - Train a base classifier on the original (unrepaired) training data.
      - At test time, "repair" the test inputs so that, for each group, the
        test distribution is standardized to align with that group's training
        distribution (z-alignment).
      - Then feed the repaired test inputs into the same trained model.

    This approach changes only the test features (post-hoc), not the trained
    model parameters, and attempts to mitigate distributional shifts or group
    disparities in feature scaling.
    '''
    #Build and fit preprocessor on train/val
    prep = build_preprocessor(pd.concat([X_tr,X_va]), protected_cols)
    prep.fit(pd.concat([X_tr,X_va]))
    #Build base estimator
    clf = build_estimator(model_name, params)
    #Fit base estimator on training data
    clf.fit(prep.transform(X_tr), y_tr)
    #Combine train and val for input repair fitting
    A_train_all = pd.concat([A_tr, A_va], axis=0)
    #Repair the rest features X_te by aligning them with the train/val distribution for each group
    X_rep = input_repair_standardize_by_group(pd.concat([X_tr, X_va]), X_te, A_train_all, A_te)
    #Predict on repaired test inputs
    p = to_proba(clf, prep.transform(X_rep))
    yhat = (p>=0.5).astype(int) #Hard predictions at 0.5 threshold
    input_repair_predictor = InputRepairPredictor(
            features=list(X_tr.columns),
            protected_cols=list(protected_cols),
            preprocessor=prep,
            estimator=clf,
            X_repair_reference=X_repair_reference,
            A_repair_reference=A_repair_reference,
            threshold=0.5,
        )

    fair_model = FairModel(
        name="Post: Input Repair (z-align)",
        features=list(X_tr.columns),
        protected_cols=list(protected_cols),
        predictor=input_repair_predictor,
        threshold=0.5,
        outcome_col=None,
        positive_label=1,
        metadata={
            "source": "FairSelect",
            "technique": "Post:Input Repair",
            "model_name": model_name,
            "model_params": params,
            "postprocessing": "input_repair_standardize_by_group",
            "repair_reference": "train_plus_validation",
            "threshold": 0.5,
        },
    )

    return evaluate_run(
        "Post: Input Repair (z-align)",
        y_te.to_numpy(),
        p,
        yhat,
        A_te,
        fair_model=fair_model,
    )




def run_reject_option_kamiran(
    model_name, params,
    X_tr, X_va, X_te,
    y_tr, y_va, y_te,
    A_tr, A_va, A_te,
    protected_cols, all_df_train,
    fairness_objective="spd",
    theta_grid=None,
    base_threshold=0.5,
    fairness_bound=None,
    max_acc_drop=None,
    unprivileged_values=None,
):
    """
    Kamiran-style Reject Option Classification (ROC) post-processing.

    IMPORTANT ADAPTATION FOR YOUR PIPELINE:
      - Keeps y_te / A_te in their original types when calling evaluate_run
        (so if your pipeline expects pandas and calls .to_numpy(), it won't crash).
      - Uses NumPy arrays internally for math, but does NOT pass them into evaluate_run.
    """

    # -----------------------------
    # Preprocess + train base model
    # -----------------------------
    prep = build_preprocessor(pd.concat([X_tr, X_va]), protected_cols)
    prep.fit(pd.concat([X_tr, X_va]))

    clf = build_estimator(model_name, params)
    clf.fit(prep.transform(X_tr), y_tr)

    p_val = to_proba(clf, prep.transform(X_va))
    p_te  = to_proba(clf, prep.transform(X_te))

    # ---- internal numpy views (NO .to_numpy calls anywhere) ----
    y_va_np = np.asarray(y_va, dtype=int)
    A_va_np = np.asarray(A_va)

    # baseline on val (internal only)
    yhat_val_base = (p_val >= base_threshold).astype(int)
    val_acc_base = float(np.mean(yhat_val_base == y_va_np))

    # -----------------------------
    # Choose unprivileged group(s)
    # -----------------------------
    if unprivileged_values is None:
        # infer from training base outcome rate (works for Series or ndarray)
        y_tr_np = np.asarray(y_tr, dtype=int)
        A_tr_np = np.asarray(A_tr)
        # group with lowest positive rate treated as unprivileged
        rates = {}
        for g in pd.Series(A_tr_np).unique():
            m = (A_tr_np == g)
            if m.sum() > 0:
                rates[g] = float(np.mean(y_tr_np[m]))
        if len(rates) > 0:
            unprivileged_values = {min(rates, key=rates.get)}
        else:
            unprivileged_values = set()
    else:
        unprivileged_values = set(unprivileged_values)

    # -----------------------------
    # Theta grid
    # -----------------------------
    if theta_grid is None:
        theta_grid = np.linspace(0.50, 0.95, 46)

    fairness_objective = str(fairness_objective).lower()

    # -----------------------------
    # Fairness metric on validation
    # -----------------------------
    def spd(yhat, A):
        # P(yhat=1|unpriv) - P(yhat=1|priv)
        unpriv = np.isin(A, list(unprivileged_values))
        priv = ~unpriv
        if unpriv.sum() == 0 or priv.sum() == 0:
            return np.nan
        return float(yhat[unpriv].mean() - yhat[priv].mean())

    def eod(y_true, yhat, A):
        # TPR(unpriv) - TPR(priv)
        unpriv = np.isin(A, list(unprivileged_values))
        priv = ~unpriv
        if unpriv.sum() == 0 or priv.sum() == 0:
            return np.nan
        cr_u = confusion_rates(np.asarray(y_true)[unpriv], np.asarray(yhat)[unpriv])
        cr_p = confusion_rates(np.asarray(y_true)[priv], np.asarray(yhat)[priv])
        return float(cr_u.get("TPR", np.nan) - cr_p.get("TPR", np.nan))

    def aod(y_true, yhat, A):
        # 0.5[(TPR diff)+(FPR diff)]
        unpriv = np.isin(A, list(unprivileged_values))
        priv = ~unpriv
        if unpriv.sum() == 0 or priv.sum() == 0:
            return np.nan
        cr_u = confusion_rates(np.asarray(y_true)[unpriv], np.asarray(yhat)[unpriv])
        cr_p = confusion_rates(np.asarray(y_true)[priv], np.asarray(yhat)[priv])
        return float(0.5 * ((cr_u.get("TPR", np.nan) - cr_p.get("TPR", np.nan)) +
                            (cr_u.get("FPR", np.nan) - cr_p.get("FPR", np.nan))))

    # -----------------------------
    # ROC post-processing primitive
    # -----------------------------
    def roc_predict(p, A, theta):
        p = np.asarray(p, dtype=float)
        A = np.asarray(A)
        yhat = (p >= base_threshold).astype(int)

        # critical region: max(p,1-p) <= theta  <=> p in [1-theta, theta]
        in_critical = (np.maximum(p, 1.0 - p) <= float(theta))

        unpriv = np.isin(A, list(unprivileged_values))
        priv = ~unpriv

        yhat[in_critical & unpriv] = 1
        yhat[in_critical & priv] = 0
        return yhat, in_critical

    # -----------------------------
    # Grid search theta on validation
    # -----------------------------
    candidates = []
    for theta in theta_grid:
        yhat_val, in_critical = roc_predict(p_val, A_va_np, theta)

        val_acc = float(np.mean(yhat_val == y_va_np))

        if fairness_objective == "spd":
            fair = spd(yhat_val, A_va_np)
        elif fairness_objective == "eod":
            fair = eod(y_va_np, yhat_val, A_va_np)
        elif fairness_objective == "aod":
            fair = aod(y_va_np, yhat_val, A_va_np)
        else:
            raise ValueError("fairness_objective must be one of: 'spd', 'eod', 'aod'")

        if np.isnan(fair):
            continue

        acc_drop = val_acc_base - val_acc
        meets_acc = True if max_acc_drop is None else (acc_drop <= max_acc_drop)
        meets_fair = True if fairness_bound is None else (abs(fair) <= fairness_bound)

        candidates.append((float(theta), float(fair), float(val_acc), float(acc_drop), float(np.mean(in_critical)),
                           bool(meets_acc), bool(meets_fair)))

    if len(candidates) == 0:
        # Fallback to baseline, BUT preserve y_te / A_te types for evaluate_run
        yhat_te = (p_te >= base_threshold).astype(int)
        return evaluate_run("Post: Kamiran Reject Option (fallback baseline)", y_te, p_te, yhat_te, A_te)

    constrained = [c for c in candidates if c[5] and c[6]]
    pool = constrained if len(constrained) else candidates

    # sort by |fair|, then higher acc, then smaller critical region
    pool.sort(key=lambda t: (abs(t[1]), -t[2], t[4]))
    best_theta, best_fair, best_acc, _, best_crit_frac, _, _ = pool[0]

    print(f"Selected theta={best_theta:.3f} with fairness={best_fair:.4f}, "
          f"val_acc={best_acc:.4f}, crit_region={best_crit_frac:.4f}")

    # -----------------------------
    # Apply on test (keep y_te/A_te as-is when evaluating)
    # -----------------------------
    # NOTE: we only need A_te as numpy internally to compute group masks;
    # we do NOT pass that numpy version into evaluate_run.
    yhat_te, _ = roc_predict(p_te, np.asarray(A_te), best_theta)

    kamiran_predictor = KamiranRejectOptionPredictor(
        features=list(X_tr.columns),
        protected_cols=list(protected_cols),
        preprocessor=prep,
        estimator=clf,
        unprivileged_values=unprivileged_values,
        theta=best_theta,
        base_threshold=base_threshold,
    )

    fair_model = FairModel(
        name=f"Post: Kamiran Reject Option (theta={best_theta:.3f}, obj={fairness_objective})",
        features=list(X_tr.columns),
        protected_cols=list(protected_cols),
        predictor=kamiran_predictor,
        threshold=base_threshold,
        outcome_col=None,
        positive_label=1,
        metadata={
            "source": "FairSelect",
            "technique": "Post:Reject-Option Kamiran",
            "model_name": model_name,
            "model_params": params,
            "postprocessing": "kamiran_reject_option",
            "fairness_objective": fairness_objective,
            "theta": float(best_theta),
            "base_threshold": float(base_threshold),
            "unprivileged_values": sorted(list(unprivileged_values)),
            "validation_fairness": float(best_fair),
            "validation_accuracy": float(best_acc),
            "validation_accuracy_drop": float(best_acc_drop),
            "validation_critical_region_fraction": float(best_crit_frac),
            "fairness_bound": fairness_bound,
            "max_acc_drop": max_acc_drop,
        },
    )

    return evaluate_run(
        f"Post: Kamiran Reject Option (theta={best_theta:.3f}, obj={fairness_objective})",
        y_te,
        p_te,
        yhat_te,
        A_te,
        fair_model=fair_model,
    )