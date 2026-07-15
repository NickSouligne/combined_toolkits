from typing import Any, Dict

import numpy as np
import pandas as pd

from .deps import FAIRLEARN_OK, ExponentiatedGradient, EqualizedOdds, DemographicParity, IsotonicRegression, AIF360_OK
from .core import build_estimator, build_preprocessor, evaluate_run, RunResult
from .utils import (
    to_proba, group_balanced_bootstrap_indices,
    fit_with_optional_sample_weight, ece_bin, confusion_rates,
)
from .techniques_pre import compute_reweights, local_massaging_fit_flip
from .FairModel_helper import GroupModelPredictor, make_standard_fair_model, StandardPredictor, make_predictor_fair_model, PrejudiceRemoverPredictor, GroupBalancedEnsemblePredictor
from FairModel import FairModel


def run_compositional_models(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected_cols, all_df_train):
    '''
    Runs compositional per-group models (in-processing): one model per group in training data
    Falls back to pooled model if group not seen in training data
    0.5 cutoff
    '''
    #Get unique groups in training data
    groups = pd.Series(A_tr).unique()
    #Build a preprocessor on train+val
    prep = build_preprocessor(pd.concat([X_tr,X_va]), protected_cols)
    #Fit the preprocessor on train+val
    prep.fit(pd.concat([X_tr,X_va]))
    models = {} #Store trained classifiers for each label
    
    #Train per-group models
    for g in groups:
        #Boolean mask to select only records from group g
        m = (A_tr==g)
        #Skip groups with too few samples (5 is arbitrary, may need to revisit)
        if m.sum()<5:
            continue
        #Build and fit the estimator on only group g data
        clf = build_estimator(model_name, params)
        clf.fit(prep.transform(X_tr[m]), y_tr[m])
        models[str(g)] = clf #Store classifier in the dict
    
    
    #Predict on test set using per-group models

    #Array to hold test-set probabilities P[i] = P(Y=1 | X_te[i])
    P = np.zeros(len(X_te))
    #Iterate over each test instance and group label
    for i, g in enumerate(A_te):
        g = str(g) #Convert to string for dict lookup

        if g in models: #If model for this group exists use it
            #Extracts the i-th test instance, transforms it, predicts probability (transform expects 2D array (shape, (1, n_features)))
            P[i] = to_proba(models[g], prep.transform(X_te.iloc[[i], :]))[0] #to_proba returns 1d array, take first element
        else:
            #Group not seen in training data, fall back to pooled model
            pooled = build_estimator(model_name, params)
            pooled.fit(prep.transform(X_tr), y_tr)
            P[i] = to_proba(pooled, prep.transform(X_te.iloc[[i], :]))[0]
    #Hard predictions at 0.5 threshold (TODO: Allow user to define different threshold)
    yhat = (P >= 0.5).astype(int)
    fallback_model = build_estimator(model_name, params)
    fallback_model.fit(prep.transform(X_tr), y_tr)

    predictor = GroupModelPredictor(
        features=X_tr.columns,
        protected_cols=protected_cols,
        preprocessor=prep,
        group_models=models,
        fallback_model=fallback_model,
        threshold=0.5,
    )

    fair_model = make_predictor_fair_model(
        name="In: Compositional (per-group)",
        features=X_tr.columns,
        protected_cols=protected_cols,
        predictor=predictor,
        threshold=0.5,
        metadata={
            "source": "FairSelect",
            "technique": "In:Compositional per-group",
            "model_name": model_name,
            "model_params": params,
            "n_group_models": len(models),
        },
    )

    return evaluate_run(
        "In: Compositional (per-group)",
        y_te.to_numpy(),
        P,
        yhat,
        A_te,
        fair_model=fair_model,
        test_index=X_te.index,
    )

def run_prejudice_remover(model_name, params,
                          X_tr, X_va, X_te, y_tr, y_va, y_te,
                          A_tr, A_va, A_te,
                          protected_cols, all_df_train,
                          *, eta: float = 25.0):
    """
    In-processing fairness regularization using AIF360 PrejudiceRemover.
    Not well validated yet, use with caution.
    """
    import sys, numpy as np, pandas as pd

    for _name, _alias in {"float": float, "int": int, "bool": bool, "object": object, "complex": complex}.items():
        if not hasattr(np, _name):
            setattr(np, _name, _alias)

    #--- Import AIF360 *inside* fn --- (Script didnt recognize this at top level, need to review why)
    try:
        from aif360.datasets import BinaryLabelDataset
        from aif360.algorithms.inprocessing import PrejudiceRemover
    except Exception as imp_err:
        print(f"[PR] aif360 import failed: {imp_err}\n"
              f"NumPy {np.__version__} @ {getattr(np,'__file__','n/a')}\n"
              f"sys.path[:5]={sys.path[:5]}", file=sys.stderr)
        return run_baseline(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected_cols, all_df_train)

    #--- Preprocess features (fit on train+val), force dense 2D ---
    #Build preprocessor on train+val
    prep = build_preprocessor(pd.concat([X_tr, X_va]), protected_cols)
    prep.fit(pd.concat([X_tr, X_va]))
    #transform train & test
    Xtr = prep.transform(X_tr); Xte = prep.transform(X_te)
    #Data check to ensure we have arrays of type float and strictly 2D
    if hasattr(Xtr, "toarray"): Xtr = Xtr.toarray()
    if hasattr(Xte, "toarray"): Xte = Xte.toarray()
    Xtr = np.asarray(Xtr, dtype=float)
    Xte = np.asarray(Xte, dtype=float)
    if Xtr.ndim == 1: Xtr = Xtr.reshape(-1, 1)
    if Xte.ndim == 1: Xte = Xte.reshape(-1, 1)

    #If only one feature remains, add a neutral dummy column to keep AIF360 strictly 2D
    if Xtr.shape[1] == 1:
        Xtr = np.column_stack([Xtr, np.zeros((Xtr.shape[0], 1), dtype=float)])
        Xte = np.column_stack([Xte, np.zeros((Xte.shape[0], 1), dtype=float)])

    #Labels & sensitive attribute (intersectional)
    #Convert labels to float 0.0/1.0
    ytr = pd.Series(y_tr).astype(float)  #ensure 0/1
    yte = pd.Series(y_te).astype(float)
    #Encode sensitive attribute as categorical codes (stable between train & test)
    cat = pd.Categorical(A_tr.astype(str))  #stable categories from train
    sens_tr = pd.Series(cat.codes, index=ytr.index).astype(float)
    sens_te = pd.Series(pd.Categorical(A_te.astype(str), categories=cat.categories).codes,
                        index=yte.index).astype(float)

    #Build DataFrames (only features + 'sensitive' + 'label')
    #Generate feature column names
    feat_cols = [f"x{i}" for i in range(Xtr.shape[1])]
    #Build training dataframe
    df_tr = pd.DataFrame(Xtr, columns=feat_cols)
    df_tr["sensitive"] = sens_tr.values
    df_tr["label"] = ytr.values
    #Build test dataframe
    df_te = pd.DataFrame(Xte, columns=feat_cols)
    df_te["sensitive"] = sens_te.values
    df_te["label"] = yte.values
    df_tr = df_tr.dropna(axis=0).reset_index(drop=True)
    df_te = df_te.dropna(axis=0).reset_index(drop=True)

    #--- BinaryLabelDataset via df= path---
    try:
        dtr = BinaryLabelDataset(
            df=df_tr,
            label_names=["label"],
            protected_attribute_names=["sensitive"],
            favorable_label=1.0, unfavorable_label=0.0,
        )
        dte = BinaryLabelDataset(
            df=df_te,
            label_names=["label"],
            protected_attribute_names=["sensitive"],
            favorable_label=1.0, unfavorable_label=0.0,
        )
    except Exception as ds_err:
        print(f"[PR] BinaryLabelDataset construction failed: {ds_err}", file=sys.stderr)
        return run_baseline(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected_cols, all_df_train)

    #Fit Prejudice Remover (Ran into some dependency issues here, need to fix properly later)
    try:
        #Initialize PrejudiceRemover with specified eta and the sensitive attribute
        pr = PrejudiceRemover(eta=float(eta), sensitive_attr="sensitive")
        import os, sys

        #Make sure the child "python" resolves to THIS interpreter 
        venv_dir = os.path.dirname(sys.executable)            
        os.environ["PATH"] = venv_dir + os.pathsep + os.environ.get("PATH", "")

        #Help the child process locate site-packages explicitly
        site_dir = os.path.dirname(os.__file__)               #stdlib dir
        os.environ.setdefault("PYTHONHOME", os.path.dirname(site_dir))
        #Ensure current sys.path entries are visible to the child (robust, but optional)
        os.environ["PYTHONPATH"] = os.pathsep.join(sys.path + [os.environ.get("PYTHONPATH","")])

        #Fit the PrejudiceRemover model on the training dataset
        pr.fit(dtr)
    except Exception as fit_err:
        print("[PR] PrejudiceRemover.fit failed:", fit_err, file=sys.stderr)
        print("[PR] Diagnostics:",
              {"numpy": np.__version__,
               "train_df_shape": df_tr.shape, "test_df_shape": df_te.shape,
               "train_head": df_tr.head(2).to_dict(orient="list")}, file=sys.stderr)
        return run_baseline(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected_cols, all_df_train)

    #Predict & evaluate

    #Use the fitted PrejudiceRemover to predict on the test BinaryLabelDataset
    dte_pred = pr.predict(dte)
    #Try to use calibrated scores if available, otherwise fall derive from labels
    if getattr(dte_pred, "scores", None) is not None:
        p = np.asarray(dte_pred.scores, dtype=float).ravel()
        p = np.clip(p, 0.0, 1.0) #Clip probabilities to [0,1]
    else:
        p = np.asarray(dte_pred.labels, dtype=float).ravel()
        if p.min() < 0:  #{-1,1} -> {0,1} (remap labels)
            p = (p > 0).astype(float)

    #Extract aligned true labels and convert to integer
    y_true = df_te["label"].to_numpy().astype(int)               #aligned after NaN drop
    yhat   = (p >= 0.5).astype(int) #Hard predictions at 0.5 threshold
    A_eval = pd.Series(df_te["sensitive"].astype(int).astype(str)) #Evaluate groups with the sensitive attribute

    pr_predictor = PrejudiceRemoverPredictor(
        features=list(X_tr.columns),
        protected_cols=list(protected_cols),
        preprocessor=prep,
        fitted_model=pr,
        group_categories=group_categories,
        feat_cols=feat_cols,
        added_dummy_col=added_dummy_col,
        threshold=0.5,
    )

    fair_model = FairModel(
        name=f"In: Fairness Regularization (Prejudice Remover, η={eta:g})",
        features=list(X_tr.columns),
        protected_cols=list(protected_cols),
        predictor=pr_predictor,
        threshold=0.5,
        outcome_col=None,
        positive_label=1,
        metadata={
            "source": "FairSelect",
            "technique": "In:Fairness Regularization (Prejudice Remover)",
            "model_name": model_name,
            "model_params": params,
            "eta": float(eta),
            "sensitive_attr": "sensitive",
            "aif360": True,
            "group_categories": group_categories,
            "feature_columns_after_preprocessing": feat_cols,
            "added_dummy_col": added_dummy_col,
        },
    )

    return evaluate_run(
        f"In: Fairness Regularization (Prejudice Remover, η={eta:g})",
        y_true,
        p,
        yhat,
        A_eval,
        fair_model=fair_model,
        test_index=X_te.index,
    )




def run_group_balanced_ensemble(model_name, params, K, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected_cols, all_df_train):
    """
    In-processing: group-balanced ensemble.

    Idea:
      - Build a preprocessor on train+val.
      - For k = 1..K:
          * Draw a group-balanced bootstrap sample of the training data.
          * Train a model on that sample.
          * Predict probabilities on the (shared) test set.
      - Average the K probability vectors to form ensemble predictions.
      - Classify at 0.5 threshold and evaluate.

    Group-balanced bootstrap:
      - group_balanced_bootstrap_indices produces indices so that each group
        is roughly equally represented in the sample, improving fairness
        robustness across groups.
    """
    #Build preprocessor on train+val
    prep = build_preprocessor(pd.concat([X_tr,X_va]), protected_cols)
    #Fit preprocessor on train+val
    Xt = prep.fit_transform(X_tr)
    #Transform test set
    Xt_te = prep.transform(X_te)
    #Convert group labels to numpy array for bootstrap 
    A_arr = A_tr.to_numpy()
    preds = [] #Store K prediction vectors

    #Train K models on group-balanced bootstraps
    for k in range(K):
        #Draw a group-balanced bootstrap sample from training data
        idx = group_balanced_bootstrap_indices(A_arr, size=len(A_arr))
        #Build estimator for the ensemble member
        clf = build_estimator(model_name, params)
        #Fit on the resampled training data
        clf.fit(Xt[idx], y_tr.to_numpy()[idx])
        #Predict probabilities on the full shared test set
        preds.append(to_proba(clf, Xt_te))
    #Average the K prediction vectors to form ensemble probabilities
    P = np.mean(np.vstack(preds), axis=0)
    yhat = (P >= 0.5).astype(int) #Hard predictions at 0.5 threshold
    ensemble_predictor = GroupBalancedEnsemblePredictor(
            features=list(X_tr.columns),
            protected_cols=list(protected_cols),
            preprocessor=prep,
            estimators=estimators,
            threshold=0.5,
        )

    fair_model = FairModel(
        name=f"In: Ensemble (K={K})",
        features=list(X_tr.columns),
        protected_cols=list(protected_cols),
        predictor=ensemble_predictor,
        threshold=0.5,
        outcome_col=None,
        positive_label=1,
        metadata={
            "source": "FairSelect",
            "technique": "In:Ensemble (K=5)",
            "model_name": model_name,
            "model_params": params,
            "K": int(K),
            "bootstrap": "group_balanced",
            "n_estimators": len(estimators),
        },
    )

    return evaluate_run(
        f"In: Ensemble (K={K})",
        y_te.to_numpy(),
        P,
        yhat,
        A_te,
        fair_model=fair_model,
        test_index=X_te.index,
    )

def run_multicalibration(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected_cols, all_df_train):
    '''
    In-processing: per-group multicalibration using isotonic regression.

    Steps:
      1. Train a base model on the training set (with a global preprocessor).
      2. On the validation set, compute predicted probabilities p_val.
      3. For each group, fit an isotonic regression model that maps p_val -> y_val.
      4. On the test set, compute base probabilities p_test, then adjust them
         via the per-group isotonic models to get p_adj.
      5. Threshold p_adj at 0.5 for hard predictions and evaluate.
    '''
    #Build preprocessor on train+val
    prep = build_preprocessor(pd.concat([X_tr,X_va]), protected_cols)
    #Fit preprocessor on train+val
    prep.fit(pd.concat([X_tr,X_va]))
    #Build base estimator
    clf = build_estimator(model_name, params)
    #Fit base model on transformed training data
    clf.fit(prep.transform(X_tr), y_tr)
    #Compute validation predicted probabilities
    p_val = to_proba(clf, prep.transform(X_va))
    #Fit per-group isotonic regression models on validation set
    iso_map = fit_isotonic_by_group(A_va, p_val, y_va.to_numpy())
    #Compute test predicted probabilities using base model
    p_test = to_proba(clf, prep.transform(X_te))
    #Apply per group isotonic calibration to adjust test probabilities
    p_adj  = apply_isotonic_by_group(A_te, p_test, iso_map)
    yhat   = (p_adj >= 0.5).astype(int) #Hard predictions at 0.5 threshold
    predictor = StandardPredictor(
        features=X_tr.columns,
        protected_cols=protected_cols,
        preprocessor=prep,
        estimator=clf,
        threshold=0.5,
        calibrators=iso_map,
    )

    fair_model = make_predictor_fair_model(
        name="In: Multicalibration (per-group isotonic)",
        features=X_tr.columns,
        protected_cols=protected_cols,
        predictor=predictor,
        threshold=0.5,
        outcome_col=None,
        metadata={
            "source": "FairSelect",
            "technique": "In:Multicalibration (isotonic)",
            "model_name": model_name,
            "model_params": params,
            "calibration": "per-group isotonic",
        },
    )

    return evaluate_run(
        "In: Multicalibration (per-group isotonic)",
        y_te.to_numpy(),
        p_adj,
        yhat,
        A_te,
        fair_model=fair_model,
        test_index=X_te.index,
    )

def run_reductions_meta(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected_cols, all_df_train, constraint="EO"):
    """
    In-processing: Fairlearn reductions (ExponentiatedGradient) wrapper.

    Parameters
    ----------
    constraint : {"EO", "DP"}
        Which fairness constraint to use:
          - "EO" : Equalized Odds
          - "DP" : Demographic Parity

    Workflow:
      1. If fairlearn is unavailable or the given model is not supported,
         fall back to the baseline.
      2. Build and fit a preprocessor on train+val.
      3. Instantiate a base estimator and wrap it in ExponentiatedGradient
         with the chosen fairness constraint.
      4. Fit on transformed training features with A_tr as sensitive_features.
      5. Predict on test, threshold at 0.5, and evaluate.
    """
    if not FAIRLEARN_OK or model_name not in ["Logistic Regression", "SVM", "Decision Tree"]:
        #we wrap a simple fallback
        return run_baseline(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected_cols, all_df_train)

    #Build preprocessor on train+val
    prep = build_preprocessor(pd.concat([X_tr,X_va]), protected_cols)
    #Fit preprocessor on train+val
    prep.fit(pd.concat([X_tr,X_va]))
    #Build base estimator
    base = build_estimator(model_name, params)
    #Choose fairness constraint
    cons = EqualizedOdds() if constraint=="EO" else DemographicParity()
    #Wrap base estimator in ExponentiatedGradient with the chosen constraint
    eg = ExponentiatedGradient(estimator=base, constraints=cons)
    #Fit on transformed training data with sensitive features
    eg.fit(prep.transform(X_tr), y_tr, sensitive_features=A_tr)
    #Predict probabilities on transformed test set
    p = to_proba(eg, prep.transform(X_te))
    yhat = (p >= 0.5).astype(int) #Hard predictions at 0.5 threshold
    fair_model = FairModel(
        name=f"In: Reductions ({constraint})",
        features=list(X_tr.columns),
        protected_cols=list(protected_cols),
        predictor=reductions_predictor,
        threshold=0.5,
        outcome_col=None,
        positive_label=1,
        metadata={
            "source": "FairSelect",
            "technique": f"In:Reductions ({constraint})",
            "model_name": model_name,
            "model_params": params,
            "constraint": constraint,
            "fairlearn": True,
            "inprocessing": "ExponentiatedGradient",
            "base_estimator": model_name,
        },
    )

    return evaluate_run(
        f"In: Reductions ({constraint})",
        y_te.to_numpy(),
        p,
        yhat,
        A_te,
        fair_model=fair_model,
        test_index=X_te.index,
    )

def run_baseline(model_name, params,
                 X_tr, X_va, X_te, y_tr, y_va, y_te,
                 A_tr, A_va, A_te, protected_cols, all_df_train) -> RunResult:
    """
    Baseline pipeline with *no fairness interventions*.

    Steps:
      1. Build a preprocessing + classifier Pipeline:
           - "prep": build_preprocessor on train+val (scaling + encoding).
           - "clf":  build_estimator (chosen model type and hyperparameters).
      2. Fit the pipeline on training data only.
      3. Compute probabilities on the test set.
      4. Threshold at 0.5 to get predictions.
      5. Evaluate via evaluate_run with name "Baseline".

    """
    from .deps import Pipeline
    #Create pipeline
    pipe = Pipeline(steps=[
        ("prep", build_preprocessor(pd.concat([X_tr, X_va]), protected_cols)),
        ("clf", build_estimator(model_name, params)),
    ])
    #Fit on training data
    pipe.fit(X_tr, y_tr)
    #Predict probabilities on test set
    p_test = to_proba(pipe.named_steps["clf"], pipe.named_steps["prep"].transform(X_te))
    yhat = (p_test >= 0.5).astype(int) #Hard predictions at 0.5 threshold
    fair_model = make_standard_fair_model(
        name="Baseline",
        features=list(X_tr.columns),
        protected_cols=list(protected_cols),
        preprocessor=pipe.named_steps["prep"],
        estimator=pipe.named_steps["clf"],
        threshold=0.5,
        metadata={
            "source": "FairSelect",
            "technique": "Baseline",
            "model_name": model_name,
            "model_params": params,
        },
    )

    return evaluate_run(
        "Baseline",
        y_te,
        p_test,
        yhat,
        A_te,
        fair_model=fair_model,
        test_index=X_te.index,
    )


def fit_isotonic_by_group(groups: pd.Series, p_val: np.ndarray, y_val: np.ndarray) -> Dict[str, IsotonicRegression]:
    '''
    Fit per-group isotonic regression models for calibration.(In-processing)

    For each group g:
      - we look at validation predictions p_val for that group,
      - and the corresponding true labels y_val,
      - we fit an IsotonicRegression model mapping scores → probabilities.

    This is used for multicalibration: each group gets its own calibration curve.


    Isotonic regressions help to smooth out a best fit line and gurantee a monotonic fit (entirely non-decreasing or non-increasing over the entire line)
    '''
    #Map of group labels to isotonic models
    models: Dict[str, IsotonicRegression] = {}

    #Iterate over each group
    for g in np.unique(groups):
        #Boolean mask to select only records from group g
        m = groups==g

        #We require at least 2 classes in the group (pos, neg) and at least 20 samples to fit the regression
        #Sample size requirement is arbitrary, may need to revisit down the line
        if m.sum() < 20 or len(np.unique(y_val[m]))<2:
            continue

        #Create the isotonic regression model (out_of_bounds="clip" will truncate any extreme values to the max or min seen during training)
        iso = IsotonicRegression(out_of_bounds="clip")
        #Fit the model using p_val (predicted score) and y_val (true labels)
        iso.fit(p_val[m], y_val[m])
        models[str(g)] = iso #Store back in the dict as a string
    return models

def apply_isotonic_by_group(groups: pd.Series, p: np.ndarray, group_iso: Dict[str, IsotonicRegression]) -> np.ndarray:
    """
    Apply per-group isotonic regression models to adjust predicted probabilities. (In-processing)

    Parameters
    ----------
    groups : pd.Series
        Group labels aligned with `p` (one label per instance).
    p : np.ndarray
        Original predicted probabilities/scores for each instance.
    group_iso : Dict[str, IsotonicRegression]
        Mapping from group label (as string) to fitted IsotonicRegression model.

    Returns
    -------
    np.ndarray
        Array of adjusted probabilities after group-specific calibration.
        Instances belonging to groups with no fitted model remain unchanged.
    """
    #Create a copy to avoid mutating the original predictions
    adj = p.copy()
    #iterate over each group and apply the corresponding isotonic regression model
    for g, iso in group_iso.items():
        #Boolean mask for instances in group g
        m = (groups==g)
        #Pass the original predictions through the isotonic regression model to get adjusted probabilities
        adj[m] = iso.predict(p[m]) #Should be better aproximated to align with empirical frequencies within each group
    return adj