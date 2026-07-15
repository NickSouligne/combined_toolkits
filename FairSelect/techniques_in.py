from typing import Any, Dict
import inspect
import warnings
import numpy as np
import pandas as pd
import sklearn
from .deps import FAIRLEARN_OK, ExponentiatedGradient, EqualizedOdds, DemographicParity, IsotonicRegression, AIF360_OK
from .core import build_estimator, build_preprocessor, evaluate_run, RunResult
from .utils import (
    to_proba, group_balanced_bootstrap_indices,
    fit_with_optional_sample_weight, ece_bin, confusion_rates,
)
from .techniques_pre import compute_reweights, local_massaging_fit_flip
from .FairModel_helper import GroupModelPredictor, make_standard_fair_model, StandardPredictor, make_predictor_fair_model, PrejudiceRemoverPredictor, GroupBalancedEnsemblePredictor
from FairModel import FairModel


def run_compositional_models(
    model_name,
    params,
    X_tr,
    X_va,
    X_te,
    y_tr,
    y_va,
    y_te,
    A_tr,
    A_va,
    A_te,
    protected_cols,
    all_df_train,
    outcome_col=None,
):
    """
    Train one model per intersectional group.

    A group-specific model is trained only when its training subset:
      1. contains at least min_group_train_size observations; and
      2. contains both outcome classes.

    Test observations whose groups do not have a valid group-specific
    model are evaluated using a pooled fallback model.
    """

    del all_df_train
    del A_va

    min_group_train_size = 5

    if not X_tr.index.equals(y_tr.index):
        y_tr = y_tr.reindex(X_tr.index)

    if not X_tr.index.equals(A_tr.index):
        A_tr = A_tr.reindex(X_tr.index)

    if not X_te.index.equals(y_te.index):
        y_te = y_te.reindex(X_te.index)

    if not X_te.index.equals(A_te.index):
        A_te = A_te.reindex(X_te.index)

    # Fit preprocessing on training data only
    prep = build_preprocessor(
        X_tr,
        protected_cols,
    )

    prep.fit(X_tr)

    Xtr_transformed = prep.transform(X_tr)
    Xte_transformed = prep.transform(X_te)

    # Always fit one pooled fallback model
    pooled_model = build_estimator(
        model_name,
        dict(params or {}),
    )

    pooled_model.fit(
        Xtr_transformed,
        y_tr.to_numpy(),
    )

    groups = pd.Series(
        A_tr.astype(str),
        index=A_tr.index,
    ).unique()

    models = {}
    skipped_groups = []

    for g in groups:
        mask = A_tr.astype(str) == str(g)

        group_indices = np.flatnonzero(
            mask.to_numpy()
        )

        group_y = y_tr.iloc[group_indices]
        group_n = len(group_indices)
        observed_classes = np.unique(
            group_y.to_numpy()
        )

        if group_n < min_group_train_size:
            skipped_groups.append({
                "group": str(g),
                "reason": "too_few_training_rows",
                "n_train": int(group_n),
                "classes": observed_classes.tolist(),
            })
            continue

        if len(observed_classes) < 2:
            skipped_groups.append({
                "group": str(g),
                "reason": "single_training_class",
                "n_train": int(group_n),
                "classes": observed_classes.tolist(),
            })
            continue

        group_model = build_estimator(
            model_name,
            dict(params or {}),
        )

        group_model.fit(
            Xtr_transformed[group_indices],
            group_y.to_numpy(),
        )

        models[str(g)] = group_model

    # Predict test probabilities
    probabilities = np.zeros(
        len(X_te),
        dtype=float,
    )

    A_test_string = A_te.astype(str).to_numpy()

    for group_name in np.unique(A_test_string):
        test_mask = A_test_string == group_name
        test_positions = np.flatnonzero(test_mask)

        selected_model = models.get(
            str(group_name),
            pooled_model,
        )

        probabilities[test_positions] = to_proba(
            selected_model,
            Xte_transformed[test_positions],
        )

    probabilities = np.clip(
        probabilities,
        0.0,
        1.0,
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    predictor = GroupModelPredictor(
        features=list(X_tr.columns),
        protected_cols=list(protected_cols),
        preprocessor=prep,
        group_models=models,
        fallback_model=pooled_model,
        threshold=0.5,
    )

    fair_model = make_predictor_fair_model(
        name="In: Compositional (per-group)",
        features=list(X_tr.columns),
        protected_cols=list(protected_cols),
        predictor=predictor,
        threshold=0.5,
        outcome_col=outcome_col,
        metadata={
            "source": "FairSelect",
            "technique": "In:Compositional per-group",
            "model_name": model_name,
            "model_params": dict(params or {}),
            "n_group_models": len(models),
            "trained_groups": sorted(models.keys()),
            "skipped_groups": skipped_groups,
            "fallback_model": "pooled",
            "min_group_train_size": min_group_train_size,
        },
    )

    notes = (
        f"Trained {len(models)} group-specific models. "
        f"Skipped {len(skipped_groups)} groups; observations from "
        "those groups use the pooled fallback model."
    )

    return evaluate_run(
        "In: Compositional (per-group)",
        y_te,
        probabilities,
        predictions,
        A_te,
        fair_model=fair_model,
        test_index=list(X_te.index),
        notes=notes,
    )

def run_prejudice_remover(model_name, params,
                          X_tr, X_va, X_te, y_tr, y_va, y_te,
                          A_tr, A_va, A_te,
                          protected_cols, all_df_train,
                          *, eta: float = 25.0, outcome_col = None):
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
        outcome_col=outcome_col,
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


def run_group_balanced_ensemble(
    model_name,
    params,
    K,
    X_tr,
    X_va,
    X_te,
    y_tr,
    y_va,
    y_te,
    A_tr,
    A_va,
    A_te,
    protected_cols,
    all_df_train,
    outcome_col=None,
):
    """
    Train an ensemble of K models using group-balanced bootstrap
    samples of the training set.

    Each fitted estimator is retained so the resulting FairModel
    can reproduce the ensemble probabilities during FairLogue
    auditing.
    """

    del all_df_train  # Not used by this runner
    del A_va          # Not used by this runner

    if K < 1:
        raise ValueError(
            f"K must be at least 1. Received K={K}."
        )

    if len(X_tr) != len(y_tr) or len(X_tr) != len(A_tr):
        raise ValueError(
            "Training inputs are not aligned: "
            f"len(X_tr)={len(X_tr)}, "
            f"len(y_tr)={len(y_tr)}, "
            f"len(A_tr)={len(A_tr)}."
        )

    if len(X_te) != len(y_te) or len(X_te) != len(A_te):
        raise ValueError(
            "Test inputs are not aligned: "
            f"len(X_te)={len(X_te)}, "
            f"len(y_te)={len(y_te)}, "
            f"len(A_te)={len(A_te)}."
        )

    # Fit preprocessing using training data only.
    # Using validation data to fit preprocessing is unnecessary
    # and can leak validation-set information.
    prep = build_preprocessor(
        X_tr,
        protected_cols,
    )

    X_train_transformed = prep.fit_transform(X_tr)
    X_test_transformed = prep.transform(X_te)

    y_train = np.asarray(
        pd.Series(y_tr).astype(int),
        dtype=int,
    ).ravel()

    group_train = (
        pd.Series(A_tr)
        .astype(str)
        .to_numpy()
    )

    # Store both:
    #   preds      -> test probability vector from each model
    #   estimators -> each fitted estimator used by FairModel
    preds = []
    estimators = []

    for k in range(int(K)):
        bootstrap_indices = group_balanced_bootstrap_indices(
            group_train,
            size=len(group_train),
        )

        bootstrap_indices = np.asarray(
            bootstrap_indices,
            dtype=int,
        )

        if bootstrap_indices.ndim != 1:
            bootstrap_indices = bootstrap_indices.ravel()

        if len(bootstrap_indices) == 0:
            raise RuntimeError(
                f"Bootstrap iteration {k + 1} returned no rows."
            )

        estimator = build_estimator(
            model_name,
            dict(params or {}),
        )

        estimator.fit(
            X_train_transformed[bootstrap_indices],
            y_train[bootstrap_indices],
        )

        model_probabilities = to_proba(
            estimator,
            X_test_transformed,
        )

        model_probabilities = np.asarray(
            model_probabilities,
            dtype=float,
        ).ravel()

        if len(model_probabilities) != len(X_te):
            raise RuntimeError(
                f"Ensemble member {k + 1} returned "
                f"{len(model_probabilities)} probabilities for "
                f"{len(X_te)} test observations."
            )

        preds.append(model_probabilities)

        # This was missing in the existing implementation.
        estimators.append(estimator)

    if not estimators:
        raise RuntimeError(
            "No ensemble estimators were successfully fitted."
        )

    probability_matrix = np.vstack(preds)

    P = np.mean(
        probability_matrix,
        axis=0,
    )

    P = np.clip(
        np.asarray(P, dtype=float).ravel(),
        0.0,
        1.0,
    )

    yhat = (P >= 0.5).astype(int)

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
        outcome_col=outcome_col,
        positive_label=1,
        metadata={
            "source": "FairSelect",
            "technique": f"In:Ensemble (K={K})",
            "model_name": model_name,
            "model_params": dict(params or {}),
            "K": int(K),
            "bootstrap": "group_balanced",
            "n_estimators": len(estimators),
        },
    )

    return evaluate_run(
        f"In: Ensemble (K={K})",
        y_te,
        P,
        yhat,
        A_te,
        fair_model=fair_model,
        test_index=list(X_te.index),
    )

def run_multicalibration(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected_cols, all_df_train, outcome_col = None):
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
        outcome_col=outcome_col,
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


class ReductionsPredictor:
    """
    FairModel-compatible wrapper for a fitted Fairlearn
    ExponentiatedGradient classifier.

    This wrapper:

    1. Accepts raw pandas DataFrames.
    2. Selects the same feature columns used during training.
    3. Applies the fitted FairSelect preprocessor.
    4. Computes the expected positive-class probability from
       the ExponentiatedGradient mixture.
    5. Produces deterministic hard predictions at a threshold.

    Notes
    -----
    ExponentiatedGradient.predict() is randomized. For model
    evaluation and FairLogue auditing, this wrapper instead uses
    the weighted expected prediction across predictors_ so that
    repeated audits are reproducible.
    """

    def __init__(
        self,
        *,
        features,
        protected_cols,
        preprocessor,
        estimator,
        threshold=0.5,
        outcome_col=None,
    ):
        self.features = list(features)
        self.protected_cols = list(protected_cols)
        self.preprocessor = preprocessor
        self.estimator = estimator
        self.threshold = float(threshold)

    def _prepare_raw_features(self, X):
        """
        Return a DataFrame containing the same raw feature columns
        used to train the FairSelect preprocessor.
        """
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_array = np.asarray(X)

            if X_array.ndim == 1:
                X_array = X_array.reshape(1, -1)

            if X_array.shape[1] != len(self.features):
                raise ValueError(
                    "ReductionsPredictor received an array with "
                    f"{X_array.shape[1]} columns, but "
                    f"{len(self.features)} features were expected."
                )

            X_df = pd.DataFrame(
                X_array,
                columns=self.features,
            )

        missing_features = [
            feature
            for feature in self.features
            if feature not in X_df.columns
        ]

        if missing_features:
            raise ValueError(
                "The input data are missing features required by "
                f"the reductions model: {missing_features}"
            )

        return X_df[self.features].copy()

    def _transform(self, X):
        X_df = self._prepare_raw_features(X)
        return self.preprocessor.transform(X_df)

    @staticmethod
    def _positive_class_output(predictor, X_transformed):
        """
        Obtain a positive-class score from one fitted base learner.

        Most ExponentiatedGradient base learners expose predict(),
        whose binary 0/1 output can be treated as the positive-class
        contribution to the mixture. If predict_proba() is available,
        use it to preserve continuous probabilities.
        """
        if hasattr(predictor, "predict_proba"):
            probabilities = np.asarray(
                predictor.predict_proba(X_transformed),
                dtype=float,
            )

            if probabilities.ndim == 2:
                if probabilities.shape[1] == 1:
                    return probabilities[:, 0]

                classes = getattr(
                    predictor,
                    "classes_",
                    None,
                )

                if classes is not None:
                    classes = np.asarray(classes)

                    positive_locations = np.flatnonzero(
                        classes == 1
                    )

                    if len(positive_locations) == 1:
                        return probabilities[
                            :,
                            positive_locations[0],
                        ]

                return probabilities[:, -1]

            return probabilities.ravel()

        predictions = np.asarray(
            predictor.predict(X_transformed),
            dtype=float,
        ).ravel()

        return predictions

    def predict_proba(self, X):
        """
        Return P(Y=1) as a one-dimensional array.

        The returned score is the weighted average of positive-class
        outputs from all fitted ExponentiatedGradient predictors.
        """
        X_transformed = self._transform(X)

        fitted_predictors = getattr(
            self.estimator,
            "predictors_",
            None,
        )

        mixture_weights = getattr(
            self.estimator,
            "weights_",
            None,
        )

        if fitted_predictors is None or mixture_weights is None:
            # Compatibility fallback for Fairlearn versions exposing
            # a probability-mass prediction method.
            if hasattr(self.estimator, "_pmf_predict"):
                probability_mass = np.asarray(
                    self.estimator._pmf_predict(
                        X_transformed
                    ),
                    dtype=float,
                )

                if (
                    probability_mass.ndim == 2
                    and probability_mass.shape[1] >= 2
                ):
                    return np.clip(
                        probability_mass[:, 1],
                        0.0,
                        1.0,
                    )

                return np.clip(
                    probability_mass.ravel(),
                    0.0,
                    1.0,
                )

            # Last-resort fallback. This can be randomized in
            # Fairlearn, so it is less desirable than predictors_.
            predictions = np.asarray(
                self.estimator.predict(X_transformed),
                dtype=float,
            ).ravel()

            return np.clip(predictions, 0.0, 1.0)

        predictors = list(fitted_predictors)

        if isinstance(mixture_weights, pd.Series):
            weights = mixture_weights.to_numpy(dtype=float)
        else:
            weights = np.asarray(
                mixture_weights,
                dtype=float,
            ).ravel()

        if len(predictors) != len(weights):
            raise RuntimeError(
                "ExponentiatedGradient predictors_ and weights_ "
                "have different lengths: "
                f"{len(predictors)} versus {len(weights)}."
            )

        valid = np.isfinite(weights) & (weights > 0)

        if not valid.any():
            raise RuntimeError(
                "ExponentiatedGradient did not produce any "
                "positive finite mixture weights."
            )

        valid_predictors = [
            predictor
            for predictor, keep in zip(predictors, valid)
            if keep
        ]

        valid_weights = weights[valid]
        valid_weights = valid_weights / valid_weights.sum()

        component_predictions = np.column_stack([
            self._positive_class_output(
                predictor,
                X_transformed,
            )
            for predictor in valid_predictors
        ])

        probabilities = component_predictions @ valid_weights

        return np.clip(
            np.asarray(probabilities, dtype=float).ravel(),
            0.0,
            1.0,
        )

    def predict(self, X):
        probabilities = self.predict_proba(X)

        return (
            probabilities >= self.threshold
        ).astype(int)

def _supports_sample_weight(estimator) -> bool:
    """
    Determine whether estimator.fit() explicitly accepts
    sample_weight or accepts arbitrary keyword arguments.
    """
    try:
        signature = inspect.signature(estimator.fit)
    except (TypeError, ValueError):
        return False

    parameters = signature.parameters

    if "sample_weight" in parameters:
        return True

    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _prepare_reductions_params(
    model_name,
    params,
):
    """
    Remove estimator-level weighting parameters that would otherwise
    compound the observation weights generated by Fairlearn.
    """
    reductions_params = dict(params or {})

    # Fairlearn generates its own observation weights during each
    # cost-sensitive classification iteration.
    reductions_params.pop("sample_weight", None)

    if model_name in {
        "Logistic Regression",
        "Random Forest",
    }:
        reductions_params["class_weight"] = None

    if model_name == "XGBoost":
        reductions_params.pop("scale_pos_weight", None)

    if model_name == "LightGBM":
        reductions_params.pop("is_unbalance", None)
        reductions_params.pop("scale_pos_weight", None)
        reductions_params.pop("class_weight", None)

    return reductions_params


def _make_constraint(constraint):
    normalized_constraint = str(constraint).strip().upper()

    if normalized_constraint in {
        "EO",
        "EQUALIZED_ODDS",
        "EQUALIZED ODDS",
    }:
        return "EO", EqualizedOdds()

    if normalized_constraint in {
        "DP",
        "DEMOGRAPHIC_PARITY",
        "DEMOGRAPHIC PARITY",
    }:
        return "DP", DemographicParity()

    raise ValueError(
        "constraint must be one of: 'EO', 'Equalized Odds', "
        "'DP', or 'Demographic Parity'. "
        f"Received {constraint!r}."
    )
def run_reductions_meta(
    model_name,
    params,
    X_tr,
    X_va,
    X_te,
    y_tr,
    y_va,
    y_te,
    A_tr,
    A_va,
    A_te,
    protected_cols,
    all_df_train,
    constraint="EO",
    outcome_col=None,
):
    """
    Run Fairlearn ExponentiatedGradient using any FairSelect
    estimator whose fit() method supports sample_weight.

    Supported FairSelect model names
    --------------------------------
    - Logistic Regression
    - Random Forest
    - Decision Tree
    - Neural Network
    - SVM
    - XGBoost
    - LightGBM

    Actual compatibility is checked from the constructed estimator,
    rather than inferred only from the model name.

    Parameters
    ----------
    model_name
        FairSelect model name.

    params
        Hyperparameters selected for the base estimator.

    X_tr, X_va, X_te
        Training, validation, and test feature DataFrames.

    y_tr, y_va, y_te
        Binary outcome labels.

    A_tr, A_va, A_te
        Intersectional protected-group labels.

    protected_cols
        Original protected-characteristic column names.

    all_df_train
        Retained for compatibility with the other FairSelect runners.

    constraint : {"EO", "DP"}
        "EO" applies Equalized Odds.
        "DP" applies Demographic Parity.

    Returns
    -------
    RunResult
        FairSelect evaluation result with an attached FairModel.
    """

    del all_df_train  # Unused in this runner

    if not FAIRLEARN_OK:
        raise ImportError(
            "Fairlearn is unavailable. ExponentiatedGradient "
            "cannot be run. Install a compatible Fairlearn version "
            "rather than silently returning a baseline result."
        )

    allowed_model_names = {
        "Logistic Regression",
        "Random Forest",
        "Decision Tree",
        "Neural Network",
        "SVM",
        "XGBoost",
        "LightGBM",
    }

    if model_name not in allowed_model_names:
        raise ValueError(
            f"Unknown FairSelect model name {model_name!r}. "
            "Expected one of: "
            f"{sorted(allowed_model_names)}"
        )

    constraint_name, fairness_constraint = (
        _make_constraint(constraint)
    )

    # ---------------------------------------------------------
    # Validate alignment before fitting
    # ---------------------------------------------------------
    if len(X_tr) != len(y_tr) or len(X_tr) != len(A_tr):
        raise ValueError(
            "Training data are not aligned: "
            f"len(X_tr)={len(X_tr)}, "
            f"len(y_tr)={len(y_tr)}, "
            f"len(A_tr)={len(A_tr)}."
        )

    if len(X_te) != len(y_te) or len(X_te) != len(A_te):
        raise ValueError(
            "Test data are not aligned: "
            f"len(X_te)={len(X_te)}, "
            f"len(y_te)={len(y_te)}, "
            f"len(A_te)={len(A_te)}."
        )

    y_train = np.asarray(
        pd.Series(y_tr).astype(int),
        dtype=int,
    ).ravel()

    y_test = np.asarray(
        pd.Series(y_te).astype(int),
        dtype=int,
    ).ravel()

    sensitive_train = (
        pd.Series(A_tr)
        .astype(str)
        .to_numpy()
    )

    sensitive_test = (
        pd.Series(A_te)
        .astype(str)
        .to_numpy()
    )

    unique_outcomes = np.unique(y_train)

    if not set(unique_outcomes).issubset({0, 1}):
        raise ValueError(
            "ExponentiatedGradient binary classification requires "
            "training labels encoded as 0 and 1. "
            f"Observed labels: {unique_outcomes.tolist()}"
        )

    if len(unique_outcomes) < 2:
        raise ValueError(
            "ExponentiatedGradient cannot be fit because the "
            "training outcome contains only one class."
        )

    unique_groups = np.unique(sensitive_train)

    if len(unique_groups) < 2:
        raise ValueError(
            "ExponentiatedGradient requires at least two observed "
            "protected groups. "
            f"Observed groups: {unique_groups.tolist()}"
        )

    # ---------------------------------------------------------
    # Fit preprocessing on training data only
    # ---------------------------------------------------------
    #
    # Do not fit the preprocessor on train + validation. Fitting on
    # training alone avoids learning scaling or encoding information
    # from validation observations.
    prep = build_preprocessor(
        X_tr,
        list(protected_cols),
    )

    X_train_transformed = prep.fit_transform(X_tr)
    X_test_transformed = prep.transform(X_te)

    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = (
            X_train_transformed.toarray()
        )

    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = (
            X_test_transformed.toarray()
        )

    X_train_transformed = np.asarray(
        X_train_transformed,
        dtype=float,
    )

    X_test_transformed = np.asarray(
        X_test_transformed,
        dtype=float,
    )

    if not np.isfinite(X_train_transformed).all():
        bad_count = int(
            (~np.isfinite(X_train_transformed)).sum()
        )

        raise ValueError(
            "The transformed training matrix contains "
            f"{bad_count} non-finite values. Add imputation to "
            "build_preprocessor() before running reductions."
        )

    if not np.isfinite(X_test_transformed).all():
        bad_count = int(
            (~np.isfinite(X_test_transformed)).sum()
        )

        raise ValueError(
            "The transformed test matrix contains "
            f"{bad_count} non-finite values."
        )

    # ---------------------------------------------------------
    # Construct the base learner
    # ---------------------------------------------------------
    reductions_params = _prepare_reductions_params(
        model_name=model_name,
        params=params,
    )
        # eps controls the allowed constraint violation. Retain the
    # Fairlearn default unless explicitly supplied in params.
    reductions_eps = float(
        reductions_params.pop(
            "reductions_eps",
            0.01,
        )
    )

    reductions_max_iter = int(
        reductions_params.pop(
            "reductions_max_iter",
            50,
        )
    )

    reductions_nu = reductions_params.pop(
        "reductions_nu",
        None,
    )

    base_estimator = build_estimator(
        model_name,
        reductions_params,
    )

    if not _supports_sample_weight(base_estimator):
        additional_note = ""

        if model_name == "Neural Network":
            additional_note = (
                " Scikit-learn added sample_weight support to "
                "MLPClassifier in version 1.7. Upgrade "
                "scikit-learn or use a weighted neural-network "
                "wrapper."
            )

        raise TypeError(
            f"{model_name} cannot be used with "
            "ExponentiatedGradient in this environment because "
            f"{type(base_estimator).__name__}.fit() does not "
            f"accept sample_weight.{additional_note}"
        )

    # ---------------------------------------------------------
    # Fit Exponentiated Gradient
    # ---------------------------------------------------------
    #


    exponentiated_gradient_kwargs = {
        "estimator": base_estimator,
        "constraints": fairness_constraint,
        "eps": reductions_eps,
        "max_iter": reductions_max_iter,
    }

    if reductions_nu is not None:
        exponentiated_gradient_kwargs["nu"] = float(
            reductions_nu
        )

    reductions_estimator = ExponentiatedGradient(
        **exponentiated_gradient_kwargs
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*lbfgs failed to converge.*",
        )

        reductions_estimator.fit(
            X_train_transformed,
            y_train,
            sensitive_features=sensitive_train,
        )

    # ---------------------------------------------------------
    # Build a deterministic expected-mixture predictor
    # ---------------------------------------------------------
    reductions_predictor = ReductionsPredictor(
        features=list(X_tr.columns),
        protected_cols=list(protected_cols),
        preprocessor=prep,
        estimator=reductions_estimator,
        threshold=0.5,
    )

    # Use the same wrapper for immediate test evaluation and for
    # subsequent FairLogue auditing.
    probabilities = reductions_predictor.predict_proba(
        X_te
    )

    predictions = reductions_predictor.predict(
        X_te
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    ).ravel()

    predictions = np.asarray(
        predictions,
        dtype=int,
    ).ravel()

    if len(probabilities) != len(y_test):
        raise RuntimeError(
            "The reductions probability vector is not aligned "
            "with the held-out labels: "
            f"{len(probabilities)} predictions versus "
            f"{len(y_test)} labels."
        )

    if len(predictions) != len(y_test):
        raise RuntimeError(
            "The reductions hard-prediction vector is not aligned "
            "with the held-out labels."
        )

    # ---------------------------------------------------------
    # Capture useful Fairlearn diagnostics
    # ---------------------------------------------------------
    mixture_weights = getattr(
        reductions_estimator,
        "weights_",
        None,
    )

    if isinstance(mixture_weights, pd.Series):
        mixture_weights_metadata = {
            str(key): float(value)
            for key, value
            in mixture_weights.items()
            if np.isfinite(value)
        }
    elif mixture_weights is not None:
        weights_array = np.asarray(
            mixture_weights,
            dtype=float,
        ).ravel()

        mixture_weights_metadata = {
            str(index): float(value)
            for index, value in enumerate(weights_array)
            if np.isfinite(value)
        }
    else:
        mixture_weights_metadata = None

    n_mixture_predictors = len(
        getattr(
            reductions_estimator,
            "predictors_",
            [],
        )
    )

    best_gap = getattr(
        reductions_estimator,
        "best_gap_",
        None,
    )

    last_iter = getattr(
        reductions_estimator,
        "last_iter_",
        None,
    )

    fair_model = FairModel(
        name=f"In: Reductions ({constraint_name})",
        features=list(X_tr.columns),
        protected_cols=list(protected_cols),
        predictor=reductions_predictor,
        threshold=0.5,
        outcome_col=outcome_col,
        positive_label=1,
        metadata={
            "source": "FairSelect",
            "technique": (
                f"In:Reductions ({constraint_name})"
            ),
            "model_name": model_name,
            "model_params_original": dict(params or {}),
            "model_params_reductions": reductions_params,
            "constraint": constraint_name,
            "fairlearn": True,
            "inprocessing": "ExponentiatedGradient",
            "base_estimator": type(
                base_estimator
            ).__name__,
            "eps": reductions_eps,
            "max_iter": reductions_max_iter,
            "nu": reductions_nu,
            "n_mixture_predictors": (
                int(n_mixture_predictors)
            ),
            "mixture_weights": (
                mixture_weights_metadata
            ),
            "best_gap": (
                float(best_gap)
                if best_gap is not None
                else None
            ),
            "last_iter": (
                int(last_iter)
                if last_iter is not None
                else None
            ),
            "prediction_mode": (
                "deterministic expected mixture"
            ),
            "sklearn_version": sklearn.__version__,
        },
    )

    return evaluate_run(
        f"In: Reductions ({constraint_name})",
        y_te,
        probabilities,
        predictions,
        pd.Series(
            sensitive_test,
            index=getattr(A_te, "index", None),
        ),
        fair_model=fair_model,
        test_index=list(X_te.index),
        notes=(
            "ExponentiatedGradient evaluated using the "
            "deterministic weighted expected prediction across "
            f"{n_mixture_predictors} fitted base learners."
        ),
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