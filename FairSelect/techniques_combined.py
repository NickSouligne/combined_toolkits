from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from .core import build_estimator, build_preprocessor, evaluate_run, RunResult
from .utils import (
    to_proba, fit_with_optional_sample_weight,
    group_balanced_bootstrap_indices, confusion_rates,
    ece_bin, youden_threshold,
)
from .techniques_pre import compute_reweights, local_massaging_fit_flip
from .techniques_post import (
    group_thresholds_youden, predict_with_group_thresholds,
    input_repair_standardize_by_group, apply_multiaccuracy_boost
)
from .deps import IMBLEARN_OK, FAIRLEARN_OK, LogisticRegression
from .techniques_in import run_prejudice_remover, fit_isotonic_by_group, apply_isotonic_by_group
from .FairModel_helper import make_standard_fair_model, CombinedPipelinePredictor
from FairModel import FairModel



def run_combined_pipeline(model_name, params,
                          X_tr, X_va, X_te, y_tr, y_va, y_te,
                          A_tr, A_va, A_te, protected_cols, all_df_train, outcome_col,
                          selected: Dict[str, bool]) -> RunResult:
    """
    Compose selected techniques into one run in this order:
      PRE  : Local Massaging -> SMOTE/Oversample -> Reweight (y,a)
      IN   : (choose at most one) {Reductions(EO), Compositional, Ensemble(K=5), Prejudice Remover}
             + optional Multicalibration (isotonic)
      POST : Input Repair -> Multiaccuracy Boost -> Youden per group -> Reject-Option Shift

    Returns a single RunResult named with the chain.
    """

    #Canonical keys (must match self.tech_vars keys exactly)
    PRE_KEYS  = ["Pre:Local Massaging", "Pre:SMOTE / Oversample", "Pre:Reweight (y,a)"]
    IN_TRAIN  = ["In:Reductions (EO)", "In:Compositional per-group", "In:Ensemble (K=5)", "In:Fairness Regularization (Prejudice Remover)"]
    IN_CAL    = "In:Multicalibration (isotonic)"
    POST_KEYS = ["Post:Input Repair", "Post:Multiaccuracy Boost", "Post:Youden per group", "Post:Reject-Option Shift"]

    #Execution plans based on selections that preserve order
    pre_plan  = [k for k in PRE_KEYS  if selected.get(k, False)]
    in_train  = [k for k in IN_TRAIN  if selected.get(k, False)][:1]  #at most one trainer
    use_mcal  = bool(selected.get(IN_CAL, False))
    post_plan = [k for k in POST_KEYS if selected.get(k, False)]

    #Readable title
    parts = [k.split(":",1)[1].strip() for k in (pre_plan + in_train)]
    if use_mcal: parts.append("Multicalibration (isotonic)")
    parts += [k.split(":",1)[1].strip() for k in post_plan]
    title = "Combined: " + (" -> ".join(parts) if parts else "(no techniques)")

    #Working copies
    Xtr, Xva, Xte = X_tr.copy(), X_va.copy(), X_te.copy() #Features
    ytr, yva, yte = y_tr.copy(), y_va.copy(), y_te.copy() #Labels
    Atr, Ava, Ate = A_tr.copy(), A_va.copy(), A_te.copy() #Group labels (intersectional)

    technique_random_state = int(
        params.get(
            "random_state",
            42,
        )
    )

    #---------- PRE ----------
    #We train on either: (a) preprocessed + SMOTE matrix, or (b) normal pipeline.

    #Flags indicating if we are using SMOTE or reweighting
    did_smote = False; sample_weight = None

    #Local Massaging
    #Relabels training examples to equalize positive rate across groups based on initial scores.
    if "Pre:Local Massaging" in pre_plan:
        #Build a temporary preprocessor on train/validation to approximate actual pipeline
        prep_tmp = build_preprocessor(pd.concat([Xtr, Xva]), protected_cols)

        #Fit the temporary preprocessor and concatenate the data for scoring
        Xt_tmp = prep_tmp.fit_transform(pd.concat([Xtr, Xva]))

        #Create an estimator for a temporary baseline model to get scores on training data
        est_tmp = build_estimator(model_name, params)

        #Fit on training portion only
        est_tmp.fit(Xt_tmp[:len(Xtr)], ytr)

        #Compute predicted scores on the training data
        scores_tr = to_proba(est_tmp, Xt_tmp[:len(Xtr)])

        #Apply local massaging relabeling, flipping and overwriting labels in ytr  
        ytr = pd.Series(local_massaging_fit_flip(ytr, scores_tr, Atr), index=ytr.index)

    #SMOTE / Oversample (class-balance on transformed space)
    #Flags indicating if class balances were applied
    Xt_for_fit = None; y_for_fit = None

    if "Pre:SMOTE / Oversample" in pre_plan:
        #Build a temporary preprocessor on train/validation to approximate actual pipeline
        prep_tmp = build_preprocessor(pd.concat([Xtr, Xva]), protected_cols)
        Xt_tr = prep_tmp.fit_transform(Xtr)
        yt_tr = ytr.to_numpy()

        #If imblearn is available, use SMOTE to balance classes (TODO: Reimplement manual SMOTE)
        if IMBLEARN_OK:
            try:
                from imblearn.over_sampling import SMOTE
                sampler = SMOTE(
                    random_state=technique_random_state,
                )
                #Fit SMOTE to training data, and generate balanced dataset by oversampling minority class
                Xt_bal, yt_bal = sampler.fit_resample(Xt_tr, yt_tr)
                #Get SMOTE flag and variables
                did_smote, Xt_for_fit, y_for_fit = True, Xt_bal, yt_bal
            except Exception:
                did_smote = False
        #If SMOTE unavailable, train normally (or rely on sample_weight if set)

    #Reweight (y,a) 
    #Kamiran-Calders style reweighting based on joint (y,a) distribution
    if "Pre:Reweight (y,a)" in pre_plan:
        #Computes sample weights for each training instance based on:
        # w(y,a) = P(y) * P(a) / P(y,a)
        sample_weight = compute_reweights(ytr, Atr)

    #---------- IN (train classifier once) ----------
    #Preprocessor for (train+val) – used for inference consistently
    prep = build_preprocessor(pd.concat([Xtr, Xva]), protected_cols)
    prep.fit(pd.concat([Xtr, Xva]))

    #Training matrix (based off whether SMOTE was performed)
    if did_smote:
        Xfit, yfit = Xt_for_fit, y_for_fit
    else:
        Xfit, yfit = prep.transform(Xtr), ytr.to_numpy()


    trained_est = None #Contains the trained estimator if applicable
    P_val = None  #validation probabilities (needed by many post steps)
    p_test = None
    ensemble_estimators = None
    predictor_mode = "standard"
    group_models_for_fairmodel = None
    fallback_estimator_for_fairmodel = None
    calibrators_for_fairmodel = None
    group_thresholds_for_fairmodel = None

    input_repair_used = False
    X_repair_reference_for_fairmodel = None
    A_repair_reference_for_fairmodel = None

    multiaccuracy_used = False
    multiaccuracy_params_for_fairmodel = None

    #Helper fn: Given a trained estimator, score val and test sets
    def _score_val_and_test(est_like):
        nonlocal P_val
        P_val = to_proba(est_like, prep.transform(Xva))
        p_test = to_proba(est_like, prep.transform(Xte))
        return p_test

    #Check if an in-processing method was selected
    if in_train:
        #Only choose the first in-processing method (TODO: Fix GUI to prevent this, or extend functionality to multiple?)
        choice = in_train[0]

        #Reductions on Equalized Odds using fairlearn
        if choice == "In:Reductions (EO)" and FAIRLEARN_OK:
            from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
            #Build base estimator without fairness constraints
            base = build_estimator(model_name, params)
            #Wrap the estimator in a ExponentiatedGradient with EO constraints (modifies the gradient of the steps during training to enforce EO)
            eg = ExponentiatedGradient(
                estimator=base,
                constraints=EqualizedOdds()
            )
            #Fit the training data using group labels as senstive features
            eg.fit(Xfit, yfit, sensitive_features=Atr)
            trained_est = eg
            #Score val and test sets
            p_test = _score_val_and_test(trained_est)
            predictor_mode = "reductions"

        #Compositional training: one model per group, fallback to pooled
        elif choice == "In:Compositional per-group":
            """
            Train one model per eligible intersectional group.

            Groups that are too small or contain only one outcome class use a
            single pooled fallback model. Any sample weights produced by an
            earlier preprocessing step, such as outcome-group reweighting, are
            retained for both pooled and group-specific fitting.
            """

            min_group_train_size = 5

            # ---------------------------------------------------------
            # Transform train, validation, and test data
            # ---------------------------------------------------------
            Xtr_t = prep.transform(Xtr)
            Xva_t = prep.transform(Xva)
            Xte_t = prep.transform(Xte)

            ytr_array = np.asarray(
                ytr,
                dtype=int,
            ).ravel()

            Atr_array = (
                pd.Series(Atr)
                .astype(str)
                .to_numpy()
            )

            Ava_array = (
                pd.Series(Ava)
                .astype(str)
                .to_numpy()
            )

            Ate_array = (
                pd.Series(Ate)
                .astype(str)
                .to_numpy()
            )

            if len(Xtr_t) != len(ytr_array):
                raise ValueError(
                    "Compositional training data are misaligned: "
                    f"Xtr_t has {len(Xtr_t)} rows but ytr has "
                    f"{len(ytr_array)} rows."
                )

            if len(Atr_array) != len(ytr_array):
                raise ValueError(
                    "Compositional protected-group labels are misaligned: "
                    f"Atr has {len(Atr_array)} rows but ytr has "
                    f"{len(ytr_array)} rows."
                )

            if np.unique(ytr_array).size < 2:
                raise ValueError(
                    "The pooled compositional model cannot be trained because "
                    "the full training set contains only one outcome class."
                )

            # ---------------------------------------------------------
            # Normalize optional sample weights
            # ---------------------------------------------------------
            if sample_weight is None:
                training_weights = None
            else:
                training_weights = np.asarray(
                    sample_weight,
                    dtype=float,
                ).ravel()

                if len(training_weights) != len(ytr_array):
                    raise ValueError(
                        "Compositional sample weights are misaligned: "
                        f"received {len(training_weights)} weights for "
                        f"{len(ytr_array)} training observations."
                    )

            # ---------------------------------------------------------
            # Fit one pooled fallback model once
            # ---------------------------------------------------------
            #
            # Do not refit this model inside the test prediction loop.
            pooled = build_estimator(
                model_name,
                dict(params or {}),
            )

            fit_with_optional_sample_weight(
                pooled,
                Xtr_t,
                ytr_array,
                sample_weight=training_weights,
            )

            # ---------------------------------------------------------
            # Fit eligible group-specific models
            # ---------------------------------------------------------
            models = {}
            compositional_group_summary = []
            skipped_compositional_groups = []

            for group_name in pd.unique(Atr_array):
                group_positions = np.flatnonzero(
                    Atr_array == str(group_name)
                )

                group_y = ytr_array[group_positions]

                observed_classes, class_counts = np.unique(
                    group_y,
                    return_counts=True,
                )

                class_count_dict = {
                    str(int(class_label)): int(class_count)
                    for class_label, class_count in zip(
                        observed_classes,
                        class_counts,
                    )
                }

                group_record = {
                    "group": str(group_name),
                    "n_train": int(len(group_positions)),
                    "n_classes": int(len(observed_classes)),
                    "classes": observed_classes.tolist(),
                    "class_counts": class_count_dict,
                    "n_negative": int((group_y == 0).sum()),
                    "n_positive": int((group_y == 1).sum()),
                }

                # A separate model is not useful for extremely small groups.
                if len(group_positions) < min_group_train_size:
                    skip_record = {
                        **group_record,
                        "status": "skipped",
                        "reason": "too_few_training_rows",
                    }

                    skipped_compositional_groups.append(skip_record)
                    compositional_group_summary.append(skip_record)

                    print(
                        "[Combined Compositional] Using pooled fallback for "
                        f"group={group_name!r}: n_train="
                        f"{len(group_positions)} is below "
                        f"min_group_train_size={min_group_train_size}."
                    )

                    continue

                # Binary classifiers such as LogisticRegression cannot fit a
                # training subset containing only one class.
                if len(observed_classes) < 2:
                    skip_record = {
                        **group_record,
                        "status": "skipped",
                        "reason": "single_training_class",
                    }

                    skipped_compositional_groups.append(skip_record)
                    compositional_group_summary.append(skip_record)

                    print(
                        "[Combined Compositional] Using pooled fallback for "
                        f"group={group_name!r}: classes="
                        f"{observed_classes.tolist()}, class_counts="
                        f"{class_count_dict}."
                    )

                    continue

                group_estimator = build_estimator(
                    model_name,
                    dict(params or {}),
                )

                group_weights = None

                if training_weights is not None:
                    group_weights = training_weights[
                        group_positions
                    ]

                try:
                    fit_with_optional_sample_weight(
                        group_estimator,
                        Xtr_t[group_positions],
                        group_y,
                        sample_weight=group_weights,
                    )

                except ValueError as exc:
                    error_text = str(exc).lower()

                    class_error_phrases = (
                        "at least 2 classes",
                        "at least two classes",
                        "only one class",
                        "contains only one class",
                    )

                    if any(
                        phrase in error_text
                        for phrase in class_error_phrases
                    ):
                        skip_record = {
                            **group_record,
                            "status": "skipped",
                            "reason": (
                                "estimator_rejected_group_classes"
                            ),
                            "error": str(exc),
                        }

                        skipped_compositional_groups.append(
                            skip_record
                        )
                        compositional_group_summary.append(
                            skip_record
                        )

                        print(
                            "[Combined Compositional] Estimator rejected "
                            f"group={group_name!r}; using pooled fallback. "
                            f"Error: {exc}"
                        )

                        continue

                    raise

                models[str(group_name)] = group_estimator

                compositional_group_summary.append({
                    **group_record,
                    "status": "trained",
                    "reason": None,
                })

            # ---------------------------------------------------------
            # Helper for group-aware probability generation
            # ---------------------------------------------------------
            def _predict_compositional_probabilities(
                X_transformed,
                group_values,
            ):
                group_values = (
                    pd.Series(group_values)
                    .astype(str)
                    .to_numpy()
                )

                probabilities = np.full(
                    len(group_values),
                    np.nan,
                    dtype=float,
                )

                prediction_sources = np.empty(
                    len(group_values),
                    dtype=object,
                )

                for group_name in pd.unique(group_values):
                    positions = np.flatnonzero(
                        group_values == str(group_name)
                    )

                    estimator = models.get(
                        str(group_name),
                        pooled,
                    )

                    source = (
                        "group_specific"
                        if str(group_name) in models
                        else "pooled_fallback"
                    )

                    group_probabilities = to_proba(
                        estimator,
                        X_transformed[positions],
                    )

                    group_probabilities = np.asarray(
                        group_probabilities,
                        dtype=float,
                    )

                    if group_probabilities.ndim == 2:
                        if group_probabilities.shape[1] >= 2:
                            group_probabilities = (
                                group_probabilities[:, 1]
                            )
                        else:
                            group_probabilities = (
                                group_probabilities[:, 0]
                            )

                    group_probabilities = (
                        group_probabilities.ravel()
                    )

                    if len(group_probabilities) != len(positions):
                        raise RuntimeError(
                            "The compositional estimator returned an "
                            "unexpected number of probabilities for "
                            f"group={group_name!r}: expected "
                            f"{len(positions)}, received "
                            f"{len(group_probabilities)}."
                        )

                    probabilities[positions] = (
                        group_probabilities
                    )

                    prediction_sources[positions] = source

                if np.isnan(probabilities).any():
                    missing_positions = np.flatnonzero(
                        np.isnan(probabilities)
                    )

                    raise RuntimeError(
                        "Compositional prediction failed to assign "
                        f"probabilities to {len(missing_positions)} "
                        "observations. First missing positions: "
                        f"{missing_positions[:20].tolist()}."
                    )

                return (
                    np.clip(probabilities, 0.0, 1.0),
                    prediction_sources,
                )

            # ---------------------------------------------------------
            # Generate validation probabilities
            # ---------------------------------------------------------
            #
            # These must use the same compositional decision rule as the test
            # probabilities because later post-processing techniques may be
            # fitted or tuned using validation predictions.
            P_val, compositional_val_sources = (
                _predict_compositional_probabilities(
                    Xva_t,
                    Ava_array,
                )
            )

            # ---------------------------------------------------------
            # Generate test probabilities
            # ---------------------------------------------------------
            p_test, compositional_test_sources = (
                _predict_compositional_probabilities(
                    Xte_t,
                    Ate_array,
                )
            )

            # ---------------------------------------------------------
            # Preserve state for later post-processing and FairModel creation
            # ---------------------------------------------------------
            #
            # The pooled model remains useful for generic rescoring and as the
            # fallback for groups without a valid group-specific estimator.
            trained_est = pooled

            predictor_mode = "compositional"

            group_models_for_fairmodel = models

            fallback_estimator_for_fairmodel = pooled

            # Store diagnostics so they can be added to FairModel metadata or
            # the final RunResult notes later in run_combined_pipeline().
            compositional_metadata = {
                "min_group_train_size": int(
                    min_group_train_size
                ),
                "n_training_groups": int(
                    len(pd.unique(Atr_array))
                ),
                "n_group_models": int(
                    len(models)
                ),
                "trained_groups": sorted(
                    models.keys()
                ),
                "n_skipped_groups": int(
                    len(skipped_compositional_groups)
                ),
                "skipped_groups": (
                    skipped_compositional_groups
                ),
                "group_training_summary": (
                    compositional_group_summary
                ),
                "fallback_model": "pooled",
                "sample_weight_used": (
                    training_weights is not None
                ),
                "validation_prediction_source_counts": {
                    str(source): int(count)
                    for source, count in (
                        pd.Series(
                            compositional_val_sources
                        )
                        .value_counts(dropna=False)
                        .items()
                    )
                },
                "test_prediction_source_counts": {
                    str(source): int(count)
                    for source, count in (
                        pd.Series(
                            compositional_test_sources
                        )
                        .value_counts(dropna=False)
                        .items()
                    )
                },
            }

        #Ensemble of K=5 group-balanced bootstrapped models
        elif choice == "In:Ensemble (K=5)":
            K = 5 #Number of bootstrap samples (TODO: Allow user to specific K)
            preds_test, preds_val = [], []
            ensemble_estimators = []  #Store individual estimators if we want to analyze them later (TODO: Integrate into RunResult)
            #Transform datasets with standard preprocessor
            Xte_t = prep.transform(Xte)
            Xva_t = prep.transform(Xva)
            for _ in range(K):
                #Build and fit general estimator
                est = build_estimator(model_name, params)
                #Draw a bootstrap sample that is roughly group-balanced
                idx = group_balanced_bootstrap_indices(Atr.to_numpy(), size=len(Atr))
                #Fit the estimator on the bootstrap sample
                fit_with_optional_sample_weight(est, Xfit[idx], yfit[idx], sample_weight=None)
                #Score test and validation probabilities for this ensemble
                preds_test.append(to_proba(est, Xte_t))
                preds_val.append(to_proba(est, Xva_t))
                #Store the individual estimator
                ensemble_estimators.append(est)
            #Aggregate predictions by averaging across K models
            p_test = np.mean(np.vstack(preds_test), axis=0)
            P_val  = np.mean(np.vstack(preds_val),  axis=0)
            trained_est = None  #ensemble isn't a single estimator; keep None
            predictor_mode = "ensemble"

        #Prejudice Remover (AIF360)
        elif choice == "In:Fairness Regularization (Prejudice Remover)":
            rr = run_prejudice_remover(
                model_name, params,
                Xtr, Xva, Xte,
                ytr, yva, yte,
                Atr, Ava, Ate,
                protected_cols,
                pd.concat([Xtr, Xva]),
                eta=25.0,
            )

            if rr.fair_model is not None:
                rr.fair_model.name = title
                rr.fair_model.metadata.update({
                    "source": "FairSelect",
                    "technique": "Combined",
                    "title": title,
                    "pre_plan": pre_plan,
                    "in_train": in_train,
                    "use_multicalibration": bool(use_mcal),
                    "post_plan": post_plan,
                    "note": (
                        "Combined pipeline currently delegates PrejudiceRemover "
                        "and returns early before later post-processing."
                    ),
                })

            return rr

        else:
            #Unknown or unavailable, use vanilla fit
            est = build_estimator(model_name, params)
            fit_with_optional_sample_weight(est, Xfit, yfit, sample_weight=sample_weight)
            trained_est = est
            p_test = _score_val_and_test(trained_est)

    else:
        #No special in-process trainer, use base model
        est = build_estimator(model_name, params)
        fit_with_optional_sample_weight(est, Xfit, yfit, sample_weight=sample_weight)
        trained_est = est
        p_test = _score_val_and_test(trained_est)

    #Optional in-process calibration layer (per-group isotonic)
    if use_mcal and P_val is not None:
        iso_map = fit_isotonic_by_group(Ava, P_val, yva.to_numpy())
        p_test = apply_isotonic_by_group(Ate, p_test, iso_map)
        calibrators_for_fairmodel = iso_map
    
    if p_test is None:
        est = build_estimator(model_name, params)
        fit_with_optional_sample_weight(
            est,
            Xfit,
            yfit,
            sample_weight=sample_weight,
        )
        trained_est = est
        p_test = _score_val_and_test(trained_est)

    #---------- POST ----------


    #Input Repair (rescore with repaired X_test)
    #Modifies test features group wise to align distributions most closely with training distribution
    if "Post:Input Repair" in post_plan:
        A_train_all = pd.concat([Atr, Ava], axis=0)
        X_repair_reference = pd.concat([Xtr, Xva], axis=0)

        X_rep = input_repair_standardize_by_group(
            X_repair_reference,
            Xte,
            A_train_all,
            Ate,
        )

        if trained_est is not None:
            p_test = to_proba(trained_est, prep.transform(X_rep))

        elif ensemble_estimators is not None and len(ensemble_estimators) > 0:
            X_rep_t = prep.transform(X_rep)

            repaired_preds = [
                to_proba(est, X_rep_t)
                for est in ensemble_estimators
            ]

            p_test = np.mean(np.vstack(repaired_preds), axis=0)

        else:
            print(
                "[Input Repair] No trained estimator or ensemble estimators available; "
                "keeping existing p_test unchanged."
            )

        input_repair_used = True
        X_repair_reference_for_fairmodel = X_repair_reference
        A_repair_reference_for_fairmodel = A_train_all

    #Multiaccuracy Boost (residual model on validation) -- Based on Kim 2018
    #Fits a residual model on validation data to boost probabilities using group wise signals, then applies to test data
    if "Post:Multiaccuracy Boost" in post_plan and P_val is not None:
        multiaccuracy_params_for_fairmodel = {
            "alpha": 0.02,
            "eta": None,
            "max_iters": 25,
            "auditor_type": "ridge",
            "random_state": 0,
            "eps": 1e-6,
            "include_group_in_auditor": True,
        }

        p_test = apply_multiaccuracy_boost(
            X_va=Xva,
            X_te=Xte,
            y_va=yva,
            A_va=Ava,
            A_te=Ate,
            p_val=P_val,
            p_test=p_test,
            prep=prep,
            **multiaccuracy_params_for_fairmodel,
        )

        multiaccuracy_used = True

    #Youden per group (threshold learning)
    #Learns optimal per-group thresholds on validation data to maximize Youden's J statistic
    yhat = (p_test >= 0.5).astype(int) #Global threshold 0.5 by default
    if "Post:Youden per group" in post_plan and P_val is not None:
        th = group_thresholds_youden(Ava, yva.to_numpy(), P_val)
        yhat = predict_with_group_thresholds(Ate, p_test, th, default=0.5)
        group_thresholds_for_fairmodel = th

    #Reject-Option Shift (based on validation TPRs at 0.5)
    #Slightly shifts thresholds in favor of group with lowest TPR on validation and against group with highest TPR
    if "Post:Reject-Option Shift" in post_plan and P_val is not None:
        yhat_val = (P_val >= 0.5).astype(int)
        groups = pd.Series(Ava).unique()
        tprs = {}

        for g in groups:
            m = (Ava == g).to_numpy()
            cr = confusion_rates(yva.to_numpy()[m], yhat_val[m])
            tprs[str(g)] = cr["TPR"]

        worst = min(tprs, key=lambda k: (-1 if np.isnan(tprs[k]) else tprs[k]))
        best = max(tprs, key=lambda k: (-1 if np.isnan(tprs[k]) else tprs[k]))

        th = {str(g): 0.5 for g in groups}

        th[str(worst)] = max(0.0, th[str(worst)] - 0.05)
        th[str(best)] = min(1.0, th[str(best)] + 0.05)

        yhat = predict_with_group_thresholds(Ate, p_test, th, default=0.5)
        group_thresholds_for_fairmodel = th

    #---------- Evaluate ----------
    combined_predictor = CombinedPipelinePredictor(
        features=list(Xtr.columns),
        protected_cols=list(protected_cols),
        preprocessor=prep,
        mode=predictor_mode,
        threshold=0.5,

        estimator=trained_est,
        ensemble_estimators=ensemble_estimators,
        group_models=group_models_for_fairmodel,
        fallback_estimator=fallback_estimator_for_fairmodel,

        calibrators=calibrators_for_fairmodel,
        group_thresholds=group_thresholds_for_fairmodel,

        input_repair=input_repair_used,
        X_repair_reference=X_repair_reference_for_fairmodel,
        A_repair_reference=A_repair_reference_for_fairmodel,

        multiaccuracy=multiaccuracy_used,
        X_val=Xva,
        y_val=yva,
        A_val=Ava,
        p_val=P_val,
        multiaccuracy_params=multiaccuracy_params_for_fairmodel,
    )

    fair_model = FairModel(
        name=title,
        features=list(Xtr.columns),
        protected_cols=list(protected_cols),
        predictor=combined_predictor,
        threshold=0.5,
        group_thresholds=group_thresholds_for_fairmodel or {},
        calibrators=calibrators_for_fairmodel or {},
        outcome_col=outcome_col,
        positive_label=1,
        metadata={
            "source": "FairSelect",
            "technique": "Combined",
            "title": title,
            "model_name": model_name,
            "model_params": params,
            "pre_plan": pre_plan,
            "in_train": in_train,
            "use_multicalibration": bool(use_mcal),
            "post_plan": post_plan,
            "predictor_mode": predictor_mode,
            "did_smote": bool(did_smote),
            "used_reweighting": "Pre:Reweight (y,a)" in pre_plan,
            "used_local_massaging": "Pre:Local Massaging" in pre_plan,
            "input_repair_used": bool(input_repair_used),
            "multiaccuracy_used": bool(multiaccuracy_used),
            "group_thresholds": group_thresholds_for_fairmodel,
        },
    )

    return evaluate_run(
        title,
        yte.to_numpy(),
        p_test,
        yhat,
        Ate,
        fair_model=fair_model,
        test_index=X_te.index,
    )



