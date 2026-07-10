import numpy as np
import pandas as pd
from ..FairModel import FairModel
from .utils import input_repair_standardize_by_group, to_proba, apply_multiaccuracy_boost

def make_standard_fair_model(
    *,
    name,
    features,
    protected_cols,
    preprocessor,
    estimator,
    threshold=0.5,
    outcome_col=None,
    metadata=None,
):
    """
    For runs whose final prediction rule is:
        raw df -> preprocessor -> estimator -> threshold

    This works for:
        - Baseline
        - Reweighting
        - SMOTE/Oversampling
        - Local Massaging
        - many simple single-estimator models
    """

    return FairModel(
        name=name,
        features=list(features),
        protected_cols=list(protected_cols),
        preprocessor=preprocessor,
        estimator=estimator,
        threshold=threshold,
        outcome_col=outcome_col,
        metadata=metadata or {},
    )

def make_predictor_fair_model(
    *,
    name,
    features,
    protected_cols,
    predictor,
    threshold=0.5,
    outcome_col=None,
    metadata=None,
):
    return FairModel(
        name=name,
        features=list(features),
        protected_cols=list(protected_cols),
        predictor=predictor,
        threshold=threshold,
        outcome_col=outcome_col,
        metadata=metadata or {},
    )

class StandardPredictor:
    """
    Prediction wrapper for:
        raw df -> preprocessor -> estimator -> optional calibrator/postprocessor
    """

    def __init__(
        self,
        *,
        features,
        protected_cols,
        preprocessor,
        estimator,
        threshold=0.5,
        group_thresholds=None,
        calibrators=None,
        postprocessor=None,
    ):
        self.features = list(features)
        self.protected_cols = list(protected_cols)
        self.preprocessor = preprocessor
        self.estimator = estimator
        self.threshold = threshold
        self.group_thresholds = group_thresholds or {}
        self.calibrators = calibrators or {}
        self.postprocessor = postprocessor

    def make_group(self, df):
        return df[self.protected_cols].astype(str).agg("|".join, axis=1)

    def predict_proba(self, df):
        X = df[self.features].copy()
        Xt = self.preprocessor.transform(X)
        p = to_proba(self.estimator, Xt)

        groups = self.make_group(df)

        if self.calibrators:
            p_adj = p.copy()
            for g in pd.Series(groups).unique():
                key = str(g)
                m = groups.astype(str) == key
                if key in self.calibrators:
                    p_adj[m] = self.calibrators[key].predict(p[m].reshape(-1, 1))
            p = p_adj

        if self.postprocessor is not None and hasattr(self.postprocessor, "predict_proba"):
            p = self.postprocessor.predict_proba(df, p, groups)

        return np.asarray(p, dtype=float)

    def predict(self, df):
        p = self.predict_proba(df)
        groups = self.make_group(df)

        yhat = np.zeros_like(p, dtype=int)

        for g in pd.Series(groups).unique():
            key = str(g)
            t = self.group_thresholds.get(key, self.threshold)
            m = groups.astype(str) == key
            yhat[m] = (p[m] >= t).astype(int)

        if self.postprocessor is not None and hasattr(self.postprocessor, "predict"):
            yhat = self.postprocessor.predict(df, p, yhat, groups)

        return yhat
    


class GroupModelPredictor:
    def __init__(
        self,
        *,
        features,
        protected_cols,
        preprocessor,
        group_models,
        fallback_model,
        threshold=0.5,
    ):
        self.features = list(features)
        self.protected_cols = list(protected_cols)
        self.preprocessor = preprocessor
        self.group_models = group_models
        self.fallback_model = fallback_model
        self.threshold = threshold

    def make_group(self, df):
        return df[self.protected_cols].astype(str).agg("|".join, axis=1)

    def predict_proba(self, df):
        groups = self.make_group(df)
        p = np.zeros(len(df), dtype=float)

        for i, g in enumerate(groups.astype(str)):
            model = self.group_models.get(g, self.fallback_model)
            X_i = df[self.features].iloc[[i]].copy()
            Xt_i = self.preprocessor.transform(X_i)
            p[i] = to_proba(model, Xt_i)[0]

        return p

    def predict(self, df):
        p = self.predict_proba(df)
        return (p >= self.threshold).astype(int)
    


class PrejudiceRemoverPredictor:
    """
    FairModel-compatible predictor for AIF360 PrejudiceRemover.

    FairModel / FairLogue only need:
        predict_proba(df)
        predict(df)

    This wrapper handles:
        raw dataframe -> FairSelect preprocessor -> AIF360 BinaryLabelDataset -> pr.predict()
    """

    def __init__(
        self,
        *,
        features,
        protected_cols,
        preprocessor,
        fitted_model,
        group_categories,
        feat_cols,
        added_dummy_col=False,
        threshold=0.5,
    ):
        self.features = list(features)
        self.protected_cols = list(protected_cols)
        self.preprocessor = preprocessor
        self.fitted_model = fitted_model
        self.group_categories = list(group_categories)
        self.feat_cols = list(feat_cols)
        self.added_dummy_col = bool(added_dummy_col)
        self.threshold = threshold

    def make_group(self, df):
        return (
            df[self.protected_cols]
            .astype(str)
            .agg("|".join, axis=1)
        )

    def _transform_features(self, df):
        X = df[self.features].copy()
        Xt = self.preprocessor.transform(X)

        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()

        Xt = np.asarray(Xt, dtype=float)

        if Xt.ndim == 1:
            Xt = Xt.reshape(-1, 1)

        if self.added_dummy_col:
            Xt = np.column_stack([
                Xt,
                np.zeros((Xt.shape[0], 1), dtype=float),
            ])

        return Xt

    def _make_binary_label_dataset(self, df):
        from aif360.datasets import BinaryLabelDataset

        Xt = self._transform_features(df)

        if Xt.shape[1] != len(self.feat_cols):
            raise ValueError(
                f"PrejudiceRemover feature mismatch: transformed data has "
                f"{Xt.shape[1]} columns, expected {len(self.feat_cols)}."
            )

        out = pd.DataFrame(Xt, columns=self.feat_cols)

        groups = self.make_group(df)
        sens = pd.Categorical(
            groups.astype(str),
            categories=self.group_categories,
        ).codes

        if np.any(sens < 0):
            unseen = sorted(groups.astype(str)[sens < 0].unique())
            raise ValueError(
                f"Prediction data contains groups not seen during "
                f"PrejudiceRemover training: {unseen}"
            )

        out["sensitive"] = sens.astype(float)

        # AIF360 BinaryLabelDataset requires a label column even at prediction time.
        if "Y" in df.columns:
            out["label"] = pd.Series(df["Y"]).astype(float).to_numpy()
        else:
            out["label"] = 0.0

        out = out.replace([np.inf, -np.inf], np.nan).fillna(0)

        return BinaryLabelDataset(
            df=out,
            label_names=["label"],
            protected_attribute_names=["sensitive"],
            favorable_label=1.0,
            unfavorable_label=0.0,
        )

    def predict_proba(self, df):
        d = self._make_binary_label_dataset(df)
        pred = self.fitted_model.predict(d)

        if getattr(pred, "scores", None) is not None:
            p = np.asarray(pred.scores, dtype=float).ravel()
            return np.clip(p, 0.0, 1.0)

        p = np.asarray(pred.labels, dtype=float).ravel()

        if p.min() < 0:
            p = (p > 0).astype(float)

        return np.clip(p, 0.0, 1.0)

    def predict(self, df):
        p = self.predict_proba(df)
        return (p >= self.threshold).astype(int)
    

class GroupBalancedEnsemblePredictor:
    """
    FairModel-compatible predictor for the group-balanced ensemble.

    Prediction rule:
        raw df
        -> fitted FairSelect preprocessor
        -> each fitted ensemble member predicts probability
        -> average probabilities
        -> threshold at 0.5
    """

    def __init__(
        self,
        *,
        features,
        protected_cols,
        preprocessor,
        estimators,
        threshold=0.5,
    ):
        self.features = list(features)
        self.protected_cols = list(protected_cols)
        self.preprocessor = preprocessor
        self.estimators = list(estimators)
        self.threshold = threshold

    def predict_proba(self, df):
        X = df[self.features].copy()
        Xt = self.preprocessor.transform(X)

        preds = [
            to_proba(est, Xt)
            for est in self.estimators
        ]

        return np.mean(np.vstack(preds), axis=0)

    def predict(self, df):
        p = self.predict_proba(df)
        return (p >= self.threshold).astype(int)
    

class ReductionsMetaPredictor:
    """
    FairModel-compatible predictor for Fairlearn ExponentiatedGradient.

    Prediction rule:
        raw df
        -> fitted FairSelect preprocessor
        -> fitted ExponentiatedGradient
        -> probability-like score
        -> threshold
    """

    def __init__(
        self,
        *,
        features,
        protected_cols,
        preprocessor,
        fitted_model,
        threshold=0.5,
    ):
        self.features = list(features)
        self.protected_cols = list(protected_cols)
        self.preprocessor = preprocessor
        self.fitted_model = fitted_model
        self.threshold = threshold

    def predict_proba(self, df):
        X = df[self.features].copy()
        Xt = self.preprocessor.transform(X)

        # Fairlearn ExponentiatedGradient often exposes _pmf_predict.
        # This returns class probabilities for the randomized classifier.
        if hasattr(self.fitted_model, "_pmf_predict"):
            pmf = self.fitted_model._pmf_predict(Xt)

            if getattr(pmf, "ndim", None) == 2 and pmf.shape[1] >= 2:
                return np.asarray(pmf[:, 1], dtype=float)

        # Fallback to shared toolkit probability converter.
        try:
            return np.asarray(to_proba(self.fitted_model, Xt), dtype=float)
        except Exception:
            # Final fallback: hard predictions as probability-like scores.
            yhat = self.fitted_model.predict(Xt, random_state=0)
            return np.asarray(yhat, dtype=float)

    def predict(self, df):
        X = df[self.features].copy()
        Xt = self.preprocessor.transform(X)

        try:
            yhat = self.fitted_model.predict(Xt, random_state=0)
            return np.asarray(yhat, dtype=int)
        except TypeError:
            p = self.predict_proba(df)
            return (p >= self.threshold).astype(int)
        

class MultiaccuracyBoostPredictor:
    """
    FairModel-compatible predictor for Multiaccuracy Boost.

    Prediction rule:
        raw df
        -> fitted FairSelect preprocessor
        -> fitted base estimator
        -> base probability
        -> apply learned multiaccuracy adjustment behavior
        -> threshold

    For this current integration, the predictor reuses the validation audit set
    stored during the original run and applies the same boost procedure to new
    dataframes passed by FairLogue Component 3.
    """

    def __init__(
        self,
        *,
        features,
        protected_cols,
        preprocessor,
        estimator,
        X_val,
        y_val,
        A_val,
        p_val,
        threshold=0.5,
        alpha=0.02,
        eta=None,
        max_iters=25,
        auditor_type="ridge",
        random_state=0,
        include_group_in_auditor=True,
    ):
        self.features = list(features)
        self.protected_cols = list(protected_cols)
        self.preprocessor = preprocessor
        self.estimator = estimator

        self.X_val = X_val.copy()
        self.y_val = y_val.copy()
        self.A_val = A_val.copy()
        self.p_val = np.asarray(p_val, dtype=float)

        self.threshold = threshold
        self.alpha = alpha
        self.eta = eta
        self.max_iters = max_iters
        self.auditor_type = auditor_type
        self.random_state = random_state
        self.include_group_in_auditor = include_group_in_auditor

    def make_group(self, df):
        return (
            df[self.protected_cols]
            .astype(str)
            .agg("|".join, axis=1)
        )

    def predict_base_proba(self, df):
        X = df[self.features].copy()
        Xt = self.preprocessor.transform(X)
        return to_proba(self.estimator, Xt)

    def predict_proba(self, df):
        X_new = df[self.features].copy()
        A_new = self.make_group(df)
        p_new = self.predict_base_proba(df)

        p_adj = apply_multiaccuracy_boost(
            X_va=self.X_val,
            X_te=X_new,
            y_va=self.y_val,
            A_va=self.A_val,
            A_te=A_new,
            p_val=self.p_val,
            p_test=p_new,
            prep=self.preprocessor,
            alpha=self.alpha,
            eta=self.eta,
            max_iters=self.max_iters,
            auditor_type=self.auditor_type,
            random_state=self.random_state,
            include_group_in_auditor=self.include_group_in_auditor,
        )

        return np.asarray(p_adj, dtype=float)

    def predict(self, df):
        p = self.predict_proba(df)
        return (p >= self.threshold).astype(int)
    

class RejectOptionShiftPredictor:
    """
    FairModel-compatible predictor for simple reject-option threshold shift.

    Prediction rule:
        raw df
        -> fitted FairSelect preprocessor
        -> fitted base estimator
        -> base probability
        -> group-specific shifted threshold
    """

    def __init__(
        self,
        *,
        features,
        protected_cols,
        preprocessor,
        estimator,
        thresholds,
        default_threshold=0.5,
    ):
        self.features = list(features)
        self.protected_cols = list(protected_cols)
        self.preprocessor = preprocessor
        self.estimator = estimator
        self.thresholds = {str(k): float(v) for k, v in thresholds.items()}
        self.default_threshold = float(default_threshold)

    def make_group(self, df):
        return (
            df[self.protected_cols]
            .astype(str)
            .agg("|".join, axis=1)
        )

    def predict_proba(self, df):
        X = df[self.features].copy()
        Xt = self.preprocessor.transform(X)
        return np.asarray(to_proba(self.estimator, Xt), dtype=float)

    def predict(self, df):
        p = self.predict_proba(df)
        groups = self.make_group(df).astype(str)

        yhat = np.zeros_like(p, dtype=int)

        for g in pd.Series(groups).unique():
            t = self.thresholds.get(str(g), self.default_threshold)
            m = groups == str(g)
            yhat[m] = (p[m] >= t).astype(int)

        return yhat
    
class InputRepairPredictor:
    """
    FairModel-compatible predictor for Post: Input Repair.

    Prediction rule:
        raw df
        -> infer intersectional group from protected columns
        -> repair input features using train/validation reference data
        -> fitted FairSelect preprocessor
        -> fitted base estimator
        -> probability / threshold
    """

    def __init__(
        self,
        *,
        features,
        protected_cols,
        preprocessor,
        estimator,
        X_repair_reference,
        A_repair_reference,
        threshold=0.5,
    ):
        self.features = list(features)
        self.protected_cols = list(protected_cols)
        self.preprocessor = preprocessor
        self.estimator = estimator

        self.X_repair_reference = X_repair_reference.copy()
        self.A_repair_reference = A_repair_reference.copy()

        self.threshold = float(threshold)

    def make_group(self, df):
        return (
            df[self.protected_cols]
            .astype(str)
            .agg("|".join, axis=1)
        )

    def repair_inputs(self, df):
        X_new = df[self.features].copy()
        A_new = self.make_group(df)

        X_rep = input_repair_standardize_by_group(
            self.X_repair_reference,
            X_new,
            self.A_repair_reference,
            A_new,
        )

        return X_rep

    def predict_proba(self, df):
        X_rep = self.repair_inputs(df)
        Xt = self.preprocessor.transform(X_rep)
        return np.asarray(to_proba(self.estimator, Xt), dtype=float)

    def predict(self, df):
        p = self.predict_proba(df)
        return (p >= self.threshold).astype(int)
    

class KamiranRejectOptionPredictor:
    """
    FairModel-compatible predictor for Kamiran-style Reject Option Classification.

    Prediction rule:
        raw df
        -> fitted FairSelect preprocessor
        -> fitted base estimator
        -> base probability
        -> reject-option decision rule using theta and unprivileged groups
    """

    def __init__(
        self,
        *,
        features,
        protected_cols,
        preprocessor,
        estimator,
        unprivileged_values,
        theta,
        base_threshold=0.5,
    ):
        self.features = list(features)
        self.protected_cols = list(protected_cols)
        self.preprocessor = preprocessor
        self.estimator = estimator
        self.unprivileged_values = {str(v) for v in unprivileged_values}
        self.theta = float(theta)
        self.base_threshold = float(base_threshold)

    def make_group(self, df):
        return (
            df[self.protected_cols]
            .astype(str)
            .agg("|".join, axis=1)
        )

    def predict_proba(self, df):
        X = df[self.features].copy()
        Xt = self.preprocessor.transform(X)
        return np.asarray(to_proba(self.estimator, Xt), dtype=float)

    def predict(self, df):
        p = self.predict_proba(df)
        A = self.make_group(df).astype(str).to_numpy()

        yhat = (p >= self.base_threshold).astype(int)

        # critical region: max(p, 1-p) <= theta
        # equivalent to p in [1-theta, theta]
        in_critical = np.maximum(p, 1.0 - p) <= self.theta

        unpriv = np.isin(A, list(self.unprivileged_values))
        priv = ~unpriv

        yhat[in_critical & unpriv] = 1
        yhat[in_critical & priv] = 0

        return yhat
    

class CombinedPipelinePredictor:
    """
    FairModel-compatible predictor for FairSelect combined pipelines.

    This object preserves the final fitted prediction behavior from:
        PRE  -> IN -> optional calibration -> optional POST

    It supports:
        - standard pooled estimator
        - Fairlearn reductions-like estimator
        - group-balanced ensemble
        - compositional per-group models
        - optional input repair
        - optional multicalibration
        - optional multiaccuracy boost
        - optional group-specific thresholds
    """

    def __init__(
        self,
        *,
        features,
        protected_cols,
        preprocessor,
        mode,
        threshold=0.5,
        estimator=None,
        ensemble_estimators=None,
        group_models=None,
        fallback_estimator=None,
        calibrators=None,
        group_thresholds=None,
        input_repair=False,
        X_repair_reference=None,
        A_repair_reference=None,
        multiaccuracy=False,
        X_val=None,
        y_val=None,
        A_val=None,
        p_val=None,
        multiaccuracy_params=None,
    ):
        self.features = list(features)
        self.protected_cols = list(protected_cols)
        self.preprocessor = preprocessor
        self.mode = mode
        self.threshold = float(threshold)

        self.estimator = estimator
        self.ensemble_estimators = list(ensemble_estimators or [])
        self.group_models = group_models or {}
        self.fallback_estimator = fallback_estimator

        self.calibrators = calibrators or {}
        self.group_thresholds = {
            str(k): float(v) for k, v in (group_thresholds or {}).items()
        }

        self.input_repair = bool(input_repair)
        self.X_repair_reference = (
            X_repair_reference.copy()
            if X_repair_reference is not None
            else None
        )
        self.A_repair_reference = (
            A_repair_reference.copy()
            if A_repair_reference is not None
            else None
        )

        self.multiaccuracy = bool(multiaccuracy)
        self.X_val = X_val.copy() if X_val is not None else None
        self.y_val = y_val.copy() if y_val is not None else None
        self.A_val = A_val.copy() if A_val is not None else None
        self.p_val = None if p_val is None else np.asarray(p_val, dtype=float)

        self.multiaccuracy_params = multiaccuracy_params or {}

    def make_group(self, df):
        return (
            df[self.protected_cols]
            .astype(str)
            .agg("|".join, axis=1)
        )

    def _prepare_X(self, df):
        X = df[self.features].copy()

        if self.input_repair:
            if self.X_repair_reference is None or self.A_repair_reference is None:
                raise ValueError(
                    "Input repair was requested, but repair reference data "
                    "was not stored in the CombinedPipelinePredictor."
                )

            A_new = self.make_group(df)

            X = input_repair_standardize_by_group(
                self.X_repair_reference,
                X,
                self.A_repair_reference,
                A_new,
            )

        return X

    def _predict_base_proba(self, df):
        X = self._prepare_X(df)
        Xt = self.preprocessor.transform(X)

        if self.mode in {"standard", "reductions"}:
            if self.estimator is None:
                raise ValueError("No estimator stored for standard/reductions mode.")

            if self.mode == "reductions" and hasattr(self.estimator, "_pmf_predict"):
                pmf = self.estimator._pmf_predict(Xt)
                if getattr(pmf, "ndim", None) == 2 and pmf.shape[1] >= 2:
                    return np.asarray(pmf[:, 1], dtype=float)

            return np.asarray(to_proba(self.estimator, Xt), dtype=float)

        if self.mode == "ensemble":
            if not self.ensemble_estimators:
                raise ValueError("No ensemble estimators stored.")

            preds = [
                to_proba(est, Xt)
                for est in self.ensemble_estimators
            ]

            return np.mean(np.vstack(preds), axis=0)

        if self.mode == "compositional":
            groups = self.make_group(df).astype(str).to_numpy()
            p = np.zeros(len(df), dtype=float)

            for i, g in enumerate(groups):
                est = self.group_models.get(str(g), self.fallback_estimator)

                if est is None:
                    raise ValueError(
                        f"No group model or fallback estimator available for group {g}."
                    )

                p[i] = to_proba(est, Xt[i:i + 1])[0]

            return p

        raise ValueError(f"Unknown combined predictor mode: {self.mode}")

    def _apply_calibration(self, df, p):
        if not self.calibrators:
            return np.asarray(p, dtype=float)

        groups = self.make_group(df).astype(str)
        p_adj = np.asarray(p, dtype=float).copy()

        for g in pd.Series(groups).unique():
            key = str(g)
            m = groups == key

            if key in self.calibrators:
                p_adj[m] = self.calibrators[key].predict(p_adj[m])

        return np.asarray(p_adj, dtype=float)

    def _apply_multiaccuracy(self, df, p):
        if not self.multiaccuracy:
            return np.asarray(p, dtype=float)

        if self.X_val is None or self.y_val is None or self.A_val is None or self.p_val is None:
            raise ValueError(
                "Multiaccuracy was requested, but validation audit data "
                "was not stored in the CombinedPipelinePredictor."
            )

        X_new = df[self.features].copy()

        if self.input_repair:
            # The current run_combined_pipeline applies input repair before
            # multiaccuracy only to p_test, while apply_multiaccuracy_boost
            # still receives raw X_te. We mirror that behavior here by passing
            # raw X_new to apply_multiaccuracy_boost.
            X_for_boost = X_new
        else:
            X_for_boost = X_new

        A_new = self.make_group(df)

        params = {
            "alpha": 0.02,
            "eta": None,
            "max_iters": 25,
            "auditor_type": "ridge",
            "random_state": 0,
            "eps": 1e-6,
            "include_group_in_auditor": True,
        }
        params.update(self.multiaccuracy_params)

        p_adj = apply_multiaccuracy_boost(
            X_va=self.X_val,
            X_te=X_for_boost,
            y_va=self.y_val,
            A_va=self.A_val,
            A_te=A_new,
            p_val=self.p_val,
            p_test=p,
            prep=self.preprocessor,
            **params,
        )

        return np.asarray(p_adj, dtype=float)

    def predict_proba(self, df):
        p = self._predict_base_proba(df)
        p = self._apply_calibration(df, p)
        p = self._apply_multiaccuracy(df, p)
        return np.asarray(p, dtype=float)

    def predict(self, df):
        p = self.predict_proba(df)
        groups = self.make_group(df).astype(str)

        if self.group_thresholds:
            yhat = np.zeros_like(p, dtype=int)

            for g in pd.Series(groups).unique():
                t = self.group_thresholds.get(str(g), self.threshold)
                m = groups == str(g)
                yhat[m] = (p[m] >= t).astype(int)

            return yhat

        return (p >= self.threshold).astype(int)