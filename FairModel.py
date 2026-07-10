#### Goal is to integrate both FairLogue and FairSelect into a single re-usable pipeline
#### To do this, we need to create a way of sharing model objects between the toolkits

#### This class will then create a single model object that can be used by both FairLogue and FairSelect

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

class FairModel:
    def __init__(
        self,
        name,
        features,
        protected_cols,

        preprocessor=None,
        estimator=None,
        predictor=None,

        threshold=0.5,
        group_thresholds=None,
        group_models=None,
        calibrators=None,
        postprocessor=None,

        outcome_col=None,
        positive_label=1,

        model_type=None,
        model_params=None,
        test_size=0.3,
        random_state=42,
        min_group_size=20,
        require_class_balance=True,
        make_plots=True,
        return_intermediates=True,
        return_non_intersectional=True,

        # Component 3 settings
        component3_method="sr",
        component3_n_splits=5,
        component3_cutoff=None,
        component3_gen_null=True,
        component3_R_null=200,
        component3_bootstrap="none",
        component3_B=500,
        component3_m_factor=0.75,

        fairlogue_component1_results=None,
        fairlogue_component1_figs=None,
        fairlogue_component1_intermediates=None,

        # Component 3 outputs
        fairlogue_component3_results=None,
        fairlogue_component3_summary=None,
        fairlogue_component3_plots=None,

        metadata=None,
    ):
        self.name = name
        self.features = None if features is None else list(features)
        self.protected_cols = list(protected_cols)

        self.preprocessor = preprocessor
        self.estimator = estimator
        self.predictor = predictor

        self.threshold = threshold
        self.group_thresholds = group_thresholds or {}
        self.group_models = group_models or {}
        self.calibrators = calibrators or {}
        self.postprocessor = postprocessor

        self.outcome_col = outcome_col
        self.positive_label = positive_label

        self.model_type = model_type
        self.model_params = model_params or {}
        self.test_size = test_size
        self.random_state = random_state
        self.min_group_size = min_group_size
        self.require_class_balance = require_class_balance
        self.make_plots = make_plots
        self.return_intermediates = return_intermediates
        self.return_non_intersectional = return_non_intersectional

        self.component3_method = component3_method
        self.component3_n_splits = component3_n_splits
        self.component3_cutoff = component3_cutoff
        self.component3_gen_null = component3_gen_null
        self.component3_R_null = component3_R_null
        self.component3_bootstrap = component3_bootstrap
        self.component3_B = component3_B
        self.component3_m_factor = component3_m_factor

        self.fairlogue_component1_results = fairlogue_component1_results
        self.fairlogue_component1_figs = fairlogue_component1_figs
        self.fairlogue_component1_intermediates = fairlogue_component1_intermediates

        self.fairlogue_component3_results = fairlogue_component3_results
        self.fairlogue_component3_summary = fairlogue_component3_summary
        self.fairlogue_component3_plots = fairlogue_component3_plots

        self.metadata = metadata or {}

    # Creates the intersectional group based off the protected columns
    def make_group(self, df):
        return df[self.protected_cols].astype(str).agg("|".join, axis=1)

    def transform(self, df):
        """
        Convert a raw dataframe into the exact feature matrix expected by the fitted model.
        """

        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

        X = df[self.features].copy()

        if self.preprocessor is not None:
            return self.preprocessor.transform(X)

        return X

    def predict_proba(self, df):
        if self.predictor is not None:
            return self.predictor.predict_proba(df)

        if self.estimator is None:
            raise ValueError("No predictor or estimator has been assigned.")

        X = self.transform(df)

        if hasattr(self.estimator, "predict_proba"):
            proba = self.estimator.predict_proba(X)
            return proba[:, 1] if proba.shape[1] == 2 else proba.max(axis=1)

        if hasattr(self.estimator, "decision_function"):
            scores = self.estimator.decision_function(X).astype(float)
            return (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

        return self.estimator.predict(X).astype(float)

    def predict(self, df):
        if self.predictor is not None:
            return self.predictor.predict(df)

        proba = self.predict_proba(df)
        return (proba >= self.threshold).astype(int)