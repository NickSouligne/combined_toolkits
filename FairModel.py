#### Goal is to integrate both FairLogue and FairSelect into a single re-usable pipeline
#### To do this, we need to create a way of sharing model objects between the toolkits

#### This class will then create a single model object that can be used by both FairLogue and FairSelect

import numpy as np
import pandas as pd

from copy import deepcopy
from typing import Any, Dict, Optional

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
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
        Convert a raw dataframe into the feature matrix expected by the model.
        """
        if self.features is None:
            raise ValueError("FairModel.features has not been configured.")

        missing = [
            col for col in self.features
            if col not in df.columns
        ]
        if missing:
            raise ValueError(
                f"Missing required feature columns: {missing}"
            )

        X = df[self.features].copy()

        if self.preprocessor is not None:
            X = self.preprocessor.transform(X)

        dense_output = self.metadata.get("dense_output", False)

        if dense_output and hasattr(X, "toarray"):
            X = X.toarray()

        return X

    def predict_proba(self, df):
        """
        Return P(Y=positive_label) as a one-dimensional array.
        """
        if self.predictor is not None:
            probabilities = self.predictor.predict_proba(df)
        else:
            if self.estimator is None:
                raise ValueError(
                    "No predictor or estimator has been assigned."
                )

            X = self.transform(df)

            if hasattr(self.estimator, "predict_proba"):
                probabilities = self.estimator.predict_proba(X)

            elif hasattr(self.estimator, "decision_function"):
                scores = np.asarray(
                    self.estimator.decision_function(X),
                    dtype=float,
                )

                # Logistic transformation is preferable to min-max scaling,
                # because min-max results depend on the current prediction batch.
                if scores.ndim == 1:
                    return 1.0 / (1.0 + np.exp(-scores))

                probabilities = np.exp(
                    scores - scores.max(axis=1, keepdims=True)
                )
                probabilities /= probabilities.sum(
                    axis=1,
                    keepdims=True,
                )

            else:
                return np.asarray(
                    self.estimator.predict(X),
                    dtype=float,
                )

        probabilities = np.asarray(probabilities)

        if probabilities.ndim == 1:
            return probabilities.astype(float)

        if probabilities.shape[1] != 2:
            raise ValueError(
                "FairLogue currently expects a binary outcome, but "
                f"predict_proba returned shape {probabilities.shape}."
            )

        classes = getattr(
            self.estimator,
            "classes_",
            np.array([0, 1]),
        )

        positive_matches = np.where(
            np.asarray(classes) == self.positive_label
        )[0]

        positive_index = (
            int(positive_matches[0])
            if len(positive_matches)
            else 1
        )

        return probabilities[:, positive_index].astype(float)

    def predict(self, df):
        if self.predictor is not None:
            return self.predictor.predict(df)

        proba = self.predict_proba(df)
        return (proba >= self.threshold).astype(int)
    
    @staticmethod
    def build_preprocessor(
        df: pd.DataFrame,
        features,
        *,
        dense_output: bool = False,
    ):
        """
        Build preprocessing from the training dataframe only.
        """
        feature_df = df[list(features)]

        numeric_cols = feature_df.select_dtypes(
            include=["number", "bool"]
        ).columns.tolist()

        categorical_cols = [
            col for col in features
            if col not in numeric_cols
        ]

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "scaler",
                    StandardScaler(with_mean=dense_output),
                ),
            ]
        )

        try:
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=not dense_output,
            )
        except TypeError:
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse=not dense_output,
            )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", encoder),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, numeric_cols),
                ("categorical", categorical_pipeline, categorical_cols),
            ],
            remainder="drop",
            sparse_threshold=0.0 if dense_output else 0.3,
        )
    
    @classmethod
    def fit_from_dataframe(
        cls,
        train_df: pd.DataFrame,
        *,
        name: str,
        outcome_col: str,
        features,
        protected_cols,
        model_type: str,
        model_params: Optional[Dict[str, Any]] = None,
        positive_label: Any = 1,
        threshold: float = 0.5,
        random_state: int = 42,
    ):
        """
        Fit a FairModel using only train_df.

        This method is safe to call independently inside each CV fold.
        """

        from FairSelect.core import build_estimator, build_preprocessor
        
        features = [
            col for col in features
            if col != outcome_col
        ]

        missing = [
            col for col in features
            if col not in train_df.columns
        ]
        if missing:
            raise ValueError(
                f"Training data is missing feature columns: {missing}"
            )

        mt = model_type.lower().strip()

        # MLP requires dense input. The tree models can also accept dense input,
        # but do not require it.
        dense_output = mt in {
            "nn",
            "mlp",
            "neural",
            "neural_network",
        }

        preprocessor = cls.build_preprocessor(
            train_df,
            features,
            dense_output=dense_output,
        )

        estimator = cls.build_estimator(
            model_type=model_type,
            model_params=model_params,
            random_state=random_state,
        )

        X_train = train_df[features].copy()
        y_train = (
            train_df[outcome_col].to_numpy() == positive_label
        ).astype(int)

        X_train_transformed = preprocessor.fit_transform(X_train)

        if hasattr(X_train_transformed, "toarray") and dense_output:
            X_train_transformed = X_train_transformed.toarray()

        estimator.fit(X_train_transformed, y_train)

        return cls(
            name=name,
            features=features,
            protected_cols=protected_cols,
            preprocessor=preprocessor,
            estimator=estimator,
            threshold=threshold,
            outcome_col=outcome_col,
            positive_label=positive_label,
            model_type=model_type,
            model_params=model_params,
            random_state=random_state,
            metadata={
                "n_train": len(train_df),
                "dense_output": dense_output,
            },
        )