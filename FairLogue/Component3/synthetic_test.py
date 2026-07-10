from pathlib import Path
import sys

import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds

# ---------------------------------------------------------------------
# Import shared FairModel from:
# Combined_Toolkits/FairModel.py
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FairModel import FairModel
from model import Model


class FairlearnEGPredictor:
    """
    FairModel-compatible wrapper around a fitted Fairlearn
    ExponentiatedGradient model.

    This wrapper gives FairModel the two methods Component 3 needs:
        predict_proba(df)
        predict(df)

    Component 3 does not need to know this came from Fairlearn.
    """

    def __init__(
        self,
        features,
        protected_cols,
        fitted_model,
        threshold=0.5,
    ):
        self.features = list(features)
        self.protected_cols = list(protected_cols)
        self.fitted_model = fitted_model
        self.threshold = threshold

    def make_sensitive_features(self, df):
        return (
            df[self.protected_cols]
            .astype(str)
            .agg("|".join, axis=1)
        )

    def predict_proba(self, df):
        """
        Return a probability-like score for Component 3's muY_<group> columns.

        Fairlearn ExponentiatedGradient is a randomized classifier. Some versions
        expose _pmf_predict(), which returns class probabilities. If unavailable,
        we fall back to hard predictions as scores.
        """

        X = df[self.features].copy()

        if hasattr(self.fitted_model, "_pmf_predict"):
            pmf = self.fitted_model._pmf_predict(X)

            if pmf.ndim == 2 and pmf.shape[1] >= 2:
                return pmf[:, 1].astype(float)

        yhat = self.fitted_model.predict(X, random_state=0)
        return np.asarray(yhat, dtype=float)

    def predict(self, df):
        """
        Return final hard predictions from the fitted in-processing model.
        """

        X = df[self.features].copy()
        yhat = self.fitted_model.predict(X, random_state=0)
        return np.asarray(yhat, dtype=int)


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    df = pd.read_csv("FairLogue\\Component3\\glaucoma_synth_component3.csv")

    df["Y"] = df["Y"].astype(int)

    protected_cols = ["A1", "A2"]
    group_col = "A1A2"

    df[group_col] = (
        df[protected_cols]
        .astype(str)
        .agg("|".join, axis=1)
    )

    protected = set(protected_cols + [group_col])
    covariates = [c for c in df.columns if c not in protected | {"Y"}]

    # ------------------------------------------------------------------
    # 2. Train/test split
    # ------------------------------------------------------------------
    train_df, test_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["Y"],
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3. Prepare training data
    # ------------------------------------------------------------------
    X_train = train_df[covariates].copy()
    y_train = train_df["Y"].astype(int)

    sensitive_train = (
        train_df[protected_cols]
        .astype(str)
        .agg("|".join, axis=1)
    )

    # ------------------------------------------------------------------
    # 4. Apply SMOTE to training data only
    # ------------------------------------------------------------------
    # For this integration test, we carry the sensitive group through SMOTE
    # as a numeric code so the in-processing model has sensitive_features
    # aligned to the resampled rows.
    # ------------------------------------------------------------------

    X_train_with_group = X_train.copy()
    X_train_with_group["_sensitive_group"] = sensitive_train.values

    group_levels = sorted(
        X_train_with_group["_sensitive_group"]
        .astype(str)
        .unique()
    )

    group_map = {g: i for i, g in enumerate(group_levels)}
    inverse_group_map = {i: g for g, i in group_map.items()}

    X_train_with_group["_sensitive_group_code"] = (
        X_train_with_group["_sensitive_group"]
        .astype(str)
        .map(group_map)
        .astype(int)
    )

    X_train_for_smote = X_train_with_group.drop(
        columns=["_sensitive_group"]
    )

    smote = SMOTE(random_state=42)

    X_train_smote_all, y_train_smote = smote.fit_resample(
        X_train_for_smote,
        y_train,
    )

    X_train_smote_all["_sensitive_group_code"] = (
        X_train_smote_all["_sensitive_group_code"]
        .round()
        .clip(lower=0, upper=len(group_levels) - 1)
        .astype(int)
    )

    sensitive_train_smote = (
        X_train_smote_all["_sensitive_group_code"]
        .map(inverse_group_map)
        .astype(str)
    )

    X_train_smote = X_train_smote_all.drop(
        columns=["_sensitive_group_code"]
    )

    # ------------------------------------------------------------------
    # 5. Fit real in-processing model
    # ------------------------------------------------------------------
    # ExponentiatedGradient is the in-processing method.
    # It changes the fitted model by enforcing a fairness constraint.
    # ------------------------------------------------------------------

    base_estimator = LogisticRegression(
        max_iter=1000,
        solver="liblinear",
    )

    eg_model = ExponentiatedGradient(
        estimator=base_estimator,
        constraints=EqualizedOdds(),
        eps=0.01,
    )

    eg_model.fit(
        X_train_smote,
        y_train_smote,
        sensitive_features=sensitive_train_smote,
    )

    # ------------------------------------------------------------------
    # 6. Wrap fitted in-processing model
    # ------------------------------------------------------------------

    eg_predictor = FairlearnEGPredictor(
        features=covariates,
        protected_cols=protected_cols,
        fitted_model=eg_model,
        threshold=0.5,
    )

    # ------------------------------------------------------------------
    # 7. Save mitigated model into FairModel
    # ------------------------------------------------------------------
    # FairModel is now the object that contains the fitted model with
    # preprocessing + in-processing already applied.
    # ------------------------------------------------------------------

    fair_model = FairModel(
        name="SMOTE + ExponentiatedGradient(EqualizedOdds)",
        features=covariates,
        protected_cols=protected_cols,
        predictor=eg_predictor,
        threshold=0.5,
        outcome_col="Y",
        positive_label=1,
        metadata={
            "pre": "SMOTE",
            "in": "ExponentiatedGradient",
            "constraint": "EqualizedOdds",
            "post": None,
            "smote_random_state": 42,
            "eg_eps": 0.01,
        },
    )

    # ------------------------------------------------------------------
    # 8. Sanity check FairModel predictions
    # ------------------------------------------------------------------

    p_test = fair_model.predict_proba(test_df)
    yhat_test = fair_model.predict(test_df)

    print("\n[FairModel sanity check]")
    print("Model:", fair_model.name)
    print("n_test:", len(test_df))
    print("Probability/score range:", float(np.min(p_test)), "to", float(np.max(p_test)))
    print("Positive prediction rate:", float(np.mean(yhat_test)))

    # ------------------------------------------------------------------
    # 9. Run FairLogue Component 3 external audit
    # ------------------------------------------------------------------
    # Important:
    # We are NOT calling m_c3.fit_fairness().
    # That would fit Component 3's internal outcome model.
    #
    # Instead, fit_fairness_from_fairmodel() uses:
    #     fair_model.predict_proba(df_cf)
    #     fair_model.predict(df_cf)
    #
    # for each counterfactual group.
    # ------------------------------------------------------------------

    m_c3 = Model(
        data=test_df.copy(),
        outcome=fair_model.outcome_col,
        protected_characteristics=tuple(fair_model.protected_cols),
        covariates=fair_model.features,
        fair_model=fair_model,
        method="sr",
        n_splits=5,
        random_state=42,
    )

    m_c3.pre_process_data()

    res_external = m_c3.fit_fairness_from_fairmodel(
        cutoff=fair_model.threshold,
        gen_null=False,
        bootstrap="none",
    )

    # ------------------------------------------------------------------
    # 10. Report Component 3 audit results
    # ------------------------------------------------------------------

    print("\n[Component 3 external audit]")
    print("Audited FairModel:", fair_model.name)
    print("Audit source:", res_external.get("audit_source"))
    print("Groups:", res_external.get("groups"))
    print("Tau:", res_external.get("tau"))

    print("\n[Point estimates]")
    print(
        m_c3
        .summarize()
        .sort_values("stat")
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    # 11. Confirm counterfactual columns were generated
    # ------------------------------------------------------------------

    est_choice = res_external["est_choice"]

    mu_cols = [c for c in est_choice.columns if c.startswith("muY_")]
    s_cols = [c for c in est_choice.columns if c.startswith("S_")]

    print("\nGenerated counterfactual probability/score columns:")
    print(mu_cols)

    print("\nGenerated counterfactual decision columns:")
    print(s_cols)