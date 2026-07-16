from __future__ import annotations
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from .Fairness import FairnessPipeline   
from .plots import get_plots
from sklearn.base import clone
from .estimation_functions import get_defs_analysis


def ensure_probabilistic_estimator(estimator, *, method: str = "isotonic", cv: int = 3):
    """
    Return an estimator with `predict_proba`.

    If `estimator` already implements `predict_proba`, return it unchanged;
    otherwise wrap with `CalibratedClassifierCV(method, cv)` to obtain calibrated
    probabilities. See docs: docs/model.md#ensure_probabilistic_estimator.
    """
    if hasattr(estimator, "predict_proba"):
        return estimator

    return CalibratedClassifierCV(estimator=estimator, method=method, cv=cv)




class Model:
    """
    Estimator-agnostic fairness facade.

    Wraps the pipeline with a simple, stateful object. Accepts any sklearn
    classifier for the outcome model (and, in DR mode, for the propensity model).
    Provides: preprocessing, fitting (SR/DR), summaries, and plots.
    See docs: docs/model.md#class-model
    """

    def __init__(
        self,
        data: pd.DataFrame,
        model_type: str = "rf",                        
        *,
        outcome_estimator: Optional[object] = None,    
        propensity_estimator: Optional[object] = None,  
        treatment: Optional[str] = None,
        outcome: Optional[str] = None,
        covariates: Optional[List[str]] = None,
        protected_characteristics: tuple = (),
        risk_score: Optional[str] = None,
        treatment_flag: bool = True,
        group_label_map: Optional[dict] = None,
        coeff_map: Optional[dict] = None,
        random_state: int = 42,
        n_splits: int = 5,
        method: str = "sr",             #'sr' or 'dr'
        auto_compute_propensity: bool = True,
        calibration_method: str = "isotonic",
        calibration_cv: int = 3,
        fair_model = None,
    ):
        """
        Initialize configuration and hold data.

        Most users set: outcome, protected_characteristics, covariates,
        and either outcome_estimator or model_type. For DR, optionally
        supply propensity_estimator (or let it auto-compute).
        See docs: docs/model.md#__init__
        """
        self.data = data.copy()
        self.Y = outcome
        self.D = treatment
        self.covariates = list(covariates) if covariates is not None else []
        self.model_type = model_type
        self.A1 = protected_characteristics[0] if len(protected_characteristics) > 0 else None
        self.A2 = protected_characteristics[1] if len(protected_characteristics) > 1 else None
        self.S_prob = risk_score
        self.treatment_flag = treatment_flag
        self.group_label_map = group_label_map
        self.coeff_map = coeff_map

        #Internals
        self.A = "A1A2"
        self.results_: Optional[Dict[str, object]] = None

        #Pipeline config
        self._outcome_estimator = outcome_estimator  #can be None → pipeline uses model_type factory
        self._propensity_estimator = propensity_estimator
        self._random_state = random_state
        self._n_splits = n_splits
        self._method = method
        self._auto_compute_propensity = auto_compute_propensity
        self._calibration_method = calibration_method
        self._calibration_cv = calibration_cv

        self.fair_model = fair_model

    #---------- basics pre-processing ----------
    def pre_process_data(self) -> None:
        if self.Y is None or self.Y not in self.data.columns:
            raise ValueError("Outcome column 'Y' must be provided and present in data.")
        self.data[self.Y] = self.data[self.Y].astype('category')

        if self.D is not None and self.D in self.data.columns:
            self.data[self.D] = self.data[self.D].astype('category')

        if not self.A1 or not self.A2:
            raise ValueError("protected_characteristics must provide (A1, A2).")

        if self.A1 not in self.data.columns or self.A2 not in self.data.columns:
            raise ValueError("A1 or A2 not found in data columns.")

        #intersectional label
        self.data[self.A] = self.data[self.A1].astype(str) + self.data[self.A2].astype(str)

        #default covariates if not provided: numeric columns except protected + Y + D + A1A2
        if not self.covariates:
            drop_cols = {self.Y, self.A1, self.A2, self.A}
            if self.D is not None:
                drop_cols.add(self.D)
            self.covariates = [
                c for c in self.data.columns
                if c not in drop_cols and self.data[c].dtype != 'O'
            ]

    def _ensure_dr_inputs(self) -> None:
        """
        In DR mode, ensure π_g(X) columns exist or auto-fit them.

        If method=='dr' and columns like group_<g>_prob are missing, fit a
        multiclass propensity model and add them. No-op otherwise.
        See docs: docs/model.md#_ensure_dr_inputs
        """
        if self._method != "dr":
            return

        #determine which columns should exist
        groups = self.data[self.A].astype(str).unique().tolist()
        missing = [g for g in groups if f"group_{g}_prob" not in self.data.columns]
        if not missing:
            return

        if not self._auto_compute_propensity:
            missing_cols = [f"group_{g}_prob" for g in missing]
            raise ValueError(
                f"DR mode selected but missing propensity columns: {missing_cols}. "
                f"Either provide them or set auto_compute_propensity=True."
            )

        #Fit π_g(X) and append group_<g>_prob columns
        self.data = Model.add_group_propensities_general(
            df=self.data,
            covariates=self.covariates,
            group_col=self.A,
            estimator=self._propensity_estimator,
            random_state=self._random_state,
            calibration_method=self._calibration_method,
            calibration_cv=self._calibration_cv,
        )

    def _prepare_fairmodel_scored_data(
        self,
        data: pd.DataFrame,
        *,
        groups_universe: Optional[List[str]] = None,
    ) -> tuple[pd.DataFrame, List[str]]:
        """
        Generate counterfactual FairModel probabilities and hard decisions.

        In DR mode, propensity scores are fitted on the supplied audit
        population after the factual intersectional group column has been
        constructed.
        """
        scored_data, observed_groups = (
            self.build_scores_from_fairmodel(
                fair_model=self.fair_model,
                data=data,
                group_col=self.A,
                outcome_col=self.Y,
            )
        )

        groups = sorted(
            groups_universe
            or observed_groups
        )

        if self._method == "dr":
            # Recompute propensities for this exact audit/resampled dataset.
            scored_data = (
                Model.add_group_propensities_general(
                    df=scored_data,
                    covariates=self.covariates,
                    group_col=self.A,
                    estimator=self._propensity_estimator,
                    random_state=self._random_state,
                    calibration_method=(
                        self._calibration_method
                    ),
                    calibration_cv=self._calibration_cv,
                )
            )

            missing_propensity_columns = [
                f"group_{group}_prob"
                for group in groups
                if (
                    f"group_{group}_prob"
                    not in scored_data.columns
                )
            ]

            if missing_propensity_columns:
                raise ValueError(
                    "DR FairModel scoring did not generate all required "
                    "propensity columns. Missing: "
                    f"{missing_propensity_columns}"
                )

        return scored_data, groups

    def _bootstrap_from_fairmodel(
        self,
        *,
        data: pd.DataFrame,
        tau: float,
        groups_universe: List[str],
        B: int,
        m_factor: float,
    ) -> List[Dict[str, float]]:
        """
        Rescaled, group-stratified bootstrap for an already-fitted FairModel.

        The FairModel remains fixed. Audit observations are resampled within
        factual intersectional groups, counterfactual scores are regenerated,
        and Component 3 statistics are recomputed.
        """
        if B < 1:
            raise ValueError(
                f"B must be at least 1. Received {B}."
            )

        if not 0 < m_factor <= 1:
            raise ValueError(
                "m_factor must be in (0, 1]. "
                f"Received {m_factor}."
            )

        base_data = data.copy()

        base_data[self.A] = (
            self.fair_model
            .make_group(base_data)
            .astype(str)
        )

        rng = np.random.default_rng(
            self._random_state + 29
        )

        n_total = len(base_data)

        if n_total == 0:
            raise ValueError(
                "Cannot bootstrap an empty audit dataset."
            )

        m_total = max(
            len(groups_universe),
            int(np.floor(n_total ** m_factor)),
        )

        factual_group_counts = (
            base_data[self.A]
            .value_counts()
            .reindex(
                groups_universe,
                fill_value=0,
            )
        )

        missing_factual_groups = factual_group_counts[
            factual_group_counts == 0
        ].index.tolist()

        if missing_factual_groups:
            raise ValueError(
                "The audit dataset does not contain every group in the "
                f"group universe: {missing_factual_groups}"
            )

        group_proportions = (
            factual_group_counts
            / factual_group_counts.sum()
        )

        requested_counts = np.floor(
            group_proportions.to_numpy()
            * m_total
        ).astype(int)

        # Guarantee at least one observation from each factual group.
        requested_counts = np.maximum(
            requested_counts,
            1,
        )

        # Adjust exactly to m_total.
        while requested_counts.sum() > m_total:
            reducible = np.flatnonzero(
                requested_counts > 1
            )

            if len(reducible) == 0:
                break

            position = reducible[
                np.argmax(
                    requested_counts[reducible]
                )
            ]

            requested_counts[position] -= 1

        while requested_counts.sum() < m_total:
            expected = (
                group_proportions.to_numpy()
                * m_total
            )

            position = int(
                np.argmax(
                    expected - requested_counts
                )
            )

            requested_counts[position] += 1

        bootstrap_results = []

        for bootstrap_number in range(B):
            sampled_parts = []

            for group, sample_count in zip(
                groups_universe,
                requested_counts,
            ):
                group_data = base_data.loc[
                    base_data[self.A] == group
                ]

                sampled_positions = rng.choice(
                    len(group_data),
                    size=int(sample_count),
                    replace=True,
                )

                sampled_parts.append(
                    group_data.iloc[
                        sampled_positions
                    ]
                )

            bootstrap_data = (
                pd.concat(
                    sampled_parts,
                    axis=0,
                    ignore_index=True,
                )
                .sample(
                    frac=1.0,
                    random_state=(
                        self._random_state
                        + 1000
                        + bootstrap_number
                    ),
                )
                .reset_index(drop=True)
            )

            scored_bootstrap, _ = (
                self._prepare_fairmodel_scored_data(
                    bootstrap_data,
                    groups_universe=groups_universe,
                )
            )

            bootstrap_defs = get_defs_analysis(
                data_with_mu=scored_bootstrap,
                group_col=self.A,
                outcome_col=self.Y,
                tau=tau,
                method=self._method,
                groups_universe=groups_universe,
            )

            bootstrap_results.append(
                bootstrap_defs
            )

        return bootstrap_results
    
    def _null_distribution_from_fairmodel(
        self,
        *,
        data: pd.DataFrame,
        tau: float,
        groups_universe: List[str],
        R_null: int,
    ) -> Dict[str, List[float]]:
        """
        Generate a permutation null for a fixed, already-fitted FairModel.

        Protected-characteristic vectors are permuted jointly across audit
        observations. This breaks their association with covariates and
        outcomes while preserving the observed joint protected-group
        distribution.

        The FairModel is not refitted.
        """
        if R_null < 1:
            raise ValueError(
                "R_null must be at least 1. "
                f"Received {R_null}."
            )

        base_data = data.copy()

        protected_columns = list(
            self.fair_model.protected_cols
        )

        missing_protected = [
            column
            for column in protected_columns
            if column not in base_data.columns
        ]

        if missing_protected:
            raise ValueError(
                "Cannot generate the FairModel null distribution because "
                "protected columns are missing: "
                f"{missing_protected}"
            )

        protected_values = (
            base_data[protected_columns]
            .to_numpy(copy=True)
        )

        rng = np.random.default_rng(
            self._random_state + 13
        )

        null_rows = []

        for permutation_number in range(R_null):
            permuted_data = base_data.copy()

            permutation = rng.permutation(
                len(permuted_data)
            )

            # Permute the complete protected vector jointly.
            permuted_data.loc[
                :,
                protected_columns,
            ] = protected_values[
                permutation
            ]

            # Rebuild the factual intersectional group after permutation.
            permuted_data[self.A] = (
                self.fair_model
                .make_group(permuted_data)
                .astype(str)
            )

            scored_permutation, _ = (
                self._prepare_fairmodel_scored_data(
                    permuted_data,
                    groups_universe=groups_universe,
                )
            )

            permutation_defs = get_defs_analysis(
                data_with_mu=scored_permutation,
                group_col=self.A,
                outcome_col=self.Y,
                tau=tau,
                method=self._method,
                groups_universe=groups_universe,
            )

            null_rows.append(
                permutation_defs
            )

        all_keys = sorted(
            set().union(
                *[
                    row.keys()
                    for row in null_rows
                ]
            )
        )

        table_null = {
            key: [
                row.get(
                    key,
                    np.nan,
                )
                for row in null_rows
            ]
            for key in all_keys
        }

        return table_null


    @staticmethod
    def add_group_propensities_general(
        df: pd.DataFrame,
        covariates: List[str],
        group_col: str = "A1A2",
        estimator: Optional[object] = None,
        random_state: int = 42,
        calibration_method: str = "isotonic",
        calibration_cv: int = 3,
    ) -> pd.DataFrame:
        """
        Fit π_g(X)=P(A=g|X) with any sklearn classifier and add columns.

        Writes one column per class: group_<g>_prob. Wraps non-probabilistic
        estimators with calibration. Returns a copy of df with added columns.
        See docs: docs/model.md#add_group_propensities_general
        """
        out = df.copy()
        X = out[covariates].to_numpy()
        y = out[group_col].astype(str).to_numpy()

        #default multiclass estimator if none provided
        base = estimator if estimator is not None else RandomForestClassifier(
            n_estimators=400, max_depth=None, random_state=random_state, n_jobs=-1
        )
        clf = ensure_probabilistic_estimator(
            clone(base), method=calibration_method, cv=calibration_cv
        )

        clf.fit(X, y)
        probs = clf.predict_proba(X)
        classes = np.asarray(clf.classes_, dtype=str)

        for j, g in enumerate(classes):
            out[f"group_{g}_prob"] = probs[:, j]

        return out


    #---------- Model info ----------
    def get_model_info(self) -> Dict[str, Any]:
        """
        Return a readable snapshot of configuration and data shape.

        Useful for logs and quick inspection (models used, method, columns, sizes).
        See docs: docs/model.md#get_model_info
        """
        return {
            "outcome_model": (
                f"custom({type(self._outcome_estimator).__name__})"
                if self._outcome_estimator is not None else self.model_type
            ),
            "propensity_model": (
                None if self._method != "dr" else (
                    f"custom({type(self._propensity_estimator).__name__})"
                    if self._propensity_estimator is not None else "default+calibration"
                )
            ),
            "method": self._method,
            "outcome": self.Y,
            "treatment": self.D,
            "protected_A1": self.A1,
            "protected_A2": self.A2,
            "group_col": self.A,
            "covariates": list(self.covariates),
            "risk_score_col": self.S_prob,
            "n_rows": int(len(self.data)),
            "n_covariates": int(len(self.covariates)),
            "n_splits": self._n_splits,
            "random_state": self._random_state,
            "auto_compute_propensity": self._auto_compute_propensity,
            "calibration": {"method": self._calibration_method, "cv": self._calibration_cv},
        }

    #---------- run fairness ----------
    def fit_fairness(
        self,
        cutoff: Optional[float] = None,
        gen_null: bool = True,
        R_null: int = 200,
        bootstrap: str = "rescaled",
        B: int = 500,
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        m_factor: float = 0.75
    ) -> Dict[str, object]:
        """
        Run the full pipeline (SR/DR): cross-fit μ, choose τ, compute defs.

        Optionally builds a permutation null and/or rescaled bootstrap.
        Returns the pipeline results dict. See docs: docs/model.md#fit_fairness
        """
        if self.A not in self.data.columns:
            raise RuntimeError("Call pre_process_data() before fit_fairness().")

        #Ensure DR inputs if needed (this will fit π_g with any estimator)
        self._ensure_dr_inputs()

        pipe = FairnessPipeline(
            group_col=self.A,
            outcome_col=self.Y,
            covariates=self.covariates,
            estimator=self._outcome_estimator,   
            model_type=self.model_type,
            n_splits=self._n_splits,
            random_state=self._random_state,
            method=self._method
        )
        self.results_ = pipe.fit(
            data=self.data,
            cutoff=cutoff,
            gen_null=gen_null,
            R_null=R_null,
            bootstrap=bootstrap,
            B=B,
            train_df=train_df,
            test_df=test_df,
            m_factor=m_factor
        )
        return self.results_

    def summarize(self) -> pd.DataFrame:
        """
        Return key statistics as a tidy table (stat, value).

        Includes aggregate disparities and per-group cFPR/cFNR and observed FPR/FNR.
        See docs: docs/model.md#summarize
        """
        if self.results_ is None:
            raise RuntimeError("Call fit_fairness() before summarize().")
        return pd.DataFrame(list(self.results_["defs"].items()), columns=["stat", "value"])

    def plots(
        self,
        alpha: float = 0.05,
        m_factor: float = 0.75,
        delta_uval: float = 0.05
    ):
        """
        Assemble plot data, optional figures, and u-values.

        Returns (est_summaries, table_null_delta, table_uval). Figures render
        when a matplotlib backend is active. See docs: docs/model.md#plots
        """
        
        if self.results_ is None:
            raise RuntimeError("Call fit_fairness() before plots().")
        return get_plots(
            results=self.results_,
            sampsize=len(self.results_.get('est_choice', [])),
            alpha=alpha,
            m_factor=m_factor,
            delta_uval=delta_uval
        )
    
    def build_scores_from_fairmodel(
        self,
        fair_model,
        data,
        group_col,
        outcome_col,
    ):
        out = data.copy()

        out[group_col] = (
            fair_model.make_group(out)
            .astype(str)
        )

        groups = sorted(
            out[group_col].unique()
        )

        expected_parts = len(
            fair_model.protected_cols
        )

        for group in groups:
            parts = str(group).split("|")

            if len(parts) != expected_parts:
                raise ValueError(
                    "Counterfactual group label cannot be "
                    "mapped to the protected columns: "
                    f"group={group!r}, "
                    f"protected_cols="
                    f"{fair_model.protected_cols}, "
                    f"parts={parts}."
                )

            df_cf = out.copy()

            for protected_column, value in zip(
                fair_model.protected_cols,
                parts,
            ):
                df_cf[protected_column] = value

            probabilities = np.asarray(
                fair_model.predict_proba(
                    df_cf
                ),
                dtype=float,
            ).ravel()

            predictions = np.asarray(
                fair_model.predict(
                    df_cf
                ),
                dtype=int,
            ).ravel()

            if len(probabilities) != len(out):
                raise RuntimeError(
                    f"FairModel returned "
                    f"{len(probabilities)} probabilities "
                    f"for {len(out)} rows under "
                    f"intervention {group!r}."
                )

            if len(predictions) != len(out):
                raise RuntimeError(
                    f"FairModel returned "
                    f"{len(predictions)} predictions "
                    f"for {len(out)} rows under "
                    f"intervention {group!r}."
                )

            out[f"muY_{group}"] = (
                probabilities
            )

            out[f"S_{group}"] = (
                predictions
            )

        return out, groups
        

    def fit_fairness_from_fairmodel(
        self,
        cutoff=None,
        gen_null=True,
        R_null=200,
        bootstrap="none",
        B=500,
        m_factor=0.75,
    ):
        """
        Audit an already-fitted FairModel using FairLogue Component 3.

        The FairModel is held fixed. Counterfactual probabilities and hard
        predictions are generated through FairModel.predict_proba() and
        FairModel.predict().

        Optional inference
        ------------------
        gen_null=True
            Generate a joint protected-characteristic permutation null.

        bootstrap="rescaled"
            Generate a group-stratified rescaled bootstrap over audit rows.

        The FairModel itself is not refitted during either procedure.
        """
        if self.fair_model is None:
            raise ValueError(
                "fair_model must be provided."
            )

        if self.A not in self.data.columns:
            raise RuntimeError(
                "Call pre_process_data() before "
                "fit_fairness_from_fairmodel()."
            )

        valid_bootstrap_options = {
            "none",
            "rescaled",
        }

        if bootstrap not in valid_bootstrap_options:
            raise ValueError(
                "bootstrap must be one of "
                f"{sorted(valid_bootstrap_options)}. "
                f"Received {bootstrap!r}."
            )

        audit_data = self.data.copy()

        # Reconstruct the factual group through the FairModel so the exact
        # same separator and protected-column ordering are used.
        audit_data[self.A] = (
            self.fair_model
            .make_group(audit_data)
            .astype(str)
        )

        groups_universe = sorted(
            audit_data[self.A]
            .unique()
            .tolist()
        )

        # ---------------------------------------------------------
        # Main observed estimate
        # ---------------------------------------------------------
        data_with_scores, groups = (
            self._prepare_fairmodel_scored_data(
                audit_data,
                groups_universe=groups_universe,
            )
        )

        tau = (
            float(cutoff)
            if cutoff is not None
            else float(
                getattr(
                    self.fair_model,
                    "threshold",
                    0.5,
                )
            )
        )

        defs = get_defs_analysis(
            data_with_mu=data_with_scores,
            group_col=self.A,
            outcome_col=self.Y,
            tau=tau,
            method=self._method,
            groups_universe=groups_universe,
        )

        results = {
            "defs": defs,
            "est_choice": data_with_scores.copy(),
            "tau": tau,
            "groups": groups,
            "audit_source": "fair_model",
            "fair_model_name": getattr(
                self.fair_model,
                "name",
                type(self.fair_model).__name__,
            ),
            "method": self._method,
            "gen_null_requested": bool(
                gen_null
            ),
            "R_null_requested": int(
                R_null
            ),
            "bootstrap_requested": (
                bootstrap
            ),
            "B_requested": int(B),
            "m_factor": float(m_factor),
        }

        # ---------------------------------------------------------
        # Permutation null
        # ---------------------------------------------------------
        if gen_null:
            results["table_null"] = (
                self._null_distribution_from_fairmodel(
                    data=audit_data,
                    tau=tau,
                    groups_universe=groups_universe,
                    R_null=int(R_null),
                )
            )

            results["gen_null_applied"] = True
            results["R_null_completed"] = int(
                R_null
            )

        else:
            results["gen_null_applied"] = False
            results["R_null_completed"] = 0

        # ---------------------------------------------------------
        # Rescaled bootstrap
        # ---------------------------------------------------------
        if bootstrap == "rescaled":
            results["boot_out"] = (
                self._bootstrap_from_fairmodel(
                    data=audit_data,
                    tau=tau,
                    groups_universe=groups_universe,
                    B=int(B),
                    m_factor=float(m_factor),
                )
            )

            results["bootstrap_applied"] = True
            results["B_completed"] = int(B)

        else:
            results["bootstrap_applied"] = False
            results["B_completed"] = 0

        self.results_ = results

        return self.results_