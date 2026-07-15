from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FunctionTransformer, Pipeline
from FairModel import FairModel
from .containers import FairnessResults, GroupRates
from .plots import (
    _plot_bar,
    _plot_bar_series_by_group,
    _plot_grouped_eods_components,
    _plot_fairness_matrix,
)
from .utilities import (
    _compute_group_rates,
    _make_ohe,
    _get_model,
    _as_prob,
)
from sklearn.model_selection import StratifiedKFold


def bootstrap_fairness_metrics(
    y_true,
    y_pred,
    groups,
    n_bootstrap: int = 1000,
    random_state: int = 42,
    min_group_size: int = 0,
    require_class_balance: bool = False,
):
    """
    Bootstrap confidence intervals for intersectional fairness metrics.

    Resamples the held-out test predictions with replacement and recomputes:
      - demographic_parity_gap
      - equalized_odds_gap_tpr
      - equalized_odds_gap_fpr
      - equal_opportunity_gap

    Returns one row per metric with:
      - observed value
      - bootstrap mean
      - bootstrap standard error
      - 95% percentile CI
      - approximate two-sided p-value for H0: metric = 0
    """

    rng = np.random.default_rng(random_state)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)

    n = len(y_true)

    def _compute_metrics(y_b, pred_b, groups_b):
        group_rates = _compute_group_rates(
            y_true=y_b,
            y_pred=pred_b,
            groups=pd.Series(groups_b)
        )

        df_groups = pd.DataFrame([gr.__dict__ for gr in group_rates])

        if min_group_size > 0:
            df_groups = df_groups[df_groups["n"] >= min_group_size]

        if require_class_balance:
            df_groups = df_groups[
                (df_groups["pos_true"] >= 1) &
                (df_groups["neg_true"] >= 1)
            ]

        def _gap(s):
            s = s.replace([np.inf, -np.inf], np.nan).dropna()
            return float(s.max() - s.min()) if not s.empty else np.nan

        if df_groups.empty:
            return {
                "demographic_parity_gap": np.nan,
                "equalized_odds_gap_tpr": np.nan,
                "equalized_odds_gap_fpr": np.nan,
                "equal_opportunity_gap": np.nan,
            }

        return {
            "demographic_parity_gap": _gap(df_groups["positive_rate"]),
            "equalized_odds_gap_tpr": _gap(df_groups["tpr"]),
            "equalized_odds_gap_fpr": _gap(df_groups["fpr"]),
            "equal_opportunity_gap": _gap(df_groups["tpr"]),
        }

    observed = _compute_metrics(y_true, y_pred, groups)

    boot_rows = []

    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)

        boot_metrics = _compute_metrics(
            y_true[idx],
            y_pred[idx],
            groups[idx]
        )

        boot_metrics["bootstrap_id"] = b + 1
        boot_rows.append(boot_metrics)

    boot_df = pd.DataFrame(boot_rows)

    summary_rows = []

    for metric, observed_value in observed.items():
        vals = boot_df[metric].replace([np.inf, -np.inf], np.nan).dropna()

        if vals.empty or pd.isna(observed_value):
            summary_rows.append({
                "metric": metric,
                "observed": observed_value,
                "bootstrap_mean": np.nan,
                "bootstrap_se": np.nan,
                "ci95_lower": np.nan,
                "ci95_upper": np.nan,
                "p_value_approx": np.nan,
                "significant_95ci": False,
            })
            continue

        ci_lower = np.percentile(vals, 2.5)
        ci_upper = np.percentile(vals, 97.5)

        # Approximate two-sided bootstrap p-value for H0: metric = 0.
        # Since these are gap metrics and are usually nonnegative,
        # this checks how often the bootstrap distribution is at or below zero.
        p_lower = np.mean(vals <= 0)
        p_upper = np.mean(vals >= 0)
        p_value = 2 * min(p_lower, p_upper)
        p_value = min(float(p_value), 1.0)

        summary_rows.append({
            "metric": metric,
            "observed": float(observed_value),
            "bootstrap_mean": float(vals.mean()),
            "bootstrap_se": float(vals.std(ddof=1)),
            "ci95_lower": float(ci_lower),
            "ci95_upper": float(ci_upper),
            "p_value_approx": p_value,
            "significant_95ci": not (ci_lower <= 0 <= ci_upper),
        })

    summary_df = pd.DataFrame(summary_rows)

    return {
        "bootstrap_samples": boot_df,
        "bootstrap_summary": summary_df,
    }


def cross_validate_intersectional_fairness(
    df: pd.DataFrame,
    outcome: str,
    protected_1: str,
    protected_2: str,
    features: Optional[List[str]] = None,
    model_type: str = "logreg",
    model_params: Optional[Dict[str, Any]] = None,
    k: int = 3,
    random_state: int = 42,
    positive_label: Any = 1,
    threshold: float = 0.5,
    min_group_size: int = 0,
    require_class_balance: bool = False,
    return_non_intersectional: bool = False,
    n_bootstrap: int = 1000,
    run_bootstrap: bool = True,
):
    """
    Runs k-fold cross-validation for intersectional fairness metrics.

    This wrapper keeps your existing evaluate_intersectional_fairness()
    function unchanged and uses its train_df / test_df interface.
    """

    df = df.copy()

    # Binary target used for stratification
    y = (df[outcome].values == positive_label).astype(int)

    # Intersectional group label
    df["_intersectional_group_cv"] = (
        df[protected_1].astype(str) + "|" + df[protected_2].astype(str)
    )

    # Optional: stratify by both outcome and intersectional group
    # This helps each fold preserve both case/control balance and group composition.
    stratify_label = (
        pd.Series(y).astype(str) + "_" + df["_intersectional_group_cv"].astype(str)
    )

    # Some rare strata may have fewer than k observations.
    # If so, fall back to outcome-only stratification.
    stratum_counts = stratify_label.value_counts()
    if (stratum_counts < k).any():
        print(
            "Warning: Some outcome-by-intersectional-group strata have fewer "
            f"than {k} observations. Falling back to outcome-only stratification."
        )
        stratify_label = y

    skf = StratifiedKFold(
        n_splits=k,
        shuffle=True,
        random_state=random_state
    )

    fold_summaries = []
    per_group_results = []
    bootstrap_summaries = []
    bootstrap_samples = []

    for fold_id, (train_idx, test_idx) in enumerate(
        skf.split(df, stratify_label), start=1
    ):
        train_df = df.iloc[train_idx].drop(columns=["_intersectional_group_cv"])
        test_df = df.iloc[test_idx].drop(columns=["_intersectional_group_cv"])

        results, figs, intermediates = evaluate_intersectional_fairness(
            df=df.drop(columns=["_intersectional_group_cv"]),
            outcome=outcome,
            protected_1=protected_1,
            protected_2=protected_2,
            features=features,
            model_type=model_type,
            model_params=model_params,
            positive_label=positive_label,
            threshold=threshold,
            make_plots=False,
            train_df=train_df,
            test_df=test_df,
            return_intermediates=True,
            return_non_intersectional=return_non_intersectional,
            min_group_size=min_group_size,
            require_class_balance=require_class_balance,
        )

        model_metrics = intermediates["model_metrics"]

        fold_summaries.append({
            "fold": fold_id,
            "accuracy": model_metrics["accuracy"],
            "auroc": model_metrics["auroc"],
            "demographic_parity_gap": results.demographic_parity_gap,
            "equalized_odds_gap_tpr": results.equalized_odds_gap_tpr,
            "equalized_odds_gap_fpr": results.equalized_odds_gap_fpr,
            "equal_opportunity_gap": results.equal_opportunity_gap,
            "n_test": len(test_df),
        })
        if run_bootstrap:
            boot = bootstrap_fairness_metrics(
                y_true=intermediates["y_test"],
                y_pred=intermediates["y_hat"],
                groups=intermediates["groups_test"],
                n_bootstrap=n_bootstrap,
                random_state=random_state + fold_id,
                min_group_size=min_group_size,
                require_class_balance=require_class_balance,
            )

            boot_summary = boot["bootstrap_summary"].copy()
            boot_summary["fold"] = fold_id
            bootstrap_summaries.append(boot_summary)

            boot_samples = boot["bootstrap_samples"].copy()
            boot_samples["fold"] = fold_id
            bootstrap_samples.append(boot_samples)

        fold_group_df = results.per_group_df.copy()
        fold_group_df["fold"] = fold_id
        per_group_results.append(fold_group_df)

    fold_metrics_df = pd.DataFrame(fold_summaries)
    per_group_cv_df = pd.concat(per_group_results, ignore_index=True)
    if run_bootstrap and bootstrap_summaries:
        bootstrap_summary_df = pd.concat(bootstrap_summaries, ignore_index=True)
        bootstrap_samples_df = pd.concat(bootstrap_samples, ignore_index=True)

        overall_bootstrap_summary = (
            bootstrap_summary_df
            .groupby("metric")
            .agg(
                mean_observed=("observed", "mean"),
                mean_bootstrap=("bootstrap_mean", "mean"),
                mean_se=("bootstrap_se", "mean"),
                mean_ci95_lower=("ci95_lower", "mean"),
                mean_ci95_upper=("ci95_upper", "mean"),
                significant_folds=("significant_95ci", "sum"),
                total_folds=("significant_95ci", "count"),
                mean_p_value=("p_value_approx", "mean"),
            )
            .reset_index()
        )

        overall_bootstrap_summary["proportion_significant_folds"] = (
            overall_bootstrap_summary["significant_folds"] /
            overall_bootstrap_summary["total_folds"]
        )

    else:
        bootstrap_summary_df = pd.DataFrame()
        bootstrap_samples_df = pd.DataFrame()
        overall_bootstrap_summary = pd.DataFrame()

    # Aggregate fold-level robustness summary
    summary_df = (
        fold_metrics_df
        .drop(columns=["fold", "n_test"])
        .agg(["mean", "std", "min", "max"])
        .T
        .reset_index()
        .rename(columns={"index": "metric"})
    )

    # 95% CI across folds using normal approximation
    summary_df["se"] = summary_df["std"] / np.sqrt(k)
    summary_df["ci95_lower"] = summary_df["mean"] - 1.96 * summary_df["se"]
    summary_df["ci95_upper"] = summary_df["mean"] + 1.96 * summary_df["se"]

    return {
        "fold_metrics": fold_metrics_df,
        "summary": summary_df,
        "per_group_cv": per_group_cv_df,
        "bootstrap_by_fold": bootstrap_summary_df,
        "bootstrap_samples": bootstrap_samples_df,
        "bootstrap_overall": overall_bootstrap_summary,
    }



def evaluate_intersectional_fairness(
        df: pd.DataFrame,
        outcome: str,
        protected_1: str,
        protected_2: str,
        features: Optional[List[str]] = None,
        model_type: str = "logreg",
        model_params: Optional[Dict[str, Any]] = None,
        fair_model=None,
        test_size: float = 0.3,
        random_state: int = 42,
        positive_label: Any = 1,
        threshold: Optional[float] = None,
        make_plots: bool = True,
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        return_intermediates: bool = False,
        return_non_intersectional: bool = False,
        min_group_size: int = 0,
        require_class_balance: bool = False,
    ):-> Tuple[FairnessResults, Dict[str, plt.Figure]]:
    
    #Check that protected characteristics and outcome are in data
    for col in (outcome, protected_1, protected_2):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in df.")
        
    #Computes fairness metrics for non intersectional groups (for optional return_non_intersectional output)
    def _compute_fairness_for_groups(group_labels, *, dropped_groups_local=None, kept_summary_local=None) -> FairnessResults:
        g_series_local = pd.Series(group_labels, index=np.arange(len(group_labels)))
        group_rates_local = _compute_group_rates(y_true=y_test, y_pred=y_hat, groups=g_series_local)
        df_groups_local = pd.DataFrame([gr.__dict__ for gr in group_rates_local])

        df_groups_filtered_local = df_groups_local.copy()
        #Filter out small groups or those without class balance, if specified (Avoid reporting NaN or 0 rates)
        if min_group_size > 0:
            df_groups_filtered_local = df_groups_filtered_local[df_groups_filtered_local["n"] >= min_group_size]
        if require_class_balance:
            df_groups_filtered_local = df_groups_filtered_local[
                (df_groups_filtered_local["pos_true"] >= 1) & (df_groups_filtered_local["neg_true"] >= 1)
            ]

        if df_groups_filtered_local.empty:
            df_for_metrics_local = df_groups_local.copy()
        else:
            df_for_metrics_local = df_groups_filtered_local

        if df_for_metrics_local.empty:
            privileged_group_local = None
            tpr_priv_local = fpr_priv_local = np.nan
        else:
            privileged_group_local = df_for_metrics_local.sort_values("n", ascending=False).iloc[0]["group"]
            tpr_priv_local = float(df_for_metrics_local.loc[df_for_metrics_local["group"] == privileged_group_local, "tpr"].iloc[0])
            fpr_priv_local = float(df_for_metrics_local.loc[df_for_metrics_local["group"] == privileged_group_local, "fpr"].iloc[0])

        df_disp_local = df_for_metrics_local.copy()
        #Find per-group disparities 
        df_disp_local["eo_diff"] = tpr_priv_local - df_disp_local["tpr"]
        df_disp_local["eod_tpr_diff"] = tpr_priv_local - df_disp_local["tpr"]
        df_disp_local["eod_fpr_diff"] = df_disp_local["fpr"] - fpr_priv_local
        df_disp_local["eod_max_abs"] = np.maximum(df_disp_local["eod_tpr_diff"].abs(), df_disp_local["eod_fpr_diff"].abs())
        per_group_with_diffs_local = df_disp_local.reset_index(drop=True)

        #Helper to avoid bad data causing errors due to NaNs
        def _gap(s: pd.Series) -> float:
            s = s.replace([np.inf, -np.inf], np.nan).dropna()
            return float(s.max() - s.min()) if not s.empty else float("nan")

        return FairnessResults(
            model=fitted_fair_model.model_type,
            groups=[GroupRates(**row) for row in df_for_metrics_local.to_dict(orient="records")],
            demographic_parity_gap=_gap(df_for_metrics_local["positive_rate"]),
            equalized_odds_gap_tpr=_gap(df_for_metrics_local["tpr"]),
            equalized_odds_gap_fpr=_gap(df_for_metrics_local["fpr"]),
            equal_opportunity_gap=_gap(df_for_metrics_local["tpr"]),
            per_group_df=per_group_with_diffs_local,
            dropped_groups=dropped_groups_local or [],
            kept_groups_summary=(kept_summary_local if kept_summary_local is not None else pd.DataFrame({"group": [], "n": []})),
        )



    #Filter out small intersectional groups pre-training
    inter_series = df[protected_1].astype(str) + "|" + df[protected_2].astype(str)
    counts = inter_series.value_counts()
    if min_group_size > 0:
        keep_groups = counts[counts >= min_group_size].index
        dropped_groups = counts[counts < min_group_size].index.tolist()
        df = df[inter_series.isin(keep_groups)].copy()
        #refresh series/counts to reflect the filtered df
        inter_series = df[protected_1].astype(str) + "|" + df[protected_2].astype(str)
        counts = inter_series.value_counts()
    else:
        keep_groups = counts.index
        dropped_groups = []

    #pre-training summary of group sizes
    kept_summary = counts.rename("n").reset_index().rename(columns={"index": "group"})

    #Recompute binary target and intersectional groups after filtering
    y = (df[outcome].values == positive_label).astype(int)
    inter = (df[protected_1].astype(str) + "|" + df[protected_2].astype(str)).values


    #Remove protected groups from feature set
    if features is None:
        X = df.drop(columns=[outcome, protected_1, protected_2])
        feature_cols = X.columns.tolist()
    else:
        #strip protecteds/target if a user accidentally included them
        feature_cols = [c for c in features if c not in (outcome, protected_1, protected_2)]
        X = df[feature_cols].copy()

    #Drop columns that are entirely NaN after filtering
    all_nan_cols = [c for c in feature_cols if X[c].isna().all()]
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)
        feature_cols = [c for c in feature_cols if c not in all_nan_cols]
    if not feature_cols:
        raise ValueError("No usable feature columns remain after filtering and dropping all-NaN columns.")



    p1 = df[protected_1].astype(str).values
    p2 = df[protected_2].astype(str).values

    # ---------------------------------------------------------
    # Build or validate the train/test dataframes
    # ---------------------------------------------------------

    if (train_df is None) ^ (test_df is None):
        raise ValueError(
            "Provide both train_df and test_df, or neither."
        )

    if train_df is not None and test_df is not None:
        # Use the caller-provided split.
        #
        # Make copies so that later modifications inside this function
        # do not alter the caller's original dataframes.
        train_eval_df = train_df.copy()
        test_eval_df = test_df.copy()

    else:
        # Split row indices rather than splitting preprocessed X arrays.
        #
        # This preserves the raw columns needed by FairModel, including:
        #   - model features
        #   - outcome
        #   - protected characteristics
        train_indices, test_indices = train_test_split(
            np.arange(len(df)),
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        train_eval_df = (
            df.iloc[train_indices]
            .copy()
            .reset_index(drop=True)
        )

        test_eval_df = (
            df.iloc[test_indices]
            .copy()
            .reset_index(drop=True)
        )
    
    # ---------------------------------------------------------
    # Validate required columns in both datasets
    # ---------------------------------------------------------

    required_columns = {
        outcome,
        protected_1,
        protected_2,
        *feature_cols,
    }

    missing_train_columns = sorted(
        required_columns.difference(train_eval_df.columns)
    )

    missing_test_columns = sorted(
        required_columns.difference(test_eval_df.columns)
    )

    if missing_train_columns:
        raise KeyError(
            "The training dataframe is missing required columns: "
            f"{missing_train_columns}"
        )

    if missing_test_columns:
        raise KeyError(
            "The test dataframe is missing required columns: "
            f"{missing_test_columns}"
        )

    # ---------------------------------------------------------
    # Remove unusable features based on training data only
    # ---------------------------------------------------------

    train_all_nan_cols = [
        col
        for col in feature_cols
        if train_eval_df[col].isna().all()
    ]

    if train_all_nan_cols:
        print(
            "Dropping feature columns that are entirely missing "
            f"in the training data: {train_all_nan_cols}"
        )

        feature_cols = [
            col
            for col in feature_cols
            if col not in train_all_nan_cols
        ]

    if not feature_cols:
        raise ValueError(
            "No usable feature columns remain in the training data."
        )

    # ---------------------------------------------------------
    # Create or reuse the FairModel
    # ---------------------------------------------------------

    if fair_model is None:
        # No model was supplied, so FairLogue trains one using
        # only the training dataframe.
        fitted_fair_model = FairModel.fit_from_dataframe(
            train_df=train_eval_df,
            name=(
                "fairlogue_component1_"
                f"{model_type.lower()}"
            ),
            outcome_col=outcome,
            features=feature_cols,
            protected_cols=[
                protected_1,
                protected_2,
            ],
            model_type=model_type,
            model_params=model_params,
            positive_label=positive_label,
            threshold=threshold,
            random_state=random_state,
        )

    else:
        # A fitted FairModel was supplied, usually from FairSelect.
        # Do not call fit again.
        fitted_fair_model = fair_model

        if not isinstance(fitted_fair_model, FairModel):
            raise TypeError(
                "fair_model must be an instance of FairModel."
            )

        if fitted_fair_model.features is None:
            raise ValueError(
                "The supplied FairModel does not define its "
                "'features' attribute."
            )

        missing_model_features = [
            col
            for col in fitted_fair_model.features
            if col not in test_eval_df.columns
        ]

        if missing_model_features:
            raise KeyError(
                "The test dataframe is missing features required "
                "by the supplied FairModel: "
                f"{missing_model_features}"
            )
    
    # ---------------------------------------------------------
    # Create test labels and protected-group arrays
    # ---------------------------------------------------------

    y_test = (
        test_eval_df[outcome].to_numpy()
        == positive_label
    ).astype(int)

    g_test = (
        test_eval_df[protected_1].astype(str)
        + "|"
        + test_eval_df[protected_2].astype(str)
    ).to_numpy()

    p1_test = (
        test_eval_df[protected_1]
        .astype(str)
        .to_numpy()
    )

    p2_test = (
        test_eval_df[protected_2]
        .astype(str)
        .to_numpy()
    )

    # These are retained for compatibility and possible
    # diagnostic output, even though they are not currently
    # used in the fairness calculations.
    y_train = (
        train_eval_df[outcome].to_numpy()
        == positive_label
    ).astype(int)

    g_train = (
        train_eval_df[protected_1].astype(str)
        + "|"
        + train_eval_df[protected_2].astype(str)
    ).to_numpy()

    p1_train = (
        train_eval_df[protected_1]
        .astype(str)
        .to_numpy()
    )

    p2_train = (
        train_eval_df[protected_2]
        .astype(str)
        .to_numpy()
    )

    # ---------------------------------------------------------
    # Predict using the FairModel interface
    # ---------------------------------------------------------

    proba = np.asarray(
        fitted_fair_model.predict_proba(test_eval_df),
        dtype=float,
    ).reshape(-1)

    if len(proba) != len(test_eval_df):
        raise ValueError(
            "FairModel.predict_proba() returned "
            f"{len(proba)} scores for "
            f"{len(test_eval_df)} test observations."
        )

    if not np.all(np.isfinite(proba)):
        bad_count = int(
            np.sum(~np.isfinite(proba))
        )

        raise ValueError(
            "FairModel.predict_proba() returned "
            f"{bad_count} non-finite scores."
        )

    decision_threshold = (
        float(threshold)
        if threshold is not None
        else float(fitted_fair_model.threshold)
    )

    y_hat = (
        proba >= decision_threshold
    ).astype(int)

    try:
        auroc = roc_auc_score(y_test, proba)
    except Exception:
        auroc = float("nan")   #if y_test has a single class

    accuracy = float(accuracy_score(y_test, y_hat))

    #quick console summary of model performance
    print(f"[Model performance] accuracy={accuracy:.3f} | AUROC={auroc:.3f}")

    #Metrics by group (on test)
    g_series = pd.Series(g_test, index=np.arange(len(g_test)))
    group_rates = _compute_group_rates(y_true=y_test, y_pred=y_hat, groups=g_series)
    df_groups = pd.DataFrame([gr.__dict__ for gr in group_rates])
    

    df_groups_filtered = df_groups.copy()

    #small-n filter on the test fold (Avoid reporting groups under a certain size)
    if min_group_size > 0:
        df_groups_filtered = df_groups_filtered[df_groups_filtered["n"] >= min_group_size]

    #optional class-balance requirement on the test fold (Avoids reporting NaN or 0 rates)
    if require_class_balance:
        df_groups_filtered = df_groups_filtered[
            (df_groups_filtered["pos_true"] >= 1) & (df_groups_filtered["neg_true"] >= 1)
        ]

    #If everything got filtered out, keep the original (but warn in metrics)
    if df_groups_filtered.empty:
        df_for_metrics = df_groups.copy()
        filtered_note = True
    else:
        df_for_metrics = df_groups_filtered
        filtered_note = False
        
    #Find the privileged group (largest n in filtered view)    
    if df_for_metrics.empty:
        privileged_group = None
        tpr_priv = fpr_priv = np.nan
    else:
        privileged_group = df_for_metrics.sort_values("n", ascending=False).iloc[0]["group"]
        tpr_priv = float(df_for_metrics.loc[df_for_metrics["group"] == privileged_group, "tpr"].iloc[0])
        fpr_priv = float(df_for_metrics.loc[df_for_metrics["group"] == privileged_group, "fpr"].iloc[0])

    #Find per-group disparities vs privileged
    df_disp = df_for_metrics.copy()
    df_disp["eo_diff"] = tpr_priv - df_disp["tpr"]                      #Equal Opportunity difference (TPR)
    df_disp["eod_tpr_diff"] = tpr_priv - df_disp["tpr"]                #EOds component on TPR
    df_disp["eod_fpr_diff"] = df_disp["fpr"] - fpr_priv                #EOds component on FPR (higher is worse)
    df_disp["eod_max_abs"] = np.maximum(df_disp["eod_tpr_diff"].abs(),
                                        df_disp["eod_fpr_diff"].abs())

    #Push these back onto results table
    per_group_with_diffs = df_disp.copy().reset_index(drop=True)

    #Fix gaps with NaN values (from single-class groups)
    def _gap(s: pd.Series) -> float:
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        return float(s.max() - s.min()) if not s.empty else float("nan")
    

    if filtered_note:
        print("Warning: All groups were filtered out by min_group_size or require_class_balance. "
              "Returning metrics on unfiltered groups, which may include NaN rates.")

    
    results = _compute_fairness_for_groups(
        g_test,
        dropped_groups_local=dropped_groups,
        kept_summary_local=kept_summary.reset_index(drop=True),
    )

    non_intersectional = None
    if return_non_intersectional:
        p1_summary = pd.Series(p1, name="group").value_counts().rename("n").reset_index().rename(columns={"index": "group"})
        p2_summary = pd.Series(p2, name="group").value_counts().rename("n").reset_index().rename(columns={"index": "group"})

        non_intersectional = {
            protected_1: _compute_fairness_for_groups(p1_test, kept_summary_local=p1_summary),
            protected_2: _compute_fairness_for_groups(p2_test, kept_summary_local=p2_summary),
        }

    #Plots use the same filtered view
    figs: Dict[str, plt.Figure] = {}
    if make_plots:
        base = df_for_metrics.set_index("group")
        figs["demographic_parity"] = _plot_bar(
            base["positive_rate"], "Demographic Parity by Group", "P(Ŷ=1)"
        )
        #Per-group Equal Opportunity difference (TPR vs privileged)
        figs["eo_diff_by_group"] = _plot_bar_series_by_group(
            df=per_group_with_diffs,
            value_col="eo_diff",
            title=f"Equal Opportunity Difference by Group (privileged: {privileged_group})",
            ylabel="TPR_priv - TPR_group"
        )

        #Per-group Equalized Odds (single-number, max abs of TPR/FPR diffs)
        figs["eods_maxabs_by_group"] = _plot_bar_series_by_group(
            df=per_group_with_diffs,
            value_col="eod_max_abs",
            title=f"Equalized Odds (max |TPR/FPR diff|) by Group (privileged: {privileged_group})",
            ylabel="max(|ΔTPR|, |ΔFPR|)"
        )

        figs["eods_components_grouped"] = _plot_grouped_eods_components(results.per_group_df)

        figs["fairness_landscape"] = _plot_fairness_matrix(
            per_group_with_diffs,
            metric_cols=[
                "positive_rate", "tpr", "fpr",
                "eo_diff", "eod_fpr_diff", "eod_max_abs",
            ],
            title=f"Fairness Metrics Matrix (privileged: {privileged_group})",
            annotate=True,
            sort_by="eod_max_abs",
            max_groups=40,      # tune as needed; set None for all groups
            normalize="zscore", # or "none" if you prefer raw-color scaling
        )

        
    if return_intermediates:
        intermediates = {
            "y_test": y_test,
            "y_hat": y_hat,
            "groups_test": g_test,
            "protected_1_test": p1_test,
            "protected_2_test": p2_test,
            "proba": proba,
            "fair_model": fitted_fair_model,
            "model_metrics": {
                "accuracy": accuracy,
                "auroc": auroc,
                "threshold": decision_threshold,
                "test_size": (
                    len(test_eval_df)
                    / (
                        len(train_eval_df)
                        + len(test_eval_df)
                    )
                ),
                "model_type": (
                    fitted_fair_model.model_type
                ),
            },
            "non_intersectional": non_intersectional,
        }
       return results, figs, intermediates

    return results, figs
