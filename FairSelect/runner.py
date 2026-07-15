# runner.py
from __future__ import annotations

from dataclasses import dataclass, field
import traceback
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
from .utils import confusion_rates, filter_intersectional_groups
import FairLogue as ift
import pandas as pd
from FairModel import FairModel

from .core import RunResult, split_data

from .techniques_pre import run_reweighting, run_smote_or_ros, run_local_massaging
from .techniques_in import (
    run_baseline,
    run_compositional_models,
    run_group_balanced_ensemble,
    run_multicalibration,
    run_reductions_meta,
    run_prejudice_remover,
)
from .techniques_post import (
    run_group_youden_postproc,
    run_multiaccuracy_boost,
    run_reject_option_shift,
    run_input_repair,
    run_reject_option_kamiran,
)
from .techniques_combined import run_combined_pipeline


# These keys match the GUI checkboxes exactly (so you can reuse saved configs).
ALL_TECHNIQUES: Sequence[str] = (
    "Pre:Reweight (y,a)",
    "Pre:SMOTE / Oversample",
    "Pre:Local Massaging",
    "In:Compositional per-group",
    "In:Ensemble (K=5)",
    "In:Multicalibration (isotonic)",
    "In:Reductions (EO)",
    "In:Fairness Regularization (Prejudice Remover)",
    "Post:Youden per group",
    "Post:Multiaccuracy Boost",
    "Post:Reject-Option Shift",
    "Post:Input Repair",
    "Post:Reject-Option Kamiran",
)


@dataclass(frozen=True)
class PipelineConfig:
    """
    Everything the GUI used to collect is now passed as a config object.

    - df_or_path: a DataFrame OR a CSV path
    - target: label column name
    - protected: list of protected attribute column name(s)
    - features: list of feature column names (should NOT include target/protected)
    - model_name: must be compatible with build_estimator() inside core.py
    - model_params: kwargs passed into the estimator builder
    - techniques: list of technique keys (see ALL_TECHNIQUES)
    - run_baseline: whether to run the pooled baseline model
    - run_combined: whether to run the combined pipeline
    - split kwargs: forwarded to split_data
    - fairlogue_comp1: whether to run FairLogue component 1 for each technique and include in the RunResult
    - fairlogue_comp2: whether to run FairLogue component 2 for each technique and include in the RunResult
    - fairlogue_comp3: whether to run FairLogue component 3 for each technique and include in the RunResult
    """
    df_or_path: Union[pd.DataFrame, str]
    target: str
    protected: Sequence[str]
    features: Sequence[str]
    model_name: str
    model_params: Dict[str, Any] = field(default_factory=dict)

    techniques: Sequence[str] = field(default_factory=list)
    run_baseline: bool = True
    run_combined: bool = False
    min_group_size: int = 20
    require_outcome_coverage: bool = True
    filter_small_groups: bool = True

    test_size: float = 0.25
    val_size: float = 0.2
    random_state: int = 42
    fairlogue_comp1: bool = False
    fairlogue_comp2: bool = False
    fairlogue_comp3: bool = False

    fairlogue_comp3_method: str = "sr"
    fairlogue_comp3_n_splits: int = 5
    fairlogue_comp3_gen_null: bool = False
    fairlogue_comp3_R_null: int = 100
    fairlogue_comp3_bootstrap: str = "none"
    fairlogue_comp3_B: int = 100

def _load_df(df_or_path: Union[pd.DataFrame, str]) -> pd.DataFrame:
    if isinstance(df_or_path, pd.DataFrame):
        return df_or_path.copy()
    if isinstance(df_or_path, str):
        return pd.read_csv(df_or_path)
    raise TypeError("df_or_path must be a pandas DataFrame or a CSV file path (str).")


def _normalize_features(
    *,
    df: pd.DataFrame,
    target: str,
    protected: Sequence[str],
    features: Sequence[str],
) -> List[str]:
    # Match GUI behavior: exclude target and protected from features. :contentReference[oaicite:3]{index=3}
    f = [c for c in features if c != target and c not in protected]
    if len(f) < 1:
        raise ValueError("features must include at least one column not in target/protected.")
    missing = [c for c in [*f, *protected, target] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")
    return f


def _selected_dict(techniques: Sequence[str]) -> Dict[str, bool]:
    s = set(techniques)
    return {k: (k in s) for k in ALL_TECHNIQUES}


def run_pipeline(cfg: PipelineConfig) -> List[RunResult]:
    """
    Single entrypoint that replaces the GUI run button.

    Returns a list of RunResult objects in the same style/order the GUI produced.
    """
    df = _load_df(cfg.df_or_path)

    protected = list(cfg.protected)
    features = _normalize_features(df=df, target=cfg.target, protected=protected, features=cfg.features)
    filter_note = ""

    if cfg.filter_small_groups:
        df, removed_groups, filter_note = filter_intersectional_groups(
            df=df,
            protected_cols=protected,
            target_col=cfg.target,
            min_group_size=cfg.min_group_size,
            require_outcome_coverage=cfg.require_outcome_coverage,
        )

        print("\n[FairSelect group filtering]")
        print(filter_note)

        if len(removed_groups) > 0:
            print("\nRemoved intersectional groups:")
            print(removed_groups.to_string(index=False))

        if df.empty:
            raise ValueError(
                "All rows were removed by the intersectional group filter. "
                "Lower min_group_size or disable require_outcome_coverage."
            )

    # Prepare data & splits (same call pattern as GUI). :contentReference[oaicite:4]{index=4}
    X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te = split_data(
        df[[*features, *protected, cfg.target]],
        cfg.target,
        protected,
        features,
        test_size=cfg.test_size,
        val_size=cfg.val_size,
        random_state=cfg.random_state,
    )

    # Keep full train (GUI does train+val concat). :contentReference[oaicite:5]{index=5}
    all_df_train = pd.concat([X_tr, X_va], axis=0)

    model_name = cfg.model_name
    params = dict(cfg.model_params)

    results: List[RunResult] = []

    # Baseline
    if cfg.run_baseline:
        results.append(
            run_baseline(
                model_name, params,
                X_tr, X_va, X_te, y_tr, y_va, y_te,
                A_tr, A_va, A_te,
                protected, all_df_train, outcome_col=cfg.target,
            )
        )

    # Technique dispatch mirrors GUI exactly. :contentReference[oaicite:6]{index=6} :contentReference[oaicite:7]{index=7}
    selected = _selected_dict(cfg.techniques)

    # Pre
    if selected["Pre:Reweight (y,a)"]:
        results.append(run_reweighting(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected, all_df_train, outcome_col=cfg.target))
    if selected["Pre:SMOTE / Oversample"]:
        results.append(run_smote_or_ros(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected, all_df_train, outcome_col=cfg.target))
    if selected["Pre:Local Massaging"]:
        results.append(run_local_massaging(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected, all_df_train, outcome_col=cfg.target))

    # In
    if selected["In:Compositional per-group"]:
        results.append(run_compositional_models(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected, all_df_train, outcome_col=cfg.target))
    if selected["In:Ensemble (K=5)"]:
        results.append(run_group_balanced_ensemble(model_name, params, 5, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected, all_df_train, outcome_col=cfg.target))
    if selected["In:Multicalibration (isotonic)"]:
        results.append(run_multicalibration(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected, all_df_train, outcome_col=cfg.target))
    if selected["In:Reductions (EO)"]:
        results.append(run_reductions_meta(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected, all_df_train, constraint="EO"))
    if selected["In:Fairness Regularization (Prejudice Remover)"]:
        results.append(run_prejudice_remover(
            model_name, params,
            X_tr, X_va, X_te, y_tr, y_va, y_te,
            A_tr, A_va, A_te,
            protected, all_df_train,
            eta=25.0,
            outcome_col=cfg.target
        ))

    # Post
    if selected["Post:Youden per group"]:
        results.append(run_group_youden_postproc(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected, all_df_train, outcome_col=cfg.target))
    if selected["Post:Multiaccuracy Boost"]:
        results.append(run_multiaccuracy_boost(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected, all_df_train, outcome_col=cfg.target))
    if selected["Post:Reject-Option Shift"]:
        results.append(run_reject_option_shift(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected, all_df_train, outcome_col=cfg.target))
    if selected["Post:Input Repair"]:
        results.append(run_input_repair(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected, all_df_train, outcome_col=cfg.target))
    if selected["Post:Reject-Option Kamiran"]:
        results.append(run_reject_option_kamiran(model_name, params, X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te, protected, all_df_train, fairness_objective="eod", 
                                                 fairness_bound=0.05, max_acc_drop=0.02, outcome_col=cfg.target))

    # Combined
    if cfg.run_combined:
        combined_rr = run_combined_pipeline(
            model_name, params,
            X_tr, X_va, X_te, y_tr, y_va, y_te,
            A_tr, A_va, A_te,
            protected, all_df_train,
            selected=selected,
            outcome_col=cfg.target
        )
        results.append(combined_rr)
    
    for r in results:
        if filter_note:
            r.notes = (r.notes + "\n" + filter_note).strip()

    if cfg.fairlogue_comp1 or cfg.fairlogue_comp3:
            for r in results:
                r.fairlogue = {}

                if cfg.fairlogue_comp1:
                    r.fairlogue["component1"] = _run_fairlogue_component1_for_result(
                        rr=r,
                        df=df,
                        target=cfg.target,
                        protected=protected,
                        features=features,
                    )

                if cfg.fairlogue_comp3:
                   r.fairlogue["component3"] = _run_fairlogue_component3_for_result(
                        rr=r,
                        df=df,
                        target=cfg.target,
                        protected=protected,
                        features=features,
                        method=cfg.fairlogue_comp3_method,
                        n_splits=cfg.fairlogue_comp3_n_splits,
                        gen_null=cfg.fairlogue_comp3_gen_null,
                        R_null=cfg.fairlogue_comp3_R_null,
                        bootstrap=cfg.fairlogue_comp3_bootstrap,
                        B=cfg.fairlogue_comp3_B,
                        random_state=cfg.random_state,
                    )


    return results



def _run_fairlogue_component1_for_result(
    *,
    rr: RunResult,
    df: pd.DataFrame,
    target: str,
    protected: list[str],
    features: list[str],
):
    """
    Run FairLogue Component 1 using the fitted FairModel attached to a FairSelect result.

    Component 1 should evaluate observed/intersectional fairness from the
    model's predictions, not refit a new model.
    """

    if getattr(rr, "fair_model", None) is None:
        return {
            "status": "skipped",
            "reason": "RunResult has no fair_model attached.",
        }

    fair_model = rr.fair_model

    if getattr(rr, "test_index", None) is None:
        return {
            "status": "skipped",
            "component": "FairLogue Component 1",
            "reason": (
                "RunResult does not contain test indices. "
                "The FairLogue audit was not run to avoid "
                "evaluating on training observations."
            ),
        }

    missing_test_indices = [
        idx for idx in rr.test_index
        if idx not in df.index
    ]

    if missing_test_indices:
        return {
            "status": "failed",
            "component": "FairLogue Component 1",
            "reason": (
                f"{len(missing_test_indices)} test indices were "
                "not found in the audit DataFrame."
            ),
        }

    audit_df = df.loc[rr.test_index].copy()

    # Use the FairModel generated by FairSelect
    audit_df["_fairselect_score"] = fair_model.predict_proba(audit_df)
    audit_df["_fairselect_pred"] = fair_model.predict(audit_df)

    # Create intersectional group
    audit_df["_intersectional_group"] = (
        audit_df[protected]
        .astype(str)
        .agg("|".join, axis=1)
    )

    # Minimal Component 1-style observed audit
    # This avoids refitting anything.
    rows = []

    for g, sub in audit_df.groupby("_intersectional_group"):
        y_true = sub[target].astype(int).to_numpy()
        yhat = sub["_fairselect_pred"].astype(int).to_numpy()
        p = sub["_fairselect_score"].astype(float).to_numpy()

        cr = confusion_rates(y_true, yhat)

        rows.append({
            "group": str(g),
            "n": int(len(sub)),
            "prevalence": float(y_true.mean()) if len(y_true) else None,
            "predicted_positive_rate": float(yhat.mean()) if len(yhat) else None,
            "mean_score": float(p.mean()) if len(p) else None,
            "TPR": cr.get("TPR"),
            "FPR": cr.get("FPR"),
            "TNR": cr.get("TNR"),
            "FNR": cr.get("FNR"),
            "PPV": cr.get("PPV"),
            "NPV": cr.get("NPV"),
        })

    group_df = pd.DataFrame(rows)

    return {
        "status": "ok",
        "component": "FairLogue Component 1",
        "audit_source": "FairSelect FairModel",
        "model_name": getattr(fair_model, "name", rr.name),
        "group_stats": group_df,
    }


def _run_fairlogue_component3_for_result(
    *,
    rr: RunResult,
    df: pd.DataFrame,
    target: str,
    protected: list[str],
    features: list[str],
    method: str = "sr",
    n_splits: int = 5,
    gen_null: bool = False,
    R_null: int = 100,
    bootstrap: str = "none",
    B: int = 100,
    random_state: int = 42,
):
    """
    Run FairLogue Component 3 using the fitted FairModel attached
    to a FairSelect result.
    """
    if getattr(rr, "fair_model", None) is None:
        return {
            "status": "skipped",
            "reason": "RunResult has no fair_model attached.",
        }

    fair_model = rr.fair_model
    if getattr(rr, "test_index", None) is None:
        return {
            "status": "skipped",
            "component": "FairLogue Component 3",
            "reason": (
                "RunResult has no test_index. Audit was not "
                "run to avoid using training observations."
            ),
        }

    missing_test_indices = [
        idx for idx in rr.test_index
        if idx not in df.index
    ]

    if missing_test_indices:
        return {
            "status": "failed",
            "component": "FairLogue Component 3",
            "reason": (
                f"{len(missing_test_indices)} test indices "
                "were not found in df."
            ),
        }

    audit_df = df.loc[rr.test_index].copy()

    if len(protected) < 2:
        return {
            "status": "skipped",
            "reason": (
                "Component 3 currently expects at least two "
                "protected characteristics."
            ),
        }

    try:
        try:
            from FairLogue.Component3.model import Model as Component3Model
        except ImportError:
            from combined_toolkits.FairLogue.Component3.model import (
                Model as Component3Model
            )

        m_c3 = Component3Model(
            data=audit_df,
            outcome=target,
            protected_characteristics=tuple(protected[:2]),
            covariates=list(features),
            fair_model=fair_model,
            method=method,
            n_splits=n_splits,
            random_state=random_state,
        )

        m_c3.pre_process_data()

        res = m_c3.fit_fairness_from_fairmodel(
            cutoff=getattr(fair_model, "threshold", 0.5),
            gen_null=gen_null,
            R_null=R_null,
            bootstrap=bootstrap,
            B=B,
        )

        summary = m_c3.summarize()

        return {
            "status": "ok",
            "component": "FairLogue Component 3",
            "audit_source": "FairSelect FairModel",
            "model_name": getattr(fair_model, "name", rr.name),
            "method": method,
            "gen_null": gen_null,
            "R_null": R_null,
            "results": res,
            "summary": summary,
        }

    except Exception as exc:
        return {
            "status": "failed",
            "component": "FairLogue Component 3",
            "model_name": getattr(fair_model, "name", rr.name),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }