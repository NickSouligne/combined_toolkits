# runner.py
from __future__ import annotations

from dataclasses import dataclass, field
import traceback
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union, Callable
from .utils import confusion_rates, filter_intersectional_groups
from FairLogue.Component1.intersectional_metrics import (
    evaluate_intersectional_fairness,
)
import pandas as pd
from FairModel import FairModel

from .core import RunResult, split_data, FairSelectPlanSpec

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
    include_protected_features: bool = False
    model_params: Dict[str, Any] = field(default_factory=dict)
    

    techniques: Sequence[str] = field(default_factory=list)
    run_baseline: bool = True
    run_combined: bool = False
    min_group_size: int = 20
    require_outcome_coverage: bool = True
    filter_small_groups: bool = True
    train_index: Optional[Sequence[Any]] = None
    validation_index: Optional[Sequence[Any]] = None
    test_index: Optional[Sequence[Any]] = None

    test_size: float = 0.25
    val_size: float = 0.2
    random_state: int = 42
    fairlogue_comp1: bool = False
    fairlogue_comp2: bool = False
    fairlogue_comp3: bool = False
    
    # FairLogue Component 1 settings
    fairlogue_comp1_make_plots: bool = True
    fairlogue_comp1_return_non_intersectional: bool = True
    fairlogue_comp1_min_group_size: int = 0
    fairlogue_comp1_require_class_balance: bool = True

    fairlogue_comp3_method: str = "sr"
    fairlogue_comp3_n_splits: int = 5
    fairlogue_comp3_gen_null: bool = False
    fairlogue_comp3_R_null: int = 100
    fairlogue_comp3_bootstrap: str = "none"
    fairlogue_comp3_B: int = 100
    fairlogue_comp3_m_factor: float = 0.75

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
    include_protected_features: bool = False,
) -> List[str]:
    features = [column for column in features if column != target]

    if include_protected_features:
        features = list(dict.fromkeys([*features, *protected]))
    else:
        features = [column for column in features if column not in protected]

    if not features:
        raise ValueError("No model features remain after feature normalization.")

    missing = [column for column in [*features, *protected, target] if column not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in df: {sorted(set(missing))}")

    return features


def _selected_dict(techniques: Sequence[str]) -> Dict[str, bool]:
    s = set(techniques)
    return {k: (k in s) for k in ALL_TECHNIQUES}

def _execute_fairselect_plan(
    *,
    plan_type: str,
    technique: Optional[str],
    selected: Dict[str, bool],
    model_name: str,
    model_params: Dict[str, Any],
    X_tr: pd.DataFrame,
    X_va: pd.DataFrame,
    X_te: pd.DataFrame,
    y_tr: pd.Series,
    y_va: pd.Series,
    y_te: pd.Series,
    A_tr: pd.Series,
    A_va: pd.Series,
    A_te: pd.Series,
    protected: Sequence[str],
    all_df_train: pd.DataFrame,
    outcome_col: str,
) -> RunResult:
    """
    Execute exactly one FairSelect plan.

    This is the sole dispatch point used by both the observed pipeline
    and Component 3 permutation-null refits.
    """
    protected = list(protected)
    params = deepcopy(model_params)

    common_args = (
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
        protected,
        all_df_train,
    )

    if plan_type == "baseline":
        return run_baseline(
            *common_args,
            outcome_col=outcome_col,
        )

    if plan_type == "combined":
        return run_combined_pipeline(
            *common_args,
            selected=dict(selected),
            outcome_col=outcome_col,
        )

    if plan_type != "single":
        raise ValueError(
            "plan_type must be one of "
            "{'baseline', 'single', 'combined'}. "
            f"Received {plan_type!r}."
        )

    if technique is None:
        raise ValueError(
            "A single-technique plan requires technique."
        )

    if technique == "Pre:Reweight (y,a)":
        return run_reweighting(
            *common_args,
            outcome_col=outcome_col,
        )

    if technique == "Pre:SMOTE / Oversample":
        return run_smote_or_ros(
            *common_args,
            outcome_col=outcome_col,
        )

    if technique == "Pre:Local Massaging":
        return run_local_massaging(
            *common_args,
            outcome_col=outcome_col,
        )

    if technique == "In:Compositional per-group":
        return run_compositional_models(
            *common_args,
            outcome_col=outcome_col,
        )

    if technique == "In:Ensemble (K=5)":
        return run_group_balanced_ensemble(
            model_name,
            params,
            5,
            X_tr,
            X_va,
            X_te,
            y_tr,
            y_va,
            y_te,
            A_tr,
            A_va,
            A_te,
            protected,
            all_df_train,
            outcome_col=outcome_col,
        )

    if technique == "In:Multicalibration (isotonic)":
        return run_multicalibration(
            *common_args,
            outcome_col=outcome_col,
        )

    if technique == "In:Reductions (EO)":
        return run_reductions_meta(
            *common_args,
            constraint="EO",
        )

    if (
        technique
        == "In:Fairness Regularization (Prejudice Remover)"
    ):
        return run_prejudice_remover(
            *common_args,
            eta=25.0,
            outcome_col=outcome_col,
        )

    if technique == "Post:Youden per group":
        return run_group_youden_postproc(
            *common_args,
            outcome_col=outcome_col,
        )

    if technique == "Post:Multiaccuracy Boost":
        return run_multiaccuracy_boost(
            *common_args,
            outcome_col=outcome_col,
        )

    if technique == "Post:Reject-Option Shift":
        return run_reject_option_shift(
            *common_args,
            outcome_col=outcome_col,
        )

    if technique == "Post:Input Repair":
        return run_input_repair(
            *common_args,
            outcome_col=outcome_col,
        )

    if technique == "Post:Reject-Option Kamiran":
        return run_reject_option_kamiran(
            *common_args,
            fairness_objective="eod",
            fairness_bound=0.05,
            max_acc_drop=0.02,
            outcome_col=outcome_col,
        )

    raise ValueError(
        f"Unsupported FairSelect technique: {technique!r}."
    )


def _attach_refit_spec(
        result: RunResult,
        spec: FairSelectPlanSpec,
    ) -> RunResult:
        result.refit_spec = spec

        if result.fair_model is not None:
            result.fair_model.refit_spec = spec

            result.fair_model.metadata[
                "fairselect_plan_type"
            ] = spec.plan_type

            result.fair_model.metadata[
                "fairselect_technique"
            ] = spec.technique

            result.fair_model.metadata[
                "fairselect_selected"
            ] = spec.selected_dict()

        return result

def refit_fairmodel_from_spec(
    *,
    df: pd.DataFrame,
    spec: FairSelectPlanSpec,
    random_state: int,
) -> FairModel:
    """
    Refit one exact FairSelect plan on a supplied DataFrame.

    The DataFrame is expected to contain the same row indices and columns
    as the cohort used for the observed fit. Protected characteristics may
    have been permuted by FairLogue Component 3.
    """
    required_columns = list(
        dict.fromkeys(
            [
                *spec.features,
                *spec.protected,
                spec.target,
            ]
        )
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The null-refit DataFrame is missing columns: "
            f"{missing_columns}"
        )

    required_indices = [
        *spec.train_index,
        *spec.validation_index,
        *spec.test_index,
    ]

    missing_indices = [
        index
        for index in required_indices
        if index not in df.index
    ]

    if missing_indices:
        raise ValueError(
            "The null-refit DataFrame is missing stored split "
            f"indices: {missing_indices[:20]}"
        )

    features = list(spec.features)
    protected = list(spec.protected)

    X_tr = df.loc[
        list(spec.train_index),
        features,
    ].copy()

    X_va = df.loc[
        list(spec.validation_index),
        features,
    ].copy()

    X_te = df.loc[
        list(spec.test_index),
        features,
    ].copy()

    y_tr = df.loc[
        list(spec.train_index),
        spec.target,
    ].copy()

    y_va = df.loc[
        list(spec.validation_index),
        spec.target,
    ].copy()

    y_te = df.loc[
        list(spec.test_index),
        spec.target,
    ].copy()

    A_tr = (
        df.loc[
            list(spec.train_index),
            protected,
        ]
        .astype(str)
        .agg("|".join, axis=1)
    )

    A_va = (
        df.loc[
            list(spec.validation_index),
            protected,
        ]
        .astype(str)
        .agg("|".join, axis=1)
    )

    A_te = (
        df.loc[
            list(spec.test_index),
            protected,
        ]
        .astype(str)
        .agg("|".join, axis=1)
    )

    all_df_train = pd.concat(
        [X_tr, X_va],
        axis=0,
    )

    model_params = deepcopy(
        dict(spec.model_params)
    )

    # All model builders should receive the null-replication seed.
    model_params["random_state"] = int(random_state)

    result = _execute_fairselect_plan(
        plan_type=spec.plan_type,
        technique=spec.technique,
        selected=spec.selected_dict(),
        model_name=spec.model_name,
        model_params=model_params,
        X_tr=X_tr,
        X_va=X_va,
        X_te=X_te,
        y_tr=y_tr,
        y_va=y_va,
        y_te=y_te,
        A_tr=A_tr,
        A_va=A_va,
        A_te=A_te,
        protected=protected,
        all_df_train=all_df_train,
        outcome_col=spec.target,
    )

    if result.fair_model is None:
        raise RuntimeError(
            "The refitted FairSelect plan did not return a FairModel."
        )

    result.refit_spec = spec
    result.fair_model.refit_spec = spec

    result.fair_model.metadata[
        "component3_null_refit"
    ] = True

    result.fair_model.metadata[
        "component3_null_random_state"
    ] = int(random_state)

    return result.fair_model



def _make_refit_spec(
        *,
        plan_type: str,
        technique: Optional[str],
        selected: Dict[str, bool],
        cfg: PipelineConfig,
        features: Sequence[str],
        protected: Sequence[str],
        train_index: Sequence[Any],
        validation_index: Sequence[Any],
        test_index: Sequence[Any],
    ) -> FairSelectPlanSpec:
        return FairSelectPlanSpec(
            plan_type=str(plan_type),
            technique=technique,
            selected=tuple(
                (key, bool(selected.get(key, False)))
                for key in ALL_TECHNIQUES
            ),
            model_name=str(cfg.model_name),
            model_params=deepcopy(
                dict(cfg.model_params)
            ),
            target=str(cfg.target),
            protected=tuple(protected),
            features=tuple(features),
            include_protected_features=bool(
                cfg.include_protected_features
            ),
            train_index=tuple(train_index),
            validation_index=tuple(validation_index),
            test_index=tuple(test_index),
            random_state=int(cfg.random_state),
        )


def run_pipeline(cfg: PipelineConfig) -> List[RunResult]:
    """
    Single entrypoint that replaces the GUI run button.

    Returns a list of RunResult objects in the same style/order the GUI produced.
    """
    if isinstance(cfg.df_or_path, pd.DataFrame):
        df = cfg.df_or_path.copy()
    else:
        df = pd.read_csv(cfg.df_or_path)

    target = cfg.target
    protected = list(cfg.protected)

    features = _normalize_features(
        df=df,
        target=target,
        protected=protected,
        features=cfg.features,
        include_protected_features=cfg.include_protected_features,
    )

    # ---------------------------------------------------------
    # Validate required columns
    # ---------------------------------------------------------
    required_columns = list(
        dict.fromkeys(
            features
            + protected
            + [target]
        )
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The input DataFrame is missing required columns: "
            f"{missing_columns}"
        )

    # ---------------------------------------------------------
    # Construct model inputs
    # ---------------------------------------------------------
    X = df[features].copy()
    y = df[target].copy()

    if len(protected) == 1:
        A = (
            df[protected[0]]
            .astype(str)
            .copy()
        )
    else:
        A = (
            df[protected]
            .astype(str)
            .agg("|".join, axis=1)
        )

    # Keep all objects on the same original DataFrame index.
    X.index = df.index
    y.index = df.index
    A.index = df.index

    if all(
        index_set is not None
        for index_set in [
            cfg.train_index,
            cfg.validation_index,
            cfg.test_index,
        ]
    ):
        train_index = list(cfg.train_index)
        validation_index = list(cfg.validation_index)
        test_index = list(cfg.test_index)

        requested = train_index + validation_index + test_index

        missing_indices = [
            index for index in requested
            if index not in df.index
        ]

        if missing_indices:
            raise ValueError(
                "Caller-provided split indices were not found in df: "
                f"{missing_indices[:20]}"
            )

        if len(set(requested)) != len(requested):
            raise ValueError(
                "Caller-provided train, validation, and test "
                "indices overlap."
            )

        X_tr = df.loc[train_index, features].copy()
        X_va = df.loc[validation_index, features].copy()
        X_te = df.loc[test_index, features].copy()

        y_tr = df.loc[train_index, cfg.target].copy()
        y_va = df.loc[validation_index, cfg.target].copy()
        y_te = df.loc[test_index, cfg.target].copy()

        A_tr = (
            df.loc[train_index, protected]
            .astype(str)
            .agg("|".join, axis=1)
        )
        A_va = (
            df.loc[validation_index, protected]
            .astype(str)
            .agg("|".join, axis=1)
        )
        A_te = (
            df.loc[test_index, protected]
            .astype(str)
            .agg("|".join, axis=1)
        )

    else:
        split_columns = list(dict.fromkeys([*features, *protected, cfg.target]))

        X_tr, X_va, X_te, y_tr, y_va, y_te, A_tr, A_va, A_te = split_data(
            df[split_columns],
            cfg.target,
            protected,
            features,
            test_size=cfg.test_size,
            val_size=cfg.val_size,
            random_state=cfg.random_state,
        )

    # Keep full train (GUI does train+val concat)
    all_df_train = pd.concat([X_tr, X_va], axis=0)

    model_name = cfg.model_name
    params = dict(cfg.model_params)

    results: List[RunResult] = []

    selected = _selected_dict(
        cfg.techniques
    )

    actual_train_index = list(X_tr.index)
    actual_validation_index = list(X_va.index)
    actual_test_index = list(X_te.index)

    def add_plan_result(
        *,
        plan_type: str,
        technique: Optional[str] = None,
    ) -> None:
        result = _execute_fairselect_plan(
            plan_type=plan_type,
            technique=technique,
            selected=selected,
            model_name=model_name,
            model_params=params,
            X_tr=X_tr,
            X_va=X_va,
            X_te=X_te,
            y_tr=y_tr,
            y_va=y_va,
            y_te=y_te,
            A_tr=A_tr,
            A_va=A_va,
            A_te=A_te,
            protected=protected,
            all_df_train=all_df_train,
            outcome_col=cfg.target,
        )

        spec = _make_refit_spec(
            plan_type=plan_type,
            technique=technique,
            selected=selected,
            cfg=cfg,
            features=features,
            protected=protected,
            train_index=actual_train_index,
            validation_index=actual_validation_index,
            test_index=actual_test_index,
        )

        results.append(
            _attach_refit_spec(
                result,
                spec,
            )
        )

    if cfg.run_baseline:
        add_plan_result(
            plan_type="baseline",
        )

    for technique in ALL_TECHNIQUES:
        if selected.get(technique, False):
            add_plan_result(
                plan_type="single",
                technique=technique,
            )

    if cfg.run_combined:
        add_plan_result(
            plan_type="combined",
        )
    
    for result in results:
        if getattr(result, "test_index", None) is None:
            result.test_index = list(
                X_te.index
            )

        if getattr(result, "y_test", None) is None:
            result.y_test = y_te.copy()

        if getattr(result, "A_test", None) is None:
            result.A_test = A_te.copy()
    

    if cfg.fairlogue_comp1 or cfg.fairlogue_comp3:
        for r in results:
            if r.fairlogue is None:
                r.fairlogue = {}

            if cfg.fairlogue_comp1:
                r.fairlogue["component1"] = (
                    _run_fairlogue_component1_for_result(
                        rr=r,
                        df=df,
                        target=cfg.target,
                        protected=protected,
                        features=features,
                        make_plots=(
                            cfg.fairlogue_comp1_make_plots
                        ),
                        return_non_intersectional=(
                            cfg
                            .fairlogue_comp1_return_non_intersectional
                        ),
                        min_group_size=(
                            cfg.fairlogue_comp1_min_group_size
                        ),
                        require_class_balance=(
                            cfg
                            .fairlogue_comp1_require_class_balance
                        ),
                        positive_label=1,
                        random_state=cfg.random_state,
                    )
                )

            if cfg.fairlogue_comp3:
                r.fairlogue["component3"] = (
                    _run_fairlogue_component3_for_result(
                        rr=r,
                        df=df,
                        target=cfg.target,
                        protected=protected,
                        features=features,
                        method=cfg.fairlogue_comp3_method,
                        n_splits=(
                            cfg.fairlogue_comp3_n_splits
                        ),
                        gen_null=(
                            cfg.fairlogue_comp3_gen_null
                        ),
                        R_null=(
                            cfg.fairlogue_comp3_R_null
                        ),
                        bootstrap=(
                            cfg.fairlogue_comp3_bootstrap
                        ),
                        B=cfg.fairlogue_comp3_B,
                        random_state=cfg.random_state,
                        m_factor=cfg.fairlogue_comp3_m_factor
                    )
                )
    for result in results:
        if result.fair_model is not None:
            result.fair_model.metadata["include_protected_features"] = cfg.include_protected_features
            result.fair_model.metadata["model_features"] = features
            result.fair_model.metadata["protected_features_in_model"] = [
                column for column in protected if column in features
            ]


    return results



def _run_fairlogue_component1_for_result(
    *,
    rr: RunResult,
    df: pd.DataFrame,
    target: str,
    protected: Sequence[str],
    features: Sequence[str],
    make_plots: bool = True,
    return_non_intersectional: bool = True,
    min_group_size: int = 0,
    require_class_balance: bool = False,
    positive_label: Any = 1,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run FairLogue Component 1 using the fitted FairModel stored
    in a FairSelect RunResult.

    This function does not reimplement fairness metrics and does
    not refit the model. It delegates the analysis directly to:

        FairLogue.Component1.intersectional_metrics
            .evaluate_intersectional_fairness()

    The FairSelect held-out test rows are supplied to FairLogue as
    test_df. All remaining rows are supplied as train_df only to
    satisfy FairLogue's caller-provided split interface and provide
    descriptive training information. The supplied FairModel is
    already fitted and is therefore not retrained.
    """

    # ---------------------------------------------------------
    # Validate the FairModel
    # ---------------------------------------------------------
    fair_model = getattr(rr, "fair_model", None)

    if fair_model is None:
        return {
            "status": "skipped",
            "component": "FairLogue Component 1",
            "reason": (
                "The FairSelect RunResult does not contain an "
                "attached FairModel."
            ),
        }

    # Ensure the model carries the correct outcome metadata.
    if getattr(fair_model, "outcome_col", None) is None:
        fair_model.outcome_col = target

    if getattr(fair_model, "positive_label", None) is None:
        fair_model.positive_label = positive_label

    # ---------------------------------------------------------
    # Component 1 currently requires two protected attributes
    # ---------------------------------------------------------
    protected = list(protected)

    if len(protected) != 2:
        return {
            "status": "skipped",
            "component": "FairLogue Component 1",
            "reason": (
                "FairLogue Component 1 currently expects exactly "
                "two protected characteristics. "
                f"Received {len(protected)}: {protected}."
            ),
        }

    protected_1, protected_2 = protected

    # ---------------------------------------------------------
    # Recover the exact FairSelect held-out test observations
    # ---------------------------------------------------------
    test_index = getattr(rr, "test_index", None)

    if test_index is None:
        return {
            "status": "skipped",
            "component": "FairLogue Component 1",
            "reason": (
                "RunResult does not contain test_index. Component 1 "
                "was not run because the FairSelect held-out test "
                "observations cannot be identified safely."
            ),
        }

    test_index = list(test_index)

    missing_test_indices = [
        index
        for index in test_index
        if index not in df.index
    ]

    if missing_test_indices:
        return {
            "status": "failed",
            "component": "FairLogue Component 1",
            "reason": (
                f"{len(missing_test_indices)} stored test indices "
                "were not found in the FairSelect DataFrame."
            ),
            "missing_test_indices": missing_test_indices[:20],
        }

    # Preserve the original row indices. FairLogue operates on copies.
    test_df = df.loc[test_index].copy()

    train_index = df.index.difference(
        pd.Index(test_index),
        sort=False,
    )

    train_df = df.loc[train_index].copy()

    if test_df.empty:
        return {
            "status": "failed",
            "component": "FairLogue Component 1",
            "reason": "The reconstructed FairSelect test set is empty.",
        }

    if train_df.empty:
        return {
            "status": "failed",
            "component": "FairLogue Component 1",
            "reason": (
                "The reconstructed non-test dataset is empty. "
                "FairLogue requires both train_df and test_df when "
                "a caller-provided split is used."
            ),
        }

    # ---------------------------------------------------------
    # Validate required columns
    # ---------------------------------------------------------
    model_features = list(
        getattr(
            fair_model,
            "features",
            features,
        )
    )

    required_columns = list(
        dict.fromkeys([target, protected_1, protected_2, *model_features])
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        return {
            "status": "failed",
            "component": "FairLogue Component 1",
            "reason": (
                "The FairLogue audit data are missing columns "
                f"required by the FairModel: {missing_columns}"
            ),
        }

    # FairLogue uses df for feature validation, group summaries,
    # and optional cohort filtering. Because FairSelect already
    # filtered df, pass the same filtered cohort here.
    fairlogue_df = df[
        required_columns
    ].copy()

    fairlogue_train_df = train_df[
        required_columns
    ].copy()

    fairlogue_test_df = test_df[
        required_columns
    ].copy()

    try:
        # -----------------------------------------------------
        # Delegate all metric computation to FairLogue
        # -----------------------------------------------------
        results, figures, intermediates = (
            evaluate_intersectional_fairness(
                df=fairlogue_df,
                outcome=target,
                protected_1=protected_1,
                protected_2=protected_2,
                features=model_features,

                # This is informational when fair_model is supplied.
                model_type=getattr(
                    fair_model,
                    "model_type",
                    "fairselect",
                ),
                model_params=None,

                # Critical integration point:
                # FairLogue uses this fitted model and does not refit.
                fair_model=fair_model,

                positive_label=positive_label,

                # None tells FairLogue to use fair_model.threshold.
                threshold=None,

                make_plots=make_plots,

                # Use FairSelect's exact held-out split.
                train_df=fairlogue_train_df,
                test_df=fairlogue_test_df,

                # Return predictions, model metrics, and
                # nonintersectional results.
                return_intermediates=True,
                return_non_intersectional=(
                    return_non_intersectional
                ),

                # FairSelect already filtered the full cohort.
                min_group_size=min_group_size,
                require_class_balance=(
                    require_class_balance
                ),
            )
        )

        # -----------------------------------------------------
        # Retain the actual FairLogue objects and expose common
        # tables explicitly for the result-saving layer
        # -----------------------------------------------------
        non_intersectional = intermediates.get(
            "non_intersectional"
        )

        component1_output = {
            "status": "ok",
            "component": "FairLogue Component 1",
            "audit_source": (
                "FairSelect fitted FairModel evaluated through "
                "FairLogue.evaluate_intersectional_fairness"
            ),
            "model_name": getattr(
                fair_model,
                "name",
                rr.name,
            ),
            "model_type": getattr(
                fair_model,
                "model_type",
                None,
            ),
            "outcome_col": target,
            "positive_label": positive_label,
            "protected_1": protected_1,
            "protected_2": protected_2,
            "n_train": int(len(fairlogue_train_df)),
            "n_test": int(len(fairlogue_test_df)),
            "test_index": test_index,

            # Preserve the full FairLogue return values.
            "results": results,
            "figures": figures,
            "intermediates": intermediates,

            # Expose commonly saved fields directly.
            "per_group_df": results.per_group_df,
            "groups": results.groups,
            "demographic_parity_gap": (
                results.demographic_parity_gap
            ),
            "equalized_odds_gap_tpr": (
                results.equalized_odds_gap_tpr
            ),
            "equalized_odds_gap_fpr": (
                results.equalized_odds_gap_fpr
            ),
            "equal_opportunity_gap": (
                results.equal_opportunity_gap
            ),
            "dropped_groups": results.dropped_groups,
            "kept_groups_summary": (
                results.kept_groups_summary
            ),

            # FairLogue's race-only/gender-only equivalents.
            "non_intersectional": non_intersectional,

            # FairLogue-generated held-out model outputs.
            "y_test": intermediates.get("y_test"),
            "y_hat": intermediates.get("y_hat"),
            "proba": intermediates.get("proba"),
            "groups_test": intermediates.get(
                "groups_test"
            ),
            "protected_1_test": intermediates.get(
                "protected_1_test"
            ),
            "protected_2_test": intermediates.get(
                "protected_2_test"
            ),
            "model_metrics": intermediates.get(
                "model_metrics"
            ),
        }

        return component1_output

    except Exception as exc:
        return {
            "status": "failed",
            "component": "FairLogue Component 1",
            "model_name": getattr(
                fair_model,
                "name",
                rr.name,
            ),
            "outcome_col": target,
            "protected_1": protected_1,
            "protected_2": protected_2,
            "n_train": int(len(fairlogue_train_df)),
            "n_test": int(len(fairlogue_test_df)),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def _run_fairlogue_component3_for_result(
    *,
    rr: RunResult,
    df: pd.DataFrame,
    target: str,
    protected: Sequence[str],
    features: Sequence[str],
    method: str = "sr",
    n_splits: int = 5,
    gen_null: bool = False,
    R_null: int = 100,
    bootstrap: str = "none",
    B: int = 100,
    m_factor: float = 0.75,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run FairLogue Component 3 using the fitted FairModel attached
    to a FairSelect RunResult.

    The audit is restricted to the exact FairSelect held-out test
    observations. The FairModel is already fitted and is not retrained.

    Parameters
    ----------
    rr
        FairSelect RunResult containing the fitted FairModel and exact
        held-out test indices.

    df
        Filtered FairSelect cohort used to construct the split.

    target
        Binary outcome column.

    protected
        Protected-characteristic columns used by the FairModel.

    features
        Model covariates.

    method
        Component 3 estimator. Expected values are typically "sr"
        or "dr".

    n_splits
        Component 3 cross-fitting setting. This may not be used by the
        fixed-FairModel inference path but is retained in the Model
        configuration.

    gen_null
        Whether to generate a permutation null distribution.

    R_null
        Number of null permutations.

    bootstrap
        Bootstrap mode, such as "none" or "rescaled".

    B
        Number of bootstrap draws.

    m_factor
        Rescaled-bootstrap exponent used to determine the bootstrap
        sample size.

    random_state
        Reproducibility seed.
    """

    fair_model = getattr(
        rr,
        "fair_model",
        None,
    )

    refit_spec = getattr(
        rr,
        "refit_spec",
        None,
    )

    if refit_spec is None:
        refit_spec = getattr(
            fair_model,
            "refit_spec",
            None,
        )

    if gen_null and refit_spec is None:
        return {
            "status": "failed",
            "component": "FairLogue Component 3",
            "model_name": getattr(
                fair_model,
                "name",
                rr.name,
            ),
            "reason": (
                "A refitted Component 3 null was requested, but "
                "the FairModel has no FairSelectPlanSpec."
            ),
        }

    if fair_model is None:
        return {
            "status": "skipped",
            "component": "FairLogue Component 3",
            "reason": (
                "RunResult has no fair_model attached."
            ),
        }

    test_index = getattr(
        rr,
        "test_index",
        None,
    )

    if test_index is None:
        return {
            "status": "skipped",
            "component": "FairLogue Component 3",
            "model_name": getattr(
                fair_model,
                "name",
                rr.name,
            ),
            "reason": (
                "RunResult has no test_index. Component 3 was not "
                "run to avoid auditing training observations."
            ),
        }

    test_index = list(test_index)

    missing_test_indices = [
        index
        for index in test_index
        if index not in df.index
    ]

    if missing_test_indices:
        return {
            "status": "failed",
            "component": "FairLogue Component 3",
            "model_name": getattr(
                fair_model,
                "name",
                rr.name,
            ),
            "reason": (
                f"{len(missing_test_indices)} stored test indices "
                "were not found in the FairSelect DataFrame."
            ),
            "missing_test_indices": (
                missing_test_indices[:20]
            ),
        }

    protected = list(protected)
    features = list(features)

    if len(protected) < 2:
        return {
            "status": "skipped",
            "component": "FairLogue Component 3",
            "model_name": getattr(
                fair_model,
                "name",
                rr.name,
            ),
            "reason": (
                "Component 3 currently expects at least two "
                "protected characteristics."
            ),
        }

    audit_df = df.loc[
        test_index
    ].copy()

    if audit_df.empty:
        return {
            "status": "failed",
            "component": "FairLogue Component 3",
            "model_name": getattr(
                fair_model,
                "name",
                rr.name,
            ),
            "reason": (
                "The reconstructed FairSelect test set is empty."
            ),
        }

    valid_methods = {
        "sr",
        "dr",
    }

    if method not in valid_methods:
        return {
            "status": "failed",
            "component": "FairLogue Component 3",
            "model_name": getattr(
                fair_model,
                "name",
                rr.name,
            ),
            "reason": (
                f"Unsupported Component 3 method {method!r}. "
                f"Expected one of {sorted(valid_methods)}."
            ),
        }

    valid_bootstrap_modes = {
        "none",
        "rescaled",
    }

    if bootstrap not in valid_bootstrap_modes:
        return {
            "status": "failed",
            "component": "FairLogue Component 3",
            "model_name": getattr(
                fair_model,
                "name",
                rr.name,
            ),
            "reason": (
                f"Unsupported bootstrap mode {bootstrap!r}. "
                f"Expected one of "
                f"{sorted(valid_bootstrap_modes)}."
            ),
        }

    if gen_null and int(R_null) < 1:
        return {
            "status": "failed",
            "component": "FairLogue Component 3",
            "model_name": getattr(
                fair_model,
                "name",
                rr.name,
            ),
            "reason": (
                "R_null must be at least 1 when gen_null=True."
            ),
        }

    if bootstrap != "none" and int(B) < 1:
        return {
            "status": "failed",
            "component": "FairLogue Component 3",
            "model_name": getattr(
                fair_model,
                "name",
                rr.name,
            ),
            "reason": (
                "B must be at least 1 when bootstrap is enabled."
            ),
        }

    if not 0 < float(m_factor) <= 1:
        return {
            "status": "failed",
            "component": "FairLogue Component 3",
            "model_name": getattr(
                fair_model,
                "name",
                rr.name,
            ),
            "reason": (
                "m_factor must be greater than 0 and less than "
                f"or equal to 1. Received {m_factor}."
            ),
        }

    try:
        try:
            from FairLogue.Component3.model import (
                Model as Component3Model,
            )

        except ImportError:
            from combined_toolkits.FairLogue.Component3.model import (
                Model as Component3Model,
            )

        model_features = list(getattr(fair_model, "features", features))
        component3_covariates = [column for column in model_features if column not in protected]

        component3_model = Component3Model(
            data=audit_df,
            outcome=target,
            protected_characteristics=tuple(
                protected[:2]
            ),
            covariates=component3_covariates,
            fair_model=fair_model,
            method=method,
            n_splits=n_splits,
            random_state=random_state,
        )

        component3_model.pre_process_data()

        def null_refit_fn(
            permuted_df: pd.DataFrame,
            null_random_state: int,
        ) -> FairModel:
            return refit_fairmodel_from_spec(
                df=permuted_df,
                spec=refit_spec,
                random_state=null_random_state,
            )

        component3_results = (
            component3_model.fit_fairness_from_fairmodel(
                cutoff=getattr(
                    fair_model,
                    "threshold",
                    0.5,
                ),
                gen_null=bool(gen_null),
                R_null=int(R_null),
                bootstrap=bootstrap,
                B=int(B),
                m_factor=float(m_factor),
                null_source_data=(
                    df
                    if gen_null
                    else None
                ),
                null_audit_index=(
                    test_index
                    if gen_null
                    else None
                ),
                null_refit_fn=(
                    null_refit_fn
                    if gen_null
                    else None
                ),
            )
        )

        summary = component3_model.summarize()

        # Determine what actually ran from the returned result rather
        # than assuming that requested inference completed.
        null_applied = bool(
            component3_results.get(
                "gen_null_applied",
                "table_null" in component3_results,
            )
        )

        bootstrap_applied = bool(
            component3_results.get(
                "bootstrap_applied",
                "boot_out" in component3_results,
            )
        )

        return {
            "status": "ok",
            "component": "FairLogue Component 3",
            "audit_source": (
                "FairSelect fitted FairModel evaluated on the "
                "held-out FairSelect test set"
            ),
            "model_name": getattr(
                fair_model,
                "name",
                rr.name,
            ),
            "method": method,
            "n_test": int(
                len(audit_df)
            ),
            "test_index": test_index,

            # Requested null settings
            "gen_null_requested": bool(
                gen_null
            ),
            "R_null_requested": int(
                R_null
            ),

            # Actual null status
            "gen_null_applied": (
                null_applied
            ),
            "R_null_completed": (
                component3_results.get(
                    "R_null_completed",
                    (
                        int(R_null)
                        if null_applied
                        else 0
                    ),
                )
            ),

            # Requested bootstrap settings
            "bootstrap_requested": (
                bootstrap
            ),
            "B_requested": int(B),
            "m_factor_requested": float(
                m_factor
            ),

            # Actual bootstrap status
            "bootstrap_applied": (
                bootstrap_applied
            ),
            "B_completed": (
                component3_results.get(
                    "B_completed",
                    (
                        len(
                            component3_results.get(
                                "boot_out",
                                [],
                            )
                        )
                        if bootstrap_applied
                        else 0
                    ),
                )
            ),

            # Preserve complete Component 3 outputs
            "results": component3_results,
            "summary": summary,
        }

    except Exception as exc:
        return {
            "status": "failed",
            "component": "FairLogue Component 3",
            "model_name": getattr(
                fair_model,
                "name",
                rr.name,
            ),
            "method": method,
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
            "m_factor_requested": float(
                m_factor
            ),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
