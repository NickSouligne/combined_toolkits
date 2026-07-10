import numpy as np
import pandas as pd
import FairLogue as ift

from FairLogue.Component1 import _compute_group_rates
from FairLogue.Component3 import FairnessPipeline

from __future__ import annotations

from typing import Sequence, Dict, Any




def run_fairlogue_observational(
    *,
    y_true,
    y_prob,
    y_pred,
    groups,
    run_name: str,
) -> Dict[str, Any]:
    """
    Run Fairlogue Component 1-style observational fairness auditing
    using predictions already produced by the first toolkit.
    """

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    groups = pd.Series(groups).astype(str).reset_index(drop=True)

    group_rates = _compute_group_rates(
        y_true=y_true,
        y_pred=y_pred,
        groups=groups,
    )

    per_group_df = pd.DataFrame([gr.__dict__ for gr in group_rates])

    def _gap(s: pd.Series) -> float:
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        return float(s.max() - s.min()) if not s.empty else float("nan")

    return {
        "component": "component1_observational",
        "run_name": run_name,
        "demographic_parity_gap": _gap(per_group_df["positive_rate"]),
        "equalized_odds_gap_tpr": _gap(per_group_df["tpr"]),
        "equalized_odds_gap_fpr": _gap(per_group_df["fpr"]),
        "equal_opportunity_gap": _gap(per_group_df["tpr"]),
        "per_group": per_group_df,
    }




def run_fairlogue_counterfactual_component3(
    *,
    df: pd.DataFrame,
    outcome_col: str,
    protected_cols: Sequence[str],
    covariates: Sequence[str],
    cutoff: float | None = None,
    method: str = "sr",
    gen_null: bool = False,
    R_null: int = 100,
    bootstrap: str = "none",
    B: int = 100,
    random_state: int = 42,
):
    """
    Run Fairlogue Component 3 counterfactual fairness audit.
    """

    if len(protected_cols) < 2:
        raise ValueError("Fairlogue Component 3 currently expects two protected characteristics.")

    work = df.copy()
    group_col = "_fairlogue_group"
    work[group_col] = (
        work[protected_cols[0]].astype(str)
        + "|"
        + work[protected_cols[1]].astype(str)
    )

    pipe = FairnessPipeline(
        group_col=group_col,
        outcome_col=outcome_col,
        covariates=list(covariates),
        model_type="rf",
        n_splits=5,
        random_state=random_state,
        method=method,
    )

    results = pipe.fit(
        data=work,
        cutoff=cutoff,
        gen_null=gen_null,
        R_null=R_null,
        bootstrap=bootstrap,
        B=B,
    )

    summary = pipe.summarize()

    return {
        "component": "component3_counterfactual",
        "results": results,
        "summary": summary,
    }