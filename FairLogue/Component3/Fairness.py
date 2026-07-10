from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
from .plots import get_plots
from .estimation_functions import analysis_estimation
from .helpers import ProbaEstimator



#----- High-level wrapper to run the full pipeline -----

class FairnessPipeline:
    """
    Thin orchestrator around the functional API. Keeps state in `results_`.
    """

    def __init__(
        self,
        group_col: str,
        outcome_col: str,
        covariates: List[str],
        estimator: Optional[ProbaEstimator] = None,
        model_type: str = "rf",
        n_splits: int = 5,
        random_state: int = 42,
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        method: str = "sr"  #sr (single robust) or 'dr' (doubly robust)
    ):
        self.group_col = group_col
        self.outcome_col = outcome_col
        self.covariates = covariates
        self.estimator = estimator
        self.model_type = model_type
        self.n_splits = n_splits
        self.random_state = random_state
        self.method = method
        self.results_: Optional[Dict[str, object]] = None

    def fit(
        self,
        data: pd.DataFrame,
        cutoff: Optional[float] = None,
        gen_null: bool = True,
        R_null: int = 5, #default should be 200, changing for testing
        bootstrap: str = "none",
        B: int = 500,
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        m_factor: float = 0.75,
    ) -> Dict[str, object]:
        
        self.results_ = analysis_estimation(
            data=data,
            group_col=self.group_col,
            outcome_col=self.outcome_col,
            covariates=self.covariates,
            cutoff=cutoff,
            model=self.estimator,
            model_type=self.model_type,
            n_splits=self.n_splits,
            random_state=self.random_state,
            gen_null=gen_null,
            R_null=R_null,
            bootstrap=bootstrap,
            B=B,
            train_df=train_df,
            test_df=test_df,
            m_factor=m_factor,
            method=self.method,
        )
        return self.results_

    def summarize(self) -> pd.DataFrame:
        if not self.results_:
            raise RuntimeError("Call fit() before summarize().")
        return pd.DataFrame(list(self.results_["defs"].items()), columns=["stat", "value"])

    def plots(self, alpha: float = 0.05, m_factor: float = 0.75, delta_uval: float = 0.10,):
        if not self.results_:
            raise RuntimeError("Call fit() before plots().")
        est, delta, uvals = get_plots(
            results=self.results_,
            sampsize=len(self.results_.get('est_choice', [])) if self.results_ else None,
            alpha=alpha, m_factor=m_factor, delta_uval=delta_uval
        )
        return est, delta, uvals