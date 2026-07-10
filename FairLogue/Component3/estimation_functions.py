from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import norm
from .helpers import _as_str_groups, ProbaEstimator
from .outcome_models import build_outcome_models_and_scores_fixed_split, build_outcome_models_and_scores, get_defs_from_rates, compute_cf_group_rates_dr, compute_cf_group_rates_sr


#-------Estimation wrappers -----------
def get_defs_analysis(
    data_with_mu: pd.DataFrame,
    group_col: str,
    outcome_col: str,
    tau: float,
    method: str = "sr",   #'sr' (single robust) or 'dr' (doubly robust)
    groups_universe: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute fairness definitions (defs) from muY_* and τ.

    Uses SR or DR (AIPW) group-rate estimators and returns
    aggregate and per-group stats. See docs: docs/estimation_functions.md#get_defs_analysis
    """
    if method == "sr":
        rates = compute_cf_group_rates_sr(
            data_with_mu, group_col, outcome_col, tau,
            groups_universe=groups_universe
        )
    elif method == "dr":
        rates = compute_cf_group_rates_dr(
            data_with_mu, group_col, outcome_col, tau,
            groups_universe=groups_universe
        )
    else:
        raise ValueError("method must be 'sr' or 'dr'")
    return get_defs_from_rates(rates)


def analysis_nulldist(
    data: pd.DataFrame,
    group_col: str,
    outcome_col: str,
    covariates: List[str],
    tau_fixed: float,
    R: int = 200,
    model: Optional[ProbaEstimator] = None,
    model_type: str = "rf",
    n_splits: int = 5,
    random_state: int = 42,
    method: str = "sr",
    groups_universe: Optional[List[str]] = None,
) -> Dict[str, List[float]]:
    """
    Permutation null: shuffle group labels, refit, recompute defs.

    Returns a dict of arrays (one key per stat) representing the null draws,
    holding τ fixed at tau_fixed. See docs: docs/estimation_functions.md#analysis_nulldist
    """
    rng = np.random.default_rng(random_state)
    out_rows: List[Dict[str, float]] = []

    for r in range(R):
        df_perm = data.copy()
        df_perm[group_col] = rng.permutation(df_perm[group_col].values)
        df_mu_perm, _, _ = build_outcome_models_and_scores(
            df_perm, group_col, outcome_col, covariates,
            model=model, model_type=model_type, n_splits=n_splits,
            random_state=random_state + r, groups_universe=groups_universe,
        )
        defs_r = get_defs_analysis(
            df_mu_perm, group_col, outcome_col, tau_fixed,
            method=method, groups_universe=groups_universe
        )
        out_rows.append(defs_r)

    keys = sorted(set().union(*[row.keys() for row in out_rows]))
    table_null: Dict[str, List[float]] = {k: [row.get(k, np.nan) for row in out_rows] for k in keys}
    return table_null


def bs_rescaled_analysis(
    data: pd.DataFrame,
    group_col: str,
    outcome_col: str,
    covariates: List[str],
    B: int = 5, #default should be 500, changing for testing
    m_factor: float = 0.75,
    model: Optional[ProbaEstimator] = None,
    model_type: str = "rf",
    n_splits: int = 5,
    random_state: int = 42,
    method: str = "sr",
    groups_universe: Optional[List[str]] = None,
) -> List[Dict[str, float]]:
    """
    Rescaled bootstrap: m = floor(n**m_factor) resamples, refit, recompute defs.

    Returns a list of dicts (one per bootstrap draw) in the same key schema as defs.
    See docs: docs/estimation_functions.md#bs_rescaled_analysis
    """
    rng = np.random.default_rng(random_state)
    n = len(data)
    m = int(np.floor(n ** m_factor))
    out: List[Dict[str, float]] = []

    for b in range(B):
        idx = rng.choice(n, size=m, replace=True)
        df_boot = data.iloc[idx].reset_index(drop=True)
        df_mu, tau_boot, _ = build_outcome_models_and_scores(
            df_boot, group_col, outcome_col, covariates,
            model=model, model_type=model_type, n_splits=n_splits,
            random_state=random_state + b, groups_universe=groups_universe,
        )
        defs_b = get_defs_analysis(
            df_mu, group_col, outcome_col, tau_boot,
            method=method, groups_universe=groups_universe
        )
        out.append(defs_b)

    return out


def analysis_estimation(
    data: pd.DataFrame,
    group_col: str,
    outcome_col: str,
    covariates: List[str],
    cutoff: Optional[float] = None,       #if None, choose Youden on OOF factual
    model: Optional[ProbaEstimator] = None,
    model_type: str = "rf",
    n_splits: int = 5,
    random_state: int = 42,
    gen_null: bool = True,
    train_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    R_null: int = 200,
    bootstrap: str = "none",             #{'none','rescaled'}
    B: int = 5, #default should be 500, changing for testing
    m_factor: float = 0.75,
    method: str = "sr",              #{'sr','dr'}
) -> Dict[str, object]:
    """
    Observed fit → muY_* and τ → defs; optional permutation null and bootstrap.

    Primary entry point used by the facade. Returns a results dict with defs,
    est_choice, tau, groups, and optionally table_null and boot_out.
    See docs: docs/estimation_functions.md#analysis_estimation
    """
    groups_universe = sorted(_as_str_groups(data[group_col]).unique().tolist())

    if (train_df is None) ^ (test_df is None):
        raise ValueError("Provide both train_df and test_df, or neither.")

    if train_df is not None:
        #groups universe should cover both
        groups_universe = sorted(
            _as_str_groups(pd.concat([train_df[group_col], test_df[group_col]])).unique().tolist()
        )
        df_mu, tau_obs, groups = build_outcome_models_and_scores_fixed_split(
            train_df=train_df,
            test_df=test_df,
            group_col=group_col,
            outcome_col=outcome_col,
            covariates=covariates,
            model=model,
            model_type=model_type,
            random_state=random_state,
            groups_universe=groups_universe,
        )
    else:
        groups_universe = sorted(_as_str_groups(data[group_col]).unique().tolist())
        df_mu, tau_obs, groups = build_outcome_models_and_scores(
            data, group_col, outcome_col, covariates,
            model=model, model_type=model_type,
            n_splits=n_splits, random_state=random_state,
            groups_universe=groups_universe,
        )
    tau = float(cutoff) if cutoff is not None else tau_obs

    #Compute metrics on observed data
    defs = get_defs_analysis(
        df_mu, group_col, outcome_col, tau,
        method=method, groups_universe=groups_universe
    )

    results: Dict[str, object] = {"defs": defs, "est_choice": df_mu.copy(), "tau": tau, "groups": groups}

    #Generate null distribution if requested
    if gen_null:
        table_null = analysis_nulldist(
            data, group_col, outcome_col, covariates, tau_fixed=tau,
            R=R_null, model=model, model_type=model_type,
            n_splits=n_splits, random_state=random_state + 13,
            method=method, groups_universe=groups_universe,
        )
        results["table_null"] = table_null

    #Generate bootstrap draws if requested
    if bootstrap == "rescaled":
        boot_out = bs_rescaled_analysis(
            data, group_col, outcome_col, covariates,
            B=B, m_factor=m_factor, model=model, model_type=model_type,
            n_splits=n_splits, random_state=random_state + 29,
            method=method, groups_universe=groups_universe,
        )
        results["boot_out"] = boot_out

    return results

#----- Bootstrap utilities + CIs ------

def get_bs_rescaled(bs_table: List[Dict[str, float]], est_vals: Dict[str, float],) -> pd.DataFrame:
    """
    Form the rescaled bootstrap matrix for CI construction.

    Aligns keys to est_vals, fills missing with NaN, and rescales by sqrt(m).
    See docs: docs/estimation_functions.md#get_bs_rescaled
    """
    if not isinstance(bs_table, list) or not bs_table:
        raise ValueError("boot_out must be a non-empty list of dicts.")

    #Extract keys and form matrix of bootstrap draws
    keys = list(est_vals.keys())
    rows = [[d.get(k, np.nan) for k in keys] for d in bs_table]
    bs_matrix = np.array(rows, dtype=float)

    #Compute scaling factor
    m = bs_matrix.shape[0]
    sqrt_m = np.sqrt(m)

    #Center and rescale
    est_vals_array = np.array([est_vals.get(k, np.nan) for k in keys], dtype=float)
    rescaled_bs_table = sqrt_m * (bs_matrix - est_vals_array)
    return pd.DataFrame(rescaled_bs_table, columns=keys)


def ci_norm(bs_rescaled: pd.DataFrame, est_named: Dict[str, float], parameter: str, sampsize: int, alpha: float) -> pd.DataFrame:
    """
    Normal-approximation CI from rescaled bootstrap draws.

    Returns var, point_est, se_est, ci_low, ci_high for a single parameter.
    See docs: docs/estimation_functions.md#ci_norm
    """
    
    var_est = np.nanvar(bs_rescaled[parameter], ddof=1) #Variance from rescaled bootstrap draws
    point_est = est_named[parameter] #Original estimate
    se_est = np.sqrt(var_est / sampsize) #Standard error 
    z = norm.ppf(1 - alpha / 2) #Z-score for two-sided CI
    ci_low = point_est - z * se_est #lower bound CI
    ci_high = point_est + z * se_est #upper bound CI

    return pd.DataFrame({
        'var_est': [var_est],
        'point_est': [point_est],
        'se_est': [se_est],
        'ci_low': [ci_low],
        'ci_high': [ci_high]
    })


def ci_tint(bs_rescaled: pd.DataFrame, est_named: Dict[str, float], parameter: str, sampsize: int, alpha: float, m_factor: float) -> pd.DataFrame:
    """
    Transformed interval (t-int) CI using rescaled bootstrap t-values.

    Stable for skewed distributions; returns point_est, se_est, low_trans, high_trans.
    See docs: docs/estimation_functions.md#ci_tint
    """
    se_table = ci_norm(bs_rescaled, est_named, parameter, sampsize, alpha)
    se_scalar = float(se_table['se_est'].values[0])
    denom = (sampsize ** m_factor) ** 0.5 
    t_values = (bs_rescaled[parameter] / (denom * se_scalar)) if se_scalar > 0 else bs_rescaled[parameter] * np.nan 
    low_q = np.nanquantile(t_values, 1 - alpha / 2) 
    high_q = np.nanquantile(t_values, alpha / 2) 
    point_est = float(se_table['point_est'].values[0])
    low_trans = point_est - se_scalar * low_q
    high_trans = point_est - se_scalar * high_q

    return pd.DataFrame({
        'point_est': [point_est],
        'se_est': [se_scalar],
        'low_trans': [low_trans],
        'high_trans': [high_trans]
    })


def ci_trunc(ci_result: pd.DataFrame, type2: str) -> pd.DataFrame:
    """
    Clip CI bounds to [0,1] for probability-like parameters.

    Supports 'norm' (ci_low/ci_high) and 'tint' (low_trans/high_trans).
    See docs: docs/estimation_functions.md#ci_trunc
    """
    res = ci_result.copy()
    if type2 == 'norm':
        res['ci_low'] = res['ci_low'].clip(lower=0)
        res['ci_high'] = res['ci_high'].clip(upper=1)
    elif type2 == 'tint':
        res['low_trans'] = res['low_trans'].clip(lower=0)
        res['high_trans'] = res['high_trans'].clip(upper=1)
    return res