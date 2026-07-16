from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from .helpers import _as_str_groups, _clip_probs, choose_threshold_youden, _add_group_dummies, ProbaEstimator, make_outcome_estimator



#-------Cross-fitting & muY outputs ----------

def build_outcome_models_and_scores_fixed_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: str,
    outcome_col: str,
    covariates: list[str],
    model=None,
    model_type: str = "rf",
    random_state: int = 42,
    groups_universe: list[str] | None = None,
):
    """
    Fit outcome model on train_df only; compute counterfactual muY_<g> on test_df only.
    Returns (df_test_with_mu, tau, groups).
    """
    tr = train_df.copy()
    te = test_df.copy()

    y_tr = tr[outcome_col].astype(int).to_numpy()
    X_tr = tr[covariates].to_numpy(dtype=float, copy=False)
    A_tr = _as_str_groups(tr[group_col]).to_numpy()

    X_te = te[covariates].to_numpy(dtype=float, copy=False)
    A_te = _as_str_groups(te[group_col]).to_numpy()

    groups = sorted(groups_universe or np.unique(np.concatenate([A_tr, A_te])).tolist())
    K = len(groups)

    g2i = {g: i for i, g in enumerate(groups)}
    A_tr_idx = np.fromiter((g2i[a] for a in A_tr), dtype=np.int64, count=len(A_tr))
    A_te_idx = np.fromiter((g2i[a] for a in A_te), dtype=np.int64, count=len(A_te))

    def _make():
        if model is None:
            return make_outcome_estimator(model_type, random_state=random_state)
        try:
            return clone(model)
        except Exception:
            return model

    # augment TRAIN: [X, one-hot(A)]
    G_tr = np.zeros((X_tr.shape[0], K), dtype=np.uint8)
    G_tr[np.arange(X_tr.shape[0]), A_tr_idx] = 1
    X_tr_aug = np.concatenate([X_tr, G_tr], axis=1)

    clf = _make()
    clf.fit(X_tr_aug, y_tr)

    # counterfactual mu on TEST for all groups
    n_te = X_te.shape[0]
    mu = np.empty((n_te, K), dtype=np.float32)
    mu.fill(np.nan)

    # simple loop over groups (fine unless K huge)
    for j, g in enumerate(groups):
        G_te = np.zeros((n_te, K), dtype=np.uint8)
        G_te[:, j] = 1
        X_te_aug = np.concatenate([X_te, G_te], axis=1)
        mu[:, j] = clf.predict_proba(X_te_aug)[:, 1].astype(np.float32, copy=False)

    # factual preds on TRAIN for tau selection
    # (train factual pred = mu at factual A; easiest is predict factual directly)
    # build factual augmented X for train
    # (already have X_tr_aug; predict on that)
    p_tr_fact = clf.predict_proba(X_tr_aug)[:, 1]
    tau = choose_threshold_youden(y_tr, p_tr_fact)

    mu_cols = [f"muY_{g}" for g in groups]
    te[mu_cols] = mu
    return te, float(tau), groups

def build_outcome_models_and_scores(
    data: pd.DataFrame,
    group_col: str,             # e.g., 'A1A2' (string codes)
    outcome_col: str,           # e.g., 'Y' (binary 0/1)
    covariates: List[str],
    model: Optional[ProbaEstimator] = None,
    model_type: str = "rf",
    n_splits: int = 5,
    random_state: int = 42,
    groups_universe: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, float, List[str]]:
    """
    Drop-in replacement that keeps the SAME external behavior and return types as the
    original function, but removes pandas usage from the hot loops.

    What stays the same downstream:
      - returns (df_with_mu_columns, tau, groups)
      - df_with_mu_columns is a pandas DataFrame, with columns muY_<g> for all groups
      - tau computed from factual OOF preds via Youden index
      - groups ordering stable sorted

    What changes internally (performance):
      - all CV work uses numpy arrays (no .iloc/.loc/.concat in the loop)
      - counterfactual prediction is done in GROUP BLOCKS to avoid huge stacked matrices
        while still reducing predict_proba calls vs pure per-group loop.
    """
    df = data.copy()

    # --- Materialize arrays ONCE ---
    y = df[outcome_col].astype(int).to_numpy()
    X = df[covariates].to_numpy(dtype=float, copy=False)
    A = _as_str_groups(df[group_col]).to_numpy()

    # --- Stable group universe ---
    groups = sorted(groups_universe or np.unique(A).tolist())
    K = len(groups)
    N, P = X.shape

    # Map group label -> integer index 0..K-1 (vectorized)
    g2i = {g: i for i, g in enumerate(groups)}
    A_idx = np.fromiter((g2i[a] for a in A), dtype=np.int64, count=N)

    # --- Allocate outputs as numpy (no DataFrame writes in loop) ---
    mu = np.empty((N, K), dtype=np.float32)
    mu.fill(np.nan)
    factual_pred = np.empty(N, dtype=np.float64)

    # --- Model factory (clone where possible) ---
    def _make():
        if model is None:
            return make_outcome_estimator(model_type, random_state=random_state)
        try:
            return clone(model)
        except Exception:
            return model

    # --- Precompute splits once (still identical estimation logic) ---
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(skf.split(X, y))

    # Heuristic: choose group-block size to control memory.
    # You can tune this constant; 16 or 32 are usually safe.
    GROUP_BLOCK = 16

    for train_idx, test_idx in splits:
        X_tr = X[train_idx]
        y_tr = y[train_idx]
        A_tr_idx = A_idx[train_idx]

        X_te = X[test_idx]
        A_te_idx = A_idx[test_idx]
        n_te = X_te.shape[0]

        # ---- Augment TRAIN: [X, one-hot(A)] ----
        G_tr = np.zeros((X_tr.shape[0], K), dtype=np.uint8)
        G_tr[np.arange(X_tr.shape[0]), A_tr_idx] = 1
        X_tr_aug = np.concatenate([X_tr, G_tr], axis=1)

        clf = _make()
        clf.fit(X_tr_aug, y_tr)

        # ---- Counterfactual prediction in GROUP BLOCKS (batched predict_proba) ----
        # Fill mu[test_idx, :] block by block.
        # For each block of groups [b0:b1):
        #   Build stacked matrix with that many group identities
        #   Call predict_proba once
        #   Reshape back into (n_te, block_size)
        for b0 in range(0, K, GROUP_BLOCK):
            b1 = min(K, b0 + GROUP_BLOCK)
            B = b1 - b0

            # Stack X_te B times
            X_te_stack = np.repeat(X_te, repeats=B, axis=0)  # (B*n_te, P)

            # Build dummy block (B*n_te, K) but only set one column per row
            # To reduce overhead, we still allocate full K here; if K is huge,
            # we can optimize further (ask and I’ll give that version).
            G_te = np.zeros((B * n_te, K), dtype=np.uint8)

            # Row r in stacked corresponds to: group = b0 + (r // n_te)
            # Within each group block, set the right dummy to 1.
            block_group_ids = (np.arange(B, dtype=np.int64) + b0)
            row_groups = np.repeat(block_group_ids, repeats=n_te)  # (B*n_te,)
            G_te[np.arange(B * n_te), row_groups] = 1

            X_te_aug = np.concatenate([X_te_stack, G_te], axis=1)  # (B*n_te, P+K)
            p = clf.predict_proba(X_te_aug)[:, 1]                  # (B*n_te,)

            # Reshape: first n_te are group b0, next n_te group b0+1, etc.
            p_mat = p.reshape(B, n_te).T  # (n_te, B)

            mu[np.asarray(test_idx), b0:b1] = p_mat.astype(np.float32, copy=False)

        # ---- factual OOF probability (for tau) ----
        factual_pred[np.asarray(test_idx)] = mu[np.asarray(test_idx), A_te_idx].astype(np.float64)

    # Choose tau via Youden on factual OOF preds
    tau = choose_threshold_youden(y, factual_pred)

    # ---- Attach mu columns to DataFrame ONCE (keeps downstream structure identical) ----
    mu_cols = [f"muY_{g}" for g in groups]
    df[mu_cols] = mu  # single assignment; avoids per-cell/.loc writes

    return df, float(tau), groups


@dataclass
class CfRates:
    """
    Container for groupwise rates.

    cfpr, cfnr: counterfactual FPR/FNR under do(A=g)
    fpr_obs, fnr_obs: observed rates under factual A.
    See docs/outcome_models.md#cfrates
    """
    cfpr: Dict[str, float]
    cfnr: Dict[str, float]
    fpr_obs: Dict[str, float]
    fnr_obs: Dict[str, float]


def _select_mu_fact(df: pd.DataFrame, A: pd.Series, groups: List[str], mu_prefix: str = "muY_") -> np.ndarray:
    """
    Extract factual predicted values from counterfactual mean outcome columns.

    Given a DataFrame containing counterfactual predictions (e.g., columns like 'muY_group'),
    this function selects, for each observation, the value corresponding to that individual's
    actual (factual) group membership.

    See docs/outcome_models.md#_select_mu_fact
    """
    #Build list of column names representing each groups predicted mu (outcome)
    mu_cols = [f"{mu_prefix}{g}" for g in groups]
    mu_mat = df[mu_cols].to_numpy()

    #Map group labels to column positions
    colpos = {g: j for j, g in enumerate(groups)}
    j = A.map(colpos).to_numpy()

    #Return factual prediction value
    return mu_mat[np.arange(len(df)), j]



#------- Group-wise rates (sr/DR) -----------
def compute_cf_group_rates_sr(
    data,
    group_col,
    outcome_col,
    tau,
    mu_prefix="muY_",
    score_prefix="S_",
    groups_universe=None,
    eps=1e-8,
):
    df = data.copy()

    A = _as_str_groups(
        df[group_col]
    )

    y = (
        df[outcome_col]
        .astype(int)
        .to_numpy()
    )

    groups = sorted(
        groups_universe
        or A.unique().tolist()
    )

    mu_fact = _select_mu_fact(
        df,
        A,
        groups,
        mu_prefix=mu_prefix,
    )

    # Use factual FairModel classifications when available.
    factual_scores = np.empty(
        len(df),
        dtype=int,
    )

    group_to_position = {
        group: position
        for position, group in enumerate(groups)
    }

    factual_group_positions = (
        A.map(group_to_position)
        .to_numpy()
    )

    score_columns = [
        f"{score_prefix}{group}"
        for group in groups
    ]

    has_score_columns = all(
        column in df.columns
        for column in score_columns
    )

    if has_score_columns:
        score_matrix = (
            df[score_columns]
            .to_numpy()
            .astype(int)
        )

        factual_scores = score_matrix[
            np.arange(len(df)),
            factual_group_positions,
        ]

    else:
        factual_scores = (
            mu_fact >= tau
        ).astype(int)

    cfpr = {}
    cfnr = {}
    fpr_obs = {}
    fnr_obs = {}

    for group in groups:
        mu_g = (
            df[f"{mu_prefix}{group}"]
            .to_numpy(dtype=float)
        )

        mu0_g = np.clip(
            1.0 - mu_g,
            eps,
            1.0,
        )

        mu1_g = np.clip(
            mu_g,
            eps,
            1.0,
        )

        score_column = (
            f"{score_prefix}{group}"
        )

        if score_column in df.columns:
            S_g = (
                df[score_column]
                .to_numpy()
                .astype(int)
            )
        else:
            S_g = (
                mu_g >= tau
            ).astype(int)

        denominator_cfpr = (
            mu0_g.sum()
        )

        denominator_cfnr = (
            mu1_g.sum()
        )

        cfpr[group] = (
            float(
                (S_g * mu0_g).sum()
                / denominator_cfpr
            )
            if (
                np.isfinite(
                    denominator_cfpr
                )
                and denominator_cfpr > 0
            )
            else np.nan
        )

        cfnr[group] = (
            float(
                (
                    (1 - S_g)
                    * mu1_g
                ).sum()
                / denominator_cfnr
            )
            if (
                np.isfinite(
                    denominator_cfnr
                )
                and denominator_cfnr > 0
            )
            else np.nan
        )

        factual_mask = (
            A == group
        ).to_numpy()

        y0 = y == 0
        y1 = y == 1

        denominator_fpr = int(
            (y0 & factual_mask).sum()
        )

        denominator_fnr = int(
            (y1 & factual_mask).sum()
        )

        fpr_obs[group] = (
            float(
                (
                    (factual_scores == 1)
                    & y0
                    & factual_mask
                ).sum()
                / denominator_fpr
            )
            if denominator_fpr > 0
            else np.nan
        )

        fnr_obs[group] = (
            float(
                (
                    (factual_scores == 0)
                    & y1
                    & factual_mask
                ).sum()
                / denominator_fnr
            )
            if denominator_fnr > 0
            else np.nan
        )

    return CfRates(
        cfpr=cfpr,
        cfnr=cfnr,
        fpr_obs=fpr_obs,
        fnr_obs=fnr_obs,
    )


def compute_cf_group_rates_dr(
    data: pd.DataFrame,
    group_col: str,
    outcome_col: str,
    tau: float,
    mu_prefix: str = "muY_",
    score_prefix: str = "S_",
    pi_prefix: str = "group_",
    groups_universe: Optional[List[str]] = None,
    eps: float = 1e-8,
) -> CfRates:
    """
    Doubly robust AIPW estimators for counterfactual FPR/FNR and
    observed factual FPR/FNR.

    When S_<group> columns are available, they are used as the
    counterfactual classification decisions. These columns should be
    generated from FairModel.predict() and therefore preserve
    group-specific thresholds and postprocessing.

    If S_<group> columns are absent, classifications are reconstructed
    from muY_<group> using the scalar threshold ``tau`` for backward
    compatibility.

    Required columns
    ----------------
    muY_<group>
        Counterfactual probability under intervention A=group.

    group_<group>_prob
        Estimated propensity P(A=group | X).

    Optional columns
    ----------------
    S_<group>
        Counterfactual hard prediction under intervention A=group.

    Parameters
    ----------
    data
        DataFrame containing outcomes, factual groups, counterfactual
        probabilities, optional counterfactual classifications, and
        propensity estimates.

    group_col
        Column containing factual intersectional group membership.

    outcome_col
        Binary observed outcome column.

    tau
        Fallback threshold used only when S_<group> columns are absent.

    mu_prefix
        Prefix for counterfactual probability columns.

    score_prefix
        Prefix for counterfactual classification columns.

    pi_prefix
        Prefix for propensity columns. A group ``g`` is expected to
        have a column named ``f"{pi_prefix}{g}_prob"``.

    groups_universe
        Explicit ordered group universe. When omitted, factual groups
        in ``group_col`` are used.

    eps
        Lower clipping bound for propensity scores and numerical
        denominators.
    """
    df = data.copy()

    A = _as_str_groups(
        df[group_col]
    )

    y = (
        df[outcome_col]
        .astype(int)
        .to_numpy()
    )

    groups = sorted(
        groups_universe
        or A.unique().tolist()
    )

    if not groups:
        raise ValueError(
            "No protected groups were available for DR estimation."
        )

    # ---------------------------------------------------------
    # Validate counterfactual probability columns
    # ---------------------------------------------------------
    missing_mu_columns = [
        f"{mu_prefix}{group}"
        for group in groups
        if f"{mu_prefix}{group}" not in df.columns
    ]

    if missing_mu_columns:
        raise ValueError(
            "DR estimation is missing counterfactual probability "
            f"columns: {missing_mu_columns}"
        )

    # ---------------------------------------------------------
    # Select factual probabilities
    # ---------------------------------------------------------
    mu_fact = _select_mu_fact(
        df,
        A,
        groups,
        mu_prefix=mu_prefix,
    )

    # ---------------------------------------------------------
    # Select factual hard predictions
    # ---------------------------------------------------------
    #
    # Each row's factual prediction is the S_<group> value
    # corresponding to that row's observed factual group.
    score_columns = [
        f"{score_prefix}{group}"
        for group in groups
    ]

    has_all_score_columns = all(
        column in df.columns
        for column in score_columns
    )

    if has_all_score_columns:
        score_matrix = (
            df[score_columns]
            .to_numpy()
            .astype(int)
        )

        group_to_position = {
            group: position
            for position, group in enumerate(groups)
        }

        factual_positions = (
            A.map(group_to_position)
            .to_numpy()
        )

        if pd.isna(factual_positions).any():
            bad_groups = sorted(
                A[
                    pd.isna(factual_positions)
                ].unique().tolist()
            )

            raise ValueError(
                "Some factual groups were not represented in the "
                f"group universe: {bad_groups}"
            )

        factual_positions = factual_positions.astype(int)

        S_fact = score_matrix[
            np.arange(len(df)),
            factual_positions,
        ]

    else:
        # Backward-compatible behavior for score tables generated
        # before S_<group> columns were added.
        S_fact = (
            mu_fact >= float(tau)
        ).astype(int)

    cfpr = {}
    cfnr = {}
    fpr_obs = {}
    fnr_obs = {}

    # ---------------------------------------------------------
    # Calculate group-specific counterfactual and factual rates
    # ---------------------------------------------------------
    for group in groups:
        mu_column = f"{mu_prefix}{group}"
        score_column = f"{score_prefix}{group}"
        propensity_column = (
            f"{pi_prefix}{group}_prob"
        )

        if propensity_column not in df.columns:
            raise ValueError(
                "DR estimation requires a propensity column for "
                f"group={group!r}: {propensity_column!r}."
            )

        mu1_g = (
            df[mu_column]
            .to_numpy(dtype=float)
        )

        mu1_g = np.clip(
            mu1_g,
            eps,
            1.0 - eps,
        )

        mu0_g = 1.0 - mu1_g

        # Preserve the FairModel's actual decision rule whenever
        # counterfactual classifications are available.
        if score_column in df.columns:
            S_g = (
                df[score_column]
                .to_numpy()
                .astype(int)
            )
        else:
            S_g = (
                mu1_g >= float(tau)
            ).astype(int)

        raw_pi_g = (
            df[propensity_column]
            .to_numpy(dtype=float)
        )

        if np.isnan(raw_pi_g).all():
            raise ValueError(
                "The propensity column contains only missing values: "
                f"{propensity_column!r}."
            )

        pi_g = np.clip(
            raw_pi_g,
            eps,
            1.0 - eps,
        )

        A_is_g = (
            A == group
        ).to_numpy(dtype=float)

        inverse_probability_weight = (
            A_is_g / pi_g
        )

        # -----------------------------------------------------
        # Counterfactual FPR
        # -----------------------------------------------------
        #
        # Numerator target:
        #     E[S^g * (1 - Y^g)]
        #
        # Denominator target:
        #     E[1 - Y^g]
        #
        # AIPW form:
        #     w*(observed contribution)
        #       - (w - 1)*(outcome-regression contribution)
        #
        observed_num_0 = (
            S_g * (1 - y)
        ).astype(float)

        modeled_num_0 = (
            S_g * mu0_g
        ).astype(float)

        observed_den_0 = (
            1 - y
        ).astype(float)

        modeled_den_0 = (
            mu0_g
        ).astype(float)

        numerator_cfpr = np.nanmean(
            inverse_probability_weight
            * observed_num_0
            - (
                inverse_probability_weight - 1.0
            )
            * modeled_num_0
        )

        denominator_cfpr = np.nanmean(
            inverse_probability_weight
            * observed_den_0
            - (
                inverse_probability_weight - 1.0
            )
            * modeled_den_0
        )

        if (
            np.isfinite(denominator_cfpr)
            and denominator_cfpr > eps
        ):
            cfpr[group] = float(
                numerator_cfpr
                / denominator_cfpr
            )
        else:
            cfpr[group] = np.nan

        # -----------------------------------------------------
        # Counterfactual FNR
        # -----------------------------------------------------
        #
        # Numerator target:
        #     E[(1 - S^g) * Y^g]
        #
        # Denominator target:
        #     E[Y^g]
        #
        observed_num_1 = (
            (1 - S_g) * y
        ).astype(float)

        modeled_num_1 = (
            (1 - S_g) * mu1_g
        ).astype(float)

        observed_den_1 = (
            y
        ).astype(float)

        modeled_den_1 = (
            mu1_g
        ).astype(float)

        numerator_cfnr = np.nanmean(
            inverse_probability_weight
            * observed_num_1
            - (
                inverse_probability_weight - 1.0
            )
            * modeled_num_1
        )

        denominator_cfnr = np.nanmean(
            inverse_probability_weight
            * observed_den_1
            - (
                inverse_probability_weight - 1.0
            )
            * modeled_den_1
        )

        if (
            np.isfinite(denominator_cfnr)
            and denominator_cfnr > eps
        ):
            cfnr[group] = float(
                numerator_cfnr
                / denominator_cfnr
            )
        else:
            cfnr[group] = np.nan

        # -----------------------------------------------------
        # Observed factual FPR/FNR
        # -----------------------------------------------------
        factual_mask = (
            A == group
        ).to_numpy()

        factual_negative = (
            y == 0
        )

        factual_positive = (
            y == 1
        )

        denominator_fpr = int(
            (
                factual_negative
                & factual_mask
            ).sum()
        )

        denominator_fnr = int(
            (
                factual_positive
                & factual_mask
            ).sum()
        )

        if denominator_fpr > 0:
            fpr_obs[group] = float(
                (
                    (S_fact == 1)
                    & factual_negative
                    & factual_mask
                ).sum()
                / denominator_fpr
            )
        else:
            fpr_obs[group] = np.nan

        if denominator_fnr > 0:
            fnr_obs[group] = float(
                (
                    (S_fact == 0)
                    & factual_positive
                    & factual_mask
                ).sum()
                / denominator_fnr
            )
        else:
            fnr_obs[group] = np.nan

    return CfRates(
        cfpr=cfpr,
        cfnr=cfnr,
        fpr_obs=fpr_obs,
        fnr_obs=fnr_obs,
    )


#-------Pairwise summaries (defs)-----------

def _pairwise_abs_diffs(vals: List[float]) -> np.ndarray:
    """
    All pairwise absolute differences, ignoring NaNs/infs.

    Utility for disparity summaries (avg/max/var). See docs/outcome_models.md#_pairwise_abs_diffs
    """
    v = np.array(vals, dtype=float)
    n = len(v)
    if n <= 1:
        return np.array([np.nan])
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.isfinite(v[i]) and np.isfinite(v[j]):
                diffs.append(abs(v[i] - v[j]))
    return np.array(diffs) if diffs else np.array([np.nan])


def get_defs_from_rates(rates: CfRates) -> Dict[str, float]:
    """
    Summaries from groupwise rates.

    Aggregates cFPR/cFNR into avg/max/var (pos/neg), and includes per-group
    cfpr_*, cfnr_*, fpr_*, fnr_*. See docs/outcome_models.md#get_defs_from_rates
    """
    groups = sorted(rates.cfpr.keys())
    cfpr_vec = [rates.cfpr[g] for g in groups]
    cfnr_vec = [rates.cfnr[g] for g in groups]

    dpos = _pairwise_abs_diffs(cfpr_vec)
    dneg = _pairwise_abs_diffs(cfnr_vec)

    defs = {
        "avg_pos": float(np.nanmean(dpos)),
        "max_pos": float(np.nanmax(dpos)),
        "var_pos": float(np.nanvar(dpos)),
        "avg_neg": float(np.nanmean(dneg)),
        "max_neg": float(np.nanmax(dneg)),
        "var_neg": float(np.nanvar(dneg)),
    }
    for g, v in rates.cfpr.items():
        defs[f"cfpr_{g}"] = float(v)
    for g, v in rates.cfnr.items():
        defs[f"cfnr_{g}"] = float(v)
    for g, v in rates.fpr_obs.items():
        defs[f"fpr_{g}"] = float(v)
    for g, v in rates.fnr_obs.items():
        defs[f"fnr_{g}"] = float(v)
    return defs
