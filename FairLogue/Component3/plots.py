from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import pandas as pd
from .estimation_functions import ci_tint, ci_trunc, get_bs_rescaled
import matplotlib.pyplot as plt
import seaborn as sns



#---- Plotting helpers -----

def annotate_plot(ax: plt.Axes, u_value: float, x_pos: float = 0.0) -> None:
    """
    Mark a vertical u-value reference on an axis.

    Draws a dashed vline at `value` and labels it “u-value = …”.
    See docs/plots.md#annotate_plot
    """
    ax.axvline(x_pos, linestyle="--", color="red")
    ymax = ax.get_ylim()[1]
    ax.annotate(
        f"u-value = {u_value:.3f}",
        xy=(x_pos, ymax * 0.8),
        xytext=(x_pos, ymax * 0.8),
        ha="right",
        va="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5),
        arrowprops=dict(arrowstyle="->", lw=0.5),
    )


def _ecdf(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array([]), np.array([])
    x = np.sort(arr)
    y = np.arange(1, x.size + 1) / x.size
    return x, y

def get_plots(results: Dict[str, object], sampsize: Optional[int] = None, alpha: float = 0.05, m_factor: float = 0.75, delta_uval: float = 0.10,):
    """
    Assemble plot tables, optional figures, and compute u-values.

    Returns a tuple:
      (est_summaries, table_null_delta, table_uval)
    - `est_summaries`: tidy table of stats (+ CIs if bootstrap present)
    - `table_null_delta`: long frame of obs−null draws for aggregate stats
    - `table_uval`: 1×k frame of u-values for aggregate stats
    Figures (histograms and optional groupwise scatter) are created if a
    matplotlib backend is active. See docs/plots.md#get_plots
    """
    #Point estimates
    if 'defs' not in results or not isinstance(results['defs'], dict):
        raise ValueError("results['defs'] must be a dict of point estimates")
    est_named = results['defs']

    if sampsize is None:
        sampsize = len(results.get('est_choice', [])) or 1

    #Bootstrap CIs
    ci_trunc_df = pd.DataFrame()
    ci_trunc_cfpr = pd.DataFrame()
    ci_trunc_cfnr = pd.DataFrame()

    if 'boot_out' in results and isinstance(results['boot_out'], list) and results['boot_out']:
        bs_rescaled = get_bs_rescaled(results['boot_out'], est_named)

        #Summary stats
        for stat in ['avg_neg', 'avg_pos', 'max_neg', 'max_pos', 'var_neg', 'var_pos']:
            if stat in bs_rescaled.columns:
                ci_t = ci_trunc(ci_tint(bs_rescaled, est_named, stat, sampsize, alpha, m_factor), 'tint')
                row = ci_t.assign(stat=stat)[['stat', 'point_est', 'se_est', 'low_trans', 'high_trans']]
                ci_trunc_df = pd.concat([ci_trunc_df, row], ignore_index=True)

        #Per-group CFPR/CFNR
        cfpr_keys = [k for k in est_named if k.startswith('cfpr_') and 'marg' not in k]
        cfnr_keys = [k for k in est_named if k.startswith('cfnr_') and 'marg' not in k]

        for n in cfpr_keys:
            if n in bs_rescaled.columns:
                ct = ci_trunc(ci_tint(bs_rescaled, est_named, n, sampsize, alpha, m_factor), 'tint')
                ci_trunc_cfpr = pd.concat([ci_trunc_cfpr, ct.assign(stat=n)[['stat', 'point_est', 'se_est', 'low_trans', 'high_trans']]], ignore_index=True)
        for n in cfnr_keys:
            if n in bs_rescaled.columns:
                ct = ci_trunc(ci_tint(bs_rescaled, est_named, n, sampsize, alpha, m_factor), 'tint')
                ci_trunc_cfnr = pd.concat([ci_trunc_cfnr, ct.assign(stat=n)[['stat', 'point_est', 'se_est', 'low_trans', 'high_trans']]], ignore_index=True)

    #Assemble estimate table
    est_named_df = pd.DataFrame(list(est_named.items()), columns=['stat', 'value'])

    def _cat(stat: str) -> str:
        if 'cfpr_' in stat:
            return 'cfpr'
        if 'cfnr_' in stat:
            return 'cfnr'
        if stat in ['avg_neg', 'max_neg', 'var_neg']:
            return 'aggregate_neg'
        if stat in ['avg_pos', 'max_pos', 'var_pos']:
            return 'aggregate_pos'
        if stat.startswith('fpr_'):
            return 'fpr'
        if stat.startswith('fnr_'):
            return 'fnr'
        return 'other'

    est_named_df['sign'] = est_named_df['stat'].map(_cat)

    allcis_trunc = pd.DataFrame(columns=['stat', 'point_est', 'se_est', 'low_trans', 'high_trans'])
    if not ci_trunc_df.empty or not ci_trunc_cfpr.empty or not ci_trunc_cfnr.empty:
        allcis_trunc = pd.concat([ci_trunc_df, ci_trunc_cfpr, ci_trunc_cfnr], ignore_index=True)

    est_summaries = est_named_df.merge(allcis_trunc, on='stat', how='left')

    #Null deltas + u-values
    def _uval(vec_null: pd.Series, obs: float, delta: float) -> float:
        v = pd.to_numeric(vec_null, errors='coerce').to_numpy()
        v = v[np.isfinite(v)]
        if not np.isfinite(obs) or v.size == 0:
            return np.nan
        return float(np.mean(obs - v > delta))

    table_null_delta = None
    table_uval = None

    table_null_raw = results.get("table_null", None)


    if table_null_raw is not None:
        # Normalize to a DataFrame
        if isinstance(table_null_raw, pd.DataFrame):
            null_df = table_null_raw.copy()
        elif isinstance(table_null_raw, dict):
            null_df = pd.DataFrame(table_null_raw)
        else:
            # last resort: try to turn it into a DataFrame
            null_df = pd.DataFrame(table_null_raw)

        keep = ["avg_neg", "avg_pos", "max_neg", "max_pos", "var_neg", "var_pos"]

        est_named_obs = pd.DataFrame(
            [(k, v) for k, v in est_named.items() if k in keep],
            columns=["stat", "value_obs"],
        )

        null_subset = null_df[[c for c in keep if c in null_df.columns]]
        table_null_delta = (
            null_subset
            .melt(var_name="stat", value_name="value_null")
            .merge(est_named_obs, on="stat", how="left")
        )
        table_null_delta["obs_minus_null"] = (
            table_null_delta["value_obs"] - table_null_delta["value_null"]
        )


        def _uval(vec_null: pd.Series, obs: float, delta: float) -> float:
            v = pd.to_numeric(vec_null, errors="coerce").to_numpy()
            v = v[np.isfinite(v)]
            if not np.isfinite(obs) or v.size == 0:
                return np.nan
            return float(np.mean(obs - v > delta))

        table_uval = pd.DataFrame(
            {
                "avg_neg": [_uval(null_df.get("avg_neg", pd.Series(dtype=float)),
                                est_named.get("avg_neg", np.nan), delta_uval)],
                "avg_pos": [_uval(null_df.get("avg_pos", pd.Series(dtype=float)),
                                est_named.get("avg_pos", np.nan), delta_uval)],
                "max_neg": [_uval(null_df.get("max_neg", pd.Series(dtype=float)),
                                est_named.get("max_neg", np.nan), delta_uval)],
                "max_pos": [_uval(null_df.get("max_pos", pd.Series(dtype=float)),
                                est_named.get("max_pos", np.nan), delta_uval)],
                "var_neg": [_uval(null_df.get("var_neg", pd.Series(dtype=float)),
                                est_named.get("var_neg", np.nan), delta_uval)],
                "var_pos": [_uval(null_df.get("var_pos", pd.Series(dtype=float)),
                                est_named.get("var_pos", np.nan), delta_uval)],
            }
        ).round(3)

        #6 panel histogram
        if table_null_delta is not None and table_uval is not None:
            stats_grid = [
                ("avg_neg", "Average (negative)"),
                ("max_neg", "Maximum (negative)"),
                ("var_neg", "Variational (negative)"),
                ("avg_pos", "Average (positive)"),
                ("max_pos", "Maximum (positive)"),
                ("var_pos", "Variational (positive)"),
            ]

            fig, axes = plt.subplots(3, 2, figsize=(15, 18))

            for ax, (stat, title) in zip(axes.flatten(), stats_grid):
                vals = (
                    table_null_delta.loc[table_null_delta["stat"] == stat, "obs_minus_null"]
                    .dropna()
                    .to_numpy()
                )

                if vals.size > 0:
                    # KDE without clipping; full distribution
                    sns.kdeplot(
                        vals,
                        ax=ax,
                        fill=True,
                        bw_adjust=0.5,
                        color="steelblue",
                    )

                    # Choose data-driven limits, then expand to include 0
                    lo = np.percentile(vals, 1.0)
                    hi = np.percentile(vals, 99.0)
                    xmin = min(lo, 0.0)
                    xmax = max(hi, 0.0)
                    ax.set_xlim([xmin, xmax])

                    # vertical red bar at 0, labeled with the u-value
                    uval = float(table_uval[stat].values[0])
                    annotate_plot(ax, uval, x_pos=0.0)

                ax.set_title(title, fontsize=14)
                ax.set_xlabel("Obs. − Null")
                ax.set_ylabel("Density")

            plt.tight_layout()
            plt.show()

            neg_stats = [("avg_neg", "Average"), ("max_neg", "Maximum"), ("var_neg", "Variational")]
            pos_stats = [("avg_pos", "Average"), ("max_pos", "Maximum"), ("var_pos", "Variational")]

            fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300, sharey=True)

            for ax, stats, panel_title in [
                (axes[0], neg_stats, "Negative"),
                (axes[1], pos_stats, "Positive"),
            ]:
                ax.axvline(delta_uval, linestyle=":", linewidth=1)
                ax.set_title(f"ECDF of Observed vs Counterfactual Disparities ({panel_title})")
                ax.set_xlabel("Obs. − Null")
                ax.set_ylim(0, 1)

                for stat, label in stats:
                    vals = (
                        table_null_delta.loc[table_null_delta["stat"] == stat, "obs_minus_null"]
                        .dropna()
                        .to_numpy()
                    )
                    xs, ys = _ecdf(vals)
                    if xs.size == 0:
                        continue

                    uval = float(table_uval[stat].values[0])
                    ax.plot(xs, ys, linewidth=2, label=f"{label} (u={uval:.3f})")

                ax.grid(True, linewidth=0.5, alpha=0.4)
                ax.legend(frameon=True, fontsize=10)


            axes[0].set_ylabel("ECDF")
            plt.tight_layout()
            plt.show()



    subgroup_cols = [
        c for c in null_df.columns
        if c.startswith("cfpr_") or c.startswith("cfnr_")
    ]

    if subgroup_cols:
        tmp = null_df[subgroup_cols].copy()
        tmp["draw"] = np.arange(len(tmp), dtype=int)

        group_null_long = tmp.melt(
            id_vars="draw",
            var_name="stat",
            value_name="value_null",
        )

        group_null_long["metric"] = group_null_long["stat"].str.split("_", n=1).str[0]
        group_null_long["group"]  = group_null_long["stat"].str.split("_", n=1).str[1]
    else:
        group_null_long = pd.DataFrame(
            columns=["draw", "stat", "metric", "group", "value_null"]
        )

    # ------------------------------------------------------------------
    # Plot subgroup null distributions (boxplots)
    # ------------------------------------------------------------------
    ACCENT = "#0072B2"
    BOX = "#D9D9D9"       
    EDGE = "#4D4D4D"      

    sns.set_theme(style="whitegrid", context="talk")  

    if not group_null_long.empty:
        for metric in ["cfnr", "cfpr"]:
            d = group_null_long[group_null_long["metric"] == metric]
            if d.empty:
                continue

            order = (
                d.groupby("group")["value_null"]
                .mean()
                .sort_values(ascending=False)
                .index.tolist()
            )

            summ = (
                d.groupby("group")["value_null"]
                .agg(
                    mean="mean",
                    lo=lambda x: np.nanpercentile(x, 2.5),
                    hi=lambda x: np.nanpercentile(x, 97.5),
                )
                .reindex(order)
                .reset_index()
            )

            plt.figure(figsize=(12, 6))
            ax = sns.boxplot(
                data=d,
                x="group",
                y="value_null",
                order=order,
                showfliers=False,
                width=0.6,
                boxprops=dict(facecolor=BOX, edgecolor=EDGE, linewidth=1.2),
                whiskerprops=dict(color=EDGE, linewidth=1.2),
                capprops=dict(color=EDGE, linewidth=1.2),
                medianprops=dict(color=EDGE, linewidth=2.0),
            )

            x = np.arange(len(order))
            y = summ["mean"].to_numpy()
            lo = summ["lo"].to_numpy()
            hi = summ["hi"].to_numpy()

            ax.scatter(x, y, color=ACCENT, s=40, zorder=3)
            ax.errorbar(
                x,
                y,
                yerr=np.vstack([y - lo, hi - y]),
                fmt="none",
                ecolor=ACCENT,
                elinewidth=2.0,
                capsize=4,
                zorder=3,
            )

            ax.set_xlabel("Group")
            ax.set_ylabel(f"Null {metric.upper()}")
            ax.set_title(f"Null distribution of subgroup {metric.upper()}")
            ax.tick_params(axis="x", rotation=45)

            # optional: remove top/right spines for a cleaner look
            sns.despine(ax=ax)

            plt.tight_layout()
            plt.show()
    
    #per-group scatter (with error bars if CI present)
    try:
        for sign, ylabel in [("cfpr", "Group cFPR Estimate"), ("cfnr", "Group cFNR Estimate")]:
            df_sig = est_summaries[est_summaries['sign'] == sign]
            if not df_sig.empty:
                plt.figure(figsize=(10, 6))
                x = np.arange(len(df_sig))
                yvals = df_sig['value'].to_numpy(dtype=float)
                plt.scatter(x, yvals)
                if {'low_trans', 'high_trans'}.issubset(df_sig.columns):
                    lows = (yvals - df_sig['low_trans'].to_numpy()).clip(min=0)
                    highs = (df_sig['high_trans'].to_numpy() - yvals).clip(min=0)
                    plt.errorbar(x, yvals, yerr=[lows, highs], fmt='none', capsize=5, elinewidth=2, alpha=0.8)
                plt.xticks(x, df_sig['stat'].tolist(), rotation=45, ha='right')
                plt.ylabel(ylabel)
                plt.tight_layout()
    except Exception:
        pass

    return est_summaries, table_null_delta, table_uval, group_null_long


