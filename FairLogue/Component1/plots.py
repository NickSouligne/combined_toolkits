from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def _plot_bar(series: pd.Series, title: str, ylabel: str):
    fig, ax = plt.subplots()
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    return fig

def _plot_bar_series_by_group(df: pd.DataFrame, value_col: str, title: str, ylabel: str):
    fig, ax = plt.subplots()
    s = df.set_index("group")[value_col]
    s.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.axhline(0.0, linestyle="--", linewidth=0.8)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    return fig

def _plot_grouped_eods_components(df: pd.DataFrame):
    #two bars per group: eod_tpr_diff and eod_fpr_diff
    fig, ax = plt.subplots()
    idx = np.arange(len(df))
    width = 0.4
    ax.bar(idx - width/2, df["eod_tpr_diff"].values, width, label="TPR diff")
    ax.bar(idx + width/2, df["eod_fpr_diff"].values, width, label="FPR diff")
    ax.set_title("Equalized Odds Components by Group (vs privileged)")
    ax.set_ylabel("Difference")
    ax.set_xticks(idx)
    ax.set_xticklabels(df["group"].tolist(), rotation=45, ha="right")
    ax.axhline(0.0, linestyle="--", linewidth=0.8)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_fairness_matrix(
    df: pd.DataFrame,
    *,
    group_col: str = "group",
    metric_cols=None,
    title: str = "Fairness Landscape Matrix (Groups x Metrics)",
    annotate: bool = True,
    sort_by: str | None = "eod_max_abs",  #set None to keep incoming order
    max_groups: int | None = None,        #set to e.g. 30 to avoid overcrowding
    normalize: str = "zscore"             #"zscore", "minmax", or "none"
):
    """
    Heatmap: rows=groups, cols=metrics. Optionally normalize columns so metrics
    with different scales are visually comparable.

    Parameters
    ----------
    df : DataFrame
        Expected to include `group_col` and metric columns.
    metric_cols : list[str] | None
        Which columns to plot. If None, uses a sensible default if present.
    sort_by : str | None
        Sort groups by a metric (descending). Useful to bring most-disparate to top.
    max_groups : int | None
        If set, keep only top max_groups after sorting.
    normalize : {"zscore","minmax","none"}
        Column-wise normalization for visualization.
    """
    if metric_cols is None:
        #Pick fairness metrics if they were computed
        candidates = [
            "positive_rate", "tpr", "fpr",
            "eo_diff", "eod_tpr_diff", "eod_fpr_diff", "eod_max_abs"
        ]
        metric_cols = [c for c in candidates if c in df.columns]

    #Handle missing data 
    if group_col not in df.columns:
        raise KeyError(f"'{group_col}' not found in df")
    missing = [c for c in metric_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing metric columns: {missing}")

    d = df[[group_col] + metric_cols].copy()

    #Sort by the largest disparities first
    if sort_by is not None and sort_by in d.columns:
        d = d.sort_values(sort_by, ascending=False)

    #Optional truncation for readability
    if max_groups is not None and max_groups > 0:
        d = d.head(max_groups)

    #Build matrix
    groups = d[group_col].astype(str).tolist()
    M = d[metric_cols].to_numpy(dtype=float)

    #Column-wise normalization for comparability
    M_plot = M.copy()
    if normalize == "zscore":
        mu = np.nanmean(M_plot, axis=0)
        sd = np.nanstd(M_plot, axis=0)
        sd = np.where(sd == 0, 1.0, sd)
        M_plot = (M_plot - mu) / sd
    elif normalize == "minmax":
        mn = np.nanmin(M_plot, axis=0)
        mx = np.nanmax(M_plot, axis=0)
        denom = np.where((mx - mn) == 0, 1.0, (mx - mn))
        M_plot = (M_plot - mn) / denom
    elif normalize == "none":
        pass
    else:
        raise ValueError("normalize must be one of: 'zscore', 'minmax', 'none'")

    #Plot
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(metric_cols) + 2),
                                    max(4, 0.25 * len(groups) + 2)))

    im = ax.imshow(M_plot, aspect="auto")

    ax.set_title(title)
    ax.set_xticks(np.arange(len(metric_cols)))
    ax.set_xticklabels(metric_cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(groups)))
    ax.set_yticklabels(groups)

    #Annotate with raw values for interpretability
    if annotate:
        for i in range(len(groups)):
            for j in range(len(metric_cols)):
                val = M[i, j]
                if np.isnan(val):
                    txt = "NA"
                else:
                    #compact formatting
                    txt = f"{val:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    #Set label based on normalization
    if normalize == "zscore":
        cbar.set_label("Z-score (column-wise)")
    elif normalize == "minmax":
        cbar.set_label("Min-max scaled (column-wise)")
    else:
        cbar.set_label("Raw value")

    fig.tight_layout()
    return fig
