"""Visualization module.

Provides plotting utilities for simulation rejection-rate curves,
MCB–DSC scatter diagrams, time-series overlays, and recalibration
diagnostic plots.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess


# Use a clean style consistent throughout the project
matplotlib.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
})


def plot_rejection_rates_mean(
    results_df: pd.DataFrame,
    k_grid: np.ndarray,
    save_path: str | None = None,
) -> None:
    """Plot rejection rates vs k for all 12 mean-forecast scenarios.

    Produces a 6×2 subplot grid (one panel per scenario, ordered 1a/1b … 6a/6b).
    Three lines per panel:
      - Blue solid  : MCB test rejection rate
      - Orange solid: DSC test rejection rate
      - Gray dotted : DM test rejection rate

    A nominal-level horizontal dashed line at 0.05 is shown in each panel.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain columns: 'scenario_id', 'k', 'rej_mcb', 'rej_dsc', 'rej_dm'.
    k_grid : np.ndarray
        Grid of k values used in the simulation (x-axis).
    save_path : str or None
        If provided, save the figure to this path (format inferred from extension).
    """
    scenario_order = ["1a", "1b", "2a", "2b", "3a", "3b", "4a", "4b", "5a", "5b", "6a", "6b"]
    n_rows, n_cols = 6, 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 15), sharex=True, sharey=True)
    fig.suptitle("Rejection rates – mean forecast scenarios", fontsize=11, y=1.01)

    for idx, sc_id in enumerate(scenario_order):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        sub = results_df[results_df["scenario_id"] == sc_id].sort_values("k")
        if sub.empty:
            ax.set_visible(False)
            continue

        ax.plot(sub["k"], sub["rej_mcb"], color="steelblue", lw=1.5, label="MCB")
        ax.plot(sub["k"], sub["rej_dsc"], color="darkorange", lw=1.5, label="DSC")
        ax.plot(sub["k"], sub["rej_dm"],  color="gray", lw=1.2, ls=":", label="DM")
        ax.axhline(0.05, color="black", lw=0.7, ls="--", alpha=0.5)

        ax.set_title(f"Scenario {sc_id}", fontsize=9)
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    # Shared axis labels
    for ax in axes[-1]:
        ax.set_xlabel("k")
    for row in axes:
        row[0].set_ylabel("Rejection rate")

    # Legend in the last panel
    handles = [
        mlines.Line2D([], [], color="steelblue",  lw=1.5, label="MCB"),
        mlines.Line2D([], [], color="darkorange", lw=1.5, label="DSC"),
        mlines.Line2D([], [], color="gray",       lw=1.2, ls=":", label="DM"),
    ]
    axes[0][1].legend(handles=handles, loc="upper left", fontsize=8, frameon=False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_rejection_rates_quantile(
    results_df: pd.DataFrame,
    k_grid: np.ndarray,
    alpha_list: list[float],
    save_path: str | None = None,
) -> None:
    """Plot rejection rates vs k for quantile-forecast scenarios by alpha level.

    Produces a grid of (n_scenarios × n_alpha) subplots.  Line styles:
      - Blue solid  : MCB test
      - Orange solid: DSC test
      - Gray dotted : DM test

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain columns: 'scenario_id', 'alpha', 'k', 'rej_mcb', 'rej_dsc', 'rej_dm'.
    k_grid : np.ndarray
    alpha_list : list of float
        Quantile levels (e.g. [0.1, 0.5, 0.9]).
    save_path : str or None
    """
    scenario_ids = results_df["scenario_id"].unique()
    n_sc = len(scenario_ids)
    n_al = len(alpha_list)

    fig, axes = plt.subplots(
        n_sc, n_al,
        figsize=(3.5 * n_al, 3.2 * n_sc),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig.suptitle("Rejection rates – quantile forecast scenarios", fontsize=11, y=1.01)

    for r, sc_id in enumerate(scenario_ids):
        for c, alpha in enumerate(alpha_list):
            ax = axes[r][c]
            sub = results_df[
                (results_df["scenario_id"] == sc_id) & (results_df["alpha"] == alpha)
            ].sort_values("k")

            if sub.empty:
                ax.set_visible(False)
                continue

            ax.plot(sub["k"], sub["rej_mcb"], color="steelblue",  lw=1.5, label="MCB")
            ax.plot(sub["k"], sub["rej_dsc"], color="darkorange", lw=1.5, label="DSC")
            ax.plot(sub["k"], sub["rej_dm"],  color="gray",       lw=1.2, ls=":", label="DM")
            ax.axhline(0.05, color="black", lw=0.7, ls="--", alpha=0.5)

            ax.set_title(f"Sc {sc_id}, α={alpha}", fontsize=8)
            ax.set_ylim(-0.02, 1.02)

    for ax in axes[-1]:
        ax.set_xlabel("k")
    for row in axes:
        row[0].set_ylabel("Rejection rate")

    handles = [
        mlines.Line2D([], [], color="steelblue",  lw=1.5, label="MCB"),
        mlines.Line2D([], [], color="darkorange", lw=1.5, label="DSC"),
        mlines.Line2D([], [], color="gray",       lw=1.2, ls=":", label="DM"),
    ]
    axes[0][-1].legend(handles=handles, loc="upper left", fontsize=8, frameon=False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_mcb_dsc_scatter(
    mcb_vals: np.ndarray,
    dsc_vals: np.ndarray,
    labels: list[str],
    unc: float,
    score_vals: np.ndarray,
    title: str = "",
    save_path: str | None = None,
) -> None:
    """MCB–DSC scatter plot with iso-score lines.

    The decomposition  S = MCB - DSC + UNC  implies  DSC = MCB + UNC - S,
    so iso-score lines (constant S) are 45-degree lines in the (MCB, DSC)
    plane:  DSC = MCB + (UNC - S).

    Parameters
    ----------
    mcb_vals : np.ndarray, shape (N,)
        MCB estimates for N forecasting models / replications.
    dsc_vals : np.ndarray, shape (N,)
        DSC estimates.
    labels : list of str, length N
        Point labels (model names or replication indices).
    unc : float
        Uncertainty component value (common to all points).
    score_vals : np.ndarray, shape (N,)
        Mean score values; used to draw iso-score lines.
    title : str
    save_path : str or None
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    # Iso-score lines
    mcb_range = np.array([
        min(0.0, float(mcb_vals.min())) - 0.05,
        float(mcb_vals.max()) + 0.05,
    ])
    palette = plt.cm.Greys(np.linspace(0.3, 0.7, len(score_vals)))
    for i, s_val in enumerate(np.sort(score_vals)):
        intercept = unc - s_val
        ax.plot(
            mcb_range,
            mcb_range + intercept,
            color=palette[i],
            lw=0.8,
            ls="--",
            alpha=0.7,
            label=f"S={s_val:.3f}",
        )

    # Scatter points
    colors = sns.color_palette("tab10", n_colors=len(mcb_vals))
    for i, (mcb, dsc, lbl) in enumerate(zip(mcb_vals, dsc_vals, labels)):
        ax.scatter(mcb, dsc, color=colors[i], s=60, zorder=5)
        ax.annotate(lbl, (mcb, dsc), textcoords="offset points", xytext=(5, 3), fontsize=7)

    ax.set_xlabel("MCB")
    ax.set_ylabel("DSC")
    ax.set_title(title)
    ax.axhline(0, color="black", lw=0.5, ls="-")
    ax.axvline(0, color="black", lw=0.5, ls="-")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_time_series(
    dates,
    Y: np.ndarray,
    X1: np.ndarray,
    X2: np.ndarray,
    labels: list[str],
    title: str = "",
    save_path: str | None = None,
) -> None:
    """Plot realized values and two forecast series over time.

    Parameters
    ----------
    dates : array-like
        Datetime or integer index for the x-axis.
    Y : np.ndarray, shape (T,)
        Realized values.
    X1 : np.ndarray, shape (T,)
        First forecast series.
    X2 : np.ndarray, shape (T,)
        Second forecast series.
    labels : list of str, length 3
        Labels for [Y, X1, X2].
    title : str
    save_path : str or None
    """
    fig, ax = plt.subplots(figsize=(10, 3.5))

    ax.plot(dates, Y,  color="black",      lw=1.0, label=labels[0], zorder=3)
    ax.plot(dates, X1, color="steelblue",  lw=1.0, ls="--", label=labels[1], zorder=2)
    ax.plot(dates, X2, color="darkorange", lw=1.0, ls="-.", label=labels[2], zorder=2)

    ax.set_title(title)
    ax.set_xlabel("Date / time index")
    ax.set_ylabel("Value")
    ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_recalibration_diagnostic(
    X: np.ndarray,
    residuals: np.ndarray,
    label: str,
    save_path: str | None = None,
) -> None:
    """Diagnostic plot of residuals against the original forecast.

    Displays the scatter of residuals  e_t = Y_t - W_t^T theta_hat  against
    the forecast X_t, together with a LOWESS smoother and approximate 95%
    pointwise confidence bands (smoother ± 1.96 * local std of residuals).

    Parameters
    ----------
    X : np.ndarray, shape (T,)
        Original forecast values (x-axis).
    residuals : np.ndarray, shape (T,)
        Recalibration residuals  Y - W^T theta_hat.
    label : str
        Forecast model label (used in the title).
    save_path : str or None
    """
    sort_idx = np.argsort(X)
    X_s = X[sort_idx]
    r_s = residuals[sort_idx]

    # LOWESS smoother (fraction chosen for a reasonable level of smoothing)
    frac = min(0.6, max(0.2, 40.0 / len(X)))
    # sm_lowess returns array of shape (n, 2): [:, 0] = sorted x, [:, 1] = smoothed y
    lowess_out = sm_lowess(r_s, X_s, frac=frac, return_sorted=True)
    smooth = lowess_out[:, 1]

    # Local std using rolling window approximation (binning approach)
    n_bins = max(10, len(X) // 20)
    bin_edges = np.linspace(X_s.min(), X_s.max(), n_bins + 1)
    local_std = np.full_like(X_s, np.std(residuals))
    for b in range(n_bins):
        mask = (X_s >= bin_edges[b]) & (X_s < bin_edges[b + 1])
        if mask.sum() >= 3:
            local_std[mask] = np.std(r_s[mask])

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.scatter(X, residuals, alpha=0.3, s=10, color="steelblue", label="Residuals")
    ax.plot(X_s, smooth, color="red", lw=1.5, label="LOWESS")
    ax.fill_between(
        X_s,
        smooth - 1.96 * local_std,
        smooth + 1.96 * local_std,
        alpha=0.15,
        color="red",
        label="95% CI (local)",
    )
    ax.axhline(0, color="black", lw=0.8, ls="--")

    ax.set_xlabel(f"Forecast {label}")
    ax.set_ylabel("Residual  Y − Ŷ")
    ax.set_title(f"Recalibration diagnostic – {label}")
    ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()
