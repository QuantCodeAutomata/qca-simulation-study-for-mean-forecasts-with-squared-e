"""Experiment 5: Score decomposition for financial VaR forecasts.

Implements VaR forecasting (HS, GARCH, GJR-GARCH) and evaluation via:
  - Score decomposition (check loss at levels alpha=0.01 and alpha=0.05)
  - Standard VaR backtests: UC, CC, DQ, DQX, VQR

NOTE: Real analysis requires E-mini S&P 500 tick data from a commercial provider.
This script uses *synthetic* daily returns to demonstrate the full framework.

Saves
-----
- results/exp5_var_backtest_table.csv
- results/exp5_mcb_dsc_1pct.png
- results/exp5_mcb_dsc_5pct.png
- results/exp5_summary.json
"""

import sys
import json
import logging
import functools
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import scipy.stats
import statsmodels.api as sm

from src.decomposition import check_loss, compute_decomposition
from src.recalibration import fit_recalibration_quantile, compute_reference_quantile
from src.hac import estimate_hac_covariance
from src.inference import (
    build_score_process_cl,
    build_xi_matrix,
    boundary_pvalue_quantile_mcb,
    boundary_pvalue_quantile_dsc,
    equal_mcb_test,
    equal_dsc_test,
    dm_test,
)
from src.visualization import plot_mcb_dsc_scatter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HS_WINDOW = 250
ALPHA_LIST = [0.01, 0.05]


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_synthetic_returns_var(
    T_train: int = 1500,
    T_eval: int = 3617,
    seed: int = 99,
) -> tuple:
    """Generate synthetic daily returns from a GJR-GARCH(1,1,1) process.

    NOTE: Using synthetic data; real analysis requires E-mini tick data
    from a commercial data provider.

    Returns
    -------
    (returns_train, returns_eval, dates_eval)
    """
    rng = np.random.default_rng(seed)
    T_total = T_train + T_eval + 200  # burn-in

    # GJR-GARCH parameters (realistic values)
    omega, alpha_g, gamma_asym, beta_g = 0.02, 0.05, 0.08, 0.88
    h = np.zeros(T_total)
    r = np.zeros(T_total)
    h[0] = omega / (1.0 - alpha_g - 0.5 * gamma_asym - beta_g)
    for t in range(1, T_total):
        indicator = 1.0 if r[t - 1] < 0 else 0.0
        h[t] = (omega
                + (alpha_g + gamma_asym * indicator) * r[t - 1] ** 2
                + beta_g * h[t - 1])
        r[t] = np.sqrt(h[t]) * rng.standard_normal()

    r_used = r[200:]  # discard burn-in
    returns_train = r_used[:T_train]
    returns_eval = r_used[T_train:]
    dates_eval = pd.date_range("2003-01-02", periods=T_eval, freq="B")
    return returns_train, returns_eval, dates_eval


# ---------------------------------------------------------------------------
# VaR forecasting models
# ---------------------------------------------------------------------------

def forecast_var_hs(
    returns: np.ndarray,
    alpha: float,
    window: int = HS_WINDOW,
) -> np.ndarray:
    """Historical Simulation VaR: rolling empirical alpha-quantile of past returns.

    VaR_t(alpha) = empirical alpha-quantile of {r_{t-window}, ..., r_{t-1}}.

    Parameters
    ----------
    returns : ndarray, shape (T_train + T_eval,)
        Full return series (training + evaluation).
    alpha : float
    window : int

    Returns
    -------
    forecasts : ndarray, shape (T_eval,)
        Rolling VaR forecasts over the evaluation period.
    """
    T = len(returns)
    T_eval = T - window
    forecasts = np.empty(T_eval)
    for i in range(T_eval):
        w = returns[i:i + window]
        forecasts[i] = np.quantile(w, alpha, method="midpoint")
    return forecasts


def forecast_var_garch(
    returns_train: np.ndarray,
    returns_eval: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """GARCH(1,1) VaR using Gaussian quantile.

    VaR_t(alpha) = mu + sigma_t * z_alpha  where z_alpha = Phi^{-1}(alpha).

    Parameters
    ----------
    returns_train : ndarray
    returns_eval  : ndarray
    alpha : float

    Returns
    -------
    forecasts : ndarray, shape (T_eval,)
    """
    from arch import arch_model

    z_alpha = scipy.stats.norm.ppf(alpha)
    returns_pct = returns_train * 100.0

    am = arch_model(returns_pct, mean="Constant", vol="Garch", p=1, q=1,
                    dist="normal", rescale=False)
    try:
        res = am.fit(disp="off", show_warning=False)
        omega_g = res.params["omega"]
        alpha_g = res.params.get("alpha[1]", 0.09)
        beta_g = res.params.get("beta[1]", 0.90)
        mu = res.params.get("mu", 0.0)
        h = float(res.conditional_volatility[-1] ** 2)
    except Exception as exc:
        logger.warning("GARCH fit failed (%s); using naive GARCH parameters", exc)
        omega_g, alpha_g, beta_g, mu = 0.02, 0.09, 0.88, 0.0
        h = float(np.var(returns_train)) * 10000.0

    returns_all_pct = np.concatenate([returns_train * 100.0, returns_eval * 100.0])
    T_train = len(returns_train)
    forecasts = np.empty(len(returns_eval))

    for i in range(len(returns_eval)):
        t = T_train + i
        r_prev = returns_all_pct[t - 1]
        h = omega_g + alpha_g * r_prev ** 2 + beta_g * h
        sigma = np.sqrt(h) / 100.0  # back to decimal scale
        forecasts[i] = mu / 100.0 + sigma * z_alpha

    return forecasts


def forecast_var_gjr(
    returns_train: np.ndarray,
    returns_eval: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """GJR-GARCH(1,1) VaR using Student-t quantile.

    Parameters
    ----------
    returns_train, returns_eval : ndarray
    alpha : float

    Returns
    -------
    forecasts : ndarray, shape (T_eval,)
    """
    from arch import arch_model

    returns_pct = returns_train * 100.0

    am = arch_model(returns_pct, mean="Constant", vol="Garch", p=1, o=1, q=1,
                    dist="t", rescale=False)
    try:
        res = am.fit(disp="off", show_warning=False)
        omega_g = res.params["omega"]
        alpha_g = res.params.get("alpha[1]", 0.05)
        gamma_asym = res.params.get("gamma[1]", 0.08)
        beta_g = res.params.get("beta[1]", 0.88)
        mu = res.params.get("mu", 0.0)
        nu = float(res.params.get("nu", 8.0))
        nu = max(nu, 2.01)
        h = float(res.conditional_volatility[-1] ** 2)
    except Exception as exc:
        logger.warning("GJR-GARCH fit failed (%s); falling back to GARCH", exc)
        return forecast_var_garch(returns_train, returns_eval, alpha)

    z_alpha = scipy.stats.t.ppf(alpha, df=nu) * np.sqrt((nu - 2.0) / nu)

    returns_all_pct = np.concatenate([returns_train * 100.0, returns_eval * 100.0])
    T_train = len(returns_train)
    forecasts = np.empty(len(returns_eval))

    for i in range(len(returns_eval)):
        t = T_train + i
        r_prev = returns_all_pct[t - 1]
        ind = 1.0 if r_prev < 0 else 0.0
        h = omega_g + (alpha_g + gamma_asym * ind) * r_prev ** 2 + beta_g * h
        sigma = np.sqrt(h) / 100.0
        forecasts[i] = mu / 100.0 + sigma * z_alpha

    return forecasts


# ---------------------------------------------------------------------------
# Backtests
# ---------------------------------------------------------------------------

def _hit_sequence(returns: np.ndarray, var_forecasts: np.ndarray, alpha: float) -> np.ndarray:
    """Binary hit sequence: 1 if return falls below VaR forecast, else 0."""
    return (returns < var_forecasts).astype(float)


def run_uc_backtest(
    returns: np.ndarray,
    var_forecasts: np.ndarray,
    alpha: float,
) -> float:
    """Unconditional coverage (UC) Kupiec test.

    LR_UC = -2 * log[ alpha^n * (1-alpha)^(T-n) / p_hat^n * (1-p_hat)^(T-n) ]
    ~ chi^2(1) under H0.

    Returns p-value.
    """
    hits = _hit_sequence(returns, var_forecasts, alpha)
    T = len(hits)
    n = int(hits.sum())
    p_hat = n / T

    if p_hat == 0.0 or p_hat == 1.0:
        return 0.0

    lr = 2.0 * (
        n * np.log(p_hat / alpha)
        + (T - n) * np.log((1.0 - p_hat) / (1.0 - alpha))
    )
    return float(scipy.stats.chi2.sf(lr, df=1))


def run_cc_backtest(
    returns: np.ndarray,
    var_forecasts: np.ndarray,
    alpha: float,
) -> float:
    """Conditional coverage (CC) Christoffersen test.

    LR_CC = LR_UC + LR_ind ~ chi^2(2) under H0.

    Returns p-value.
    """
    hits = _hit_sequence(returns, var_forecasts, alpha)
    T = len(hits)
    n = int(hits.sum())
    p_hat = n / T

    # Transition counts
    T00 = int(np.sum((hits[:-1] == 0) & (hits[1:] == 0)))
    T01 = int(np.sum((hits[:-1] == 0) & (hits[1:] == 1)))
    T10 = int(np.sum((hits[:-1] == 1) & (hits[1:] == 0)))
    T11 = int(np.sum((hits[:-1] == 1) & (hits[1:] == 1)))

    pi0 = T01 / (T00 + T01) if (T00 + T01) > 0 else 0.0
    pi1 = T11 / (T10 + T11) if (T10 + T11) > 0 else 0.0
    pi_bar = (T01 + T11) / (T00 + T01 + T10 + T11)

    def safe_log(x):
        return np.log(x) if x > 0 else 0.0

    lr_ind = 2.0 * (
        T00 * safe_log(1.0 - pi0) + T01 * safe_log(pi0)
        + T10 * safe_log(1.0 - pi1) + T11 * safe_log(pi1)
        - (T00 + T10) * safe_log(1.0 - pi_bar) - (T01 + T11) * safe_log(pi_bar)
    )

    if p_hat == 0.0 or p_hat == 1.0:
        lr_uc = 0.0
    else:
        lr_uc = 2.0 * (
            n * safe_log(p_hat / alpha)
            + (T - n) * safe_log((1.0 - p_hat) / (1.0 - alpha))
        )

    lr_cc = lr_uc + lr_ind
    return float(scipy.stats.chi2.sf(lr_cc, df=2))


def run_dq_backtest(
    returns: np.ndarray,
    var_forecasts: np.ndarray,
    alpha: float,
    lags: int = 4,
) -> float:
    """Dynamic Quantile (DQ) test.

    Regresses (hit_t - alpha) on lagged (hit_{t-j} - alpha) for j=1,...,lags
    plus a constant.  Wald statistic for H0: all coefficients = 0 ~ chi^2(lags+1).

    Returns p-value.
    """
    hits = _hit_sequence(returns, var_forecasts, alpha) - alpha
    T = len(hits)
    if T <= lags + 1:
        return 1.0

    y = hits[lags:]
    Xmat = np.column_stack(
        [np.ones(T - lags)] + [hits[lags - j:-j] for j in range(1, lags + 1)]
    )

    try:
        res = sm.OLS(y, Xmat).fit(cov_type="HC0")
        wald = float(res.ess / res.mse_resid)  # F-statistic * k
        wald_stat = wald * (lags + 1)
        return float(scipy.stats.chi2.sf(wald_stat, df=lags + 1))
    except Exception:
        return 1.0


def run_dqx_backtest(
    returns: np.ndarray,
    var_forecasts: np.ndarray,
    alpha: float,
    lags: int = 4,
) -> float:
    """DQ test with VaR included as a regressor.

    Adds VaR_t to the DQ regressors; H0 is that all coefficients = 0.

    Returns p-value.
    """
    hits = _hit_sequence(returns, var_forecasts, alpha) - alpha
    T = len(hits)
    if T <= lags + 2:
        return 1.0

    y = hits[lags:]
    var_reg = var_forecasts[lags:]
    Xmat = np.column_stack(
        [np.ones(T - lags)]
        + [hits[lags - j:-j] for j in range(1, lags + 1)]
        + [var_reg]
    )

    try:
        res = sm.OLS(y, Xmat).fit(cov_type="HC0")
        wald_stat = float(res.ess / res.mse_resid) * (lags + 2)
        return float(scipy.stats.chi2.sf(wald_stat, df=lags + 2))
    except Exception:
        return 1.0


def run_vqr_backtest(
    returns: np.ndarray,
    var_forecasts: np.ndarray,
    alpha: float,
) -> float:
    """VQR (Value-at-Risk Quantile Regression) backtest.

    Runs quantile regression of r_t on (1, VaR_t) at level alpha and tests
    H0: intercept = 0, slope = 1 via a Wald statistic.

    Returns p-value.
    """
    T = len(returns)
    W = sm.add_constant(var_forecasts, has_constant="add")

    try:
        res = sm.QuantReg(returns, W).fit(q=alpha, disp=False)
        # Test H0: intercept = 0, slope = 1
        # Use the QR score test approach: test if theta = (0, 1) is rejected
        theta_hat = res.params
        diff = theta_hat - np.array([0.0, 1.0])
        # Wald statistic (se from asymptotic covariance)
        try:
            cov = res.cov_params()
            wald = float(diff @ np.linalg.solve(cov, diff))
            return float(scipy.stats.chi2.sf(wald, df=2))
        except np.linalg.LinAlgError:
            return 1.0
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Full VaR analysis
# ---------------------------------------------------------------------------

def run_var_analysis(
    returns_eval: np.ndarray,
    var_forecasts: dict,
    alpha: float,
) -> dict:
    """Run score decomposition + backtests for all VaR models.

    Parameters
    ----------
    returns_eval : ndarray, shape (T_eval,)
    var_forecasts : dict
        Mapping model_name -> VaR forecast array, shape (T_eval,).
    alpha : float

    Returns
    -------
    dict with decomposition and backtest results.
    """
    Y = returns_eval
    cl_fn = functools.partial(check_loss, alpha=alpha)
    r_hat = compute_reference_quantile(Y, alpha)
    model_names = list(var_forecasts.keys())
    T = len(Y)

    decomps = {}
    recalibs = {}
    for name, X in var_forecasts.items():
        _, Xc = fit_recalibration_quantile(X, Y, alpha)
        decomps[name] = compute_decomposition(cl_fn, X, Y, Xc, r_hat)
        recalibs[name] = Xc

    # Backtests
    backtests = {}
    for name, X in var_forecasts.items():
        backtests[name] = {
            "p_uc": run_uc_backtest(Y, X, alpha),
            "p_cc": run_cc_backtest(Y, X, alpha),
            "p_dq": run_dq_backtest(Y, X, alpha),
            "p_dqx": run_dqx_backtest(Y, X, alpha),
            "p_vqr": run_vqr_backtest(Y, X, alpha),
        }

    # Pairwise inference
    pairwise = {}
    for i, n1 in enumerate(model_names):
        for n2 in model_names[i + 1:]:
            X1, Xc1 = var_forecasts[n1], recalibs[n1]
            X2, Xc2 = var_forecasts[n2], recalibs[n2]
            d1, d2 = decomps[n1], decomps[n2]

            Z = build_score_process_cl(X1, Y, Xc1, r_hat, X2, Xc2, alpha)
            Omega = estimate_hac_covariance(Z)
            Xi = build_xi_matrix()

            bp1_mcb = boundary_pvalue_quantile_mcb(T, d1["MCB"], X1, Xc1, Y, alpha)
            bp2_mcb = boundary_pvalue_quantile_mcb(T, d2["MCB"], X2, Xc2, Y, alpha)
            bp1_dsc = boundary_pvalue_quantile_dsc(T, d1["DSC"], Xc1, Y, alpha, r_hat)
            bp2_dsc = boundary_pvalue_quantile_dsc(T, d2["DSC"], Xc2, Y, alpha, r_hat)

            p_mcb = equal_mcb_test(T, d1["MCB"], d2["MCB"], Xi, Omega, bp1_mcb, bp2_mcb)
            p_dsc = equal_dsc_test(T, d1["DSC"], d2["DSC"], Xi, Omega, bp1_dsc, bp2_dsc)
            mcb_dsc = np.array([d1["MCB"], d1["DSC"], d2["MCB"], d2["DSC"]])
            p_dm = dm_test(T, Xi, Omega, mcb_dsc)

            pairwise[f"{n1}_vs_{n2}"] = {
                "p_mcb": float(p_mcb),
                "p_dsc": float(p_dsc),
                "p_dm": float(p_dm),
            }

    return {
        "alpha": alpha,
        "T": T,
        "r_hat": float(r_hat),
        "decompositions": decomps,
        "backtests": backtests,
        "pairwise": pairwise,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    logger.info("=== Experiment 5: VaR Forecast Evaluation ===")
    logger.info("NOTE: Using synthetic data; real analysis requires E-mini "
                "tick data from a commercial data provider.")

    T_train = 1500
    T_eval = 3617
    returns_train, returns_eval, dates_eval = generate_synthetic_returns_var(
        T_train, T_eval
    )
    returns_all = np.concatenate([returns_train, returns_eval])

    all_results = {}
    backtest_rows = []

    for alpha in ALPHA_LIST:
        logger.info("--- alpha = %.2f ---", alpha)

        # HS: use all data up to each evaluation point
        var_hs = forecast_var_hs(returns_all, alpha, window=HS_WINDOW)
        # Align HS with eval period (HS needs window of history)
        var_hs = var_hs[-T_eval:]

        logger.info("Fitting GARCH for VaR ...")
        var_garch = forecast_var_garch(returns_train, returns_eval, alpha)

        logger.info("Fitting GJR-GARCH for VaR ...")
        var_gjr = forecast_var_gjr(returns_train, returns_eval, alpha)

        var_forecasts = {"HS": var_hs, "GARCH": var_garch, "GJR": var_gjr}

        logger.info("Running VaR analysis ...")
        result = run_var_analysis(returns_eval, var_forecasts, alpha)
        all_results[f"alpha_{alpha}"] = result

        for name in var_forecasts:
            d = result["decompositions"][name]
            bt = result["backtests"][name]
            backtest_rows.append({
                "alpha": alpha,
                "model": name,
                "S": round(d["S"], 6),
                "MCB": round(d["MCB"], 6),
                "DSC": round(d["DSC"], 6),
                "UNC": round(d["UNC"], 6),
                "p_uc": round(bt["p_uc"], 4),
                "p_cc": round(bt["p_cc"], 4),
                "p_dq": round(bt["p_dq"], 4),
                "p_dqx": round(bt["p_dqx"], 4),
                "p_vqr": round(bt["p_vqr"], 4),
            })

        # MCB-DSC scatter per alpha
        mcb_vals = np.array([result["decompositions"][n]["MCB"] for n in var_forecasts])
        dsc_vals = np.array([result["decompositions"][n]["DSC"] for n in var_forecasts])
        score_vals = np.array([result["decompositions"][n]["S"] for n in var_forecasts])
        unc = result["decompositions"]["HS"]["UNC"]

        alpha_str = f"{int(alpha * 100)}pct"
        scatter_path = str(results_dir / f"exp5_mcb_dsc_{alpha_str}.png")
        plot_mcb_dsc_scatter(
            mcb_vals=mcb_vals,
            dsc_vals=dsc_vals,
            labels=list(var_forecasts.keys()),
            unc=unc,
            score_vals=score_vals,
            title=f"VaR Forecasts: MCB vs DSC (alpha={alpha})",
            save_path=scatter_path,
        )
        logger.info("Saved MCB–DSC scatter to %s", scatter_path)

    # Backtest table
    bt_df = pd.DataFrame(backtest_rows)
    bt_path = results_dir / "exp5_var_backtest_table.csv"
    bt_df.to_csv(bt_path, index=False)
    logger.info("Saved backtest table to %s", bt_path)
    logger.info("\n%s", bt_df.to_string(index=False))

    # Summary JSON
    def _make_serializable(obj):
        if isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, float)):
            return round(float(obj), 6)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        return obj

    summary = {
        "note": ("Using synthetic data; real analysis requires E-mini tick data "
                 "from a commercial data provider."),
        "T_train": T_train,
        "T_eval": T_eval,
        "results": _make_serializable(all_results),
    }
    # Strip non-serializable decomposition objects from nested dict
    for key in summary["results"]:
        for name in list(summary["results"][key].get("decompositions", {})):
            d = summary["results"][key]["decompositions"][name]
            summary["results"][key]["decompositions"][name] = {
                k: round(float(v), 6) for k, v in d.items()
            }

    summary_path = results_dir / "exp5_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary to %s", summary_path)
    logger.info("=== Experiment 5 complete ===")


if __name__ == "__main__":
    main()
