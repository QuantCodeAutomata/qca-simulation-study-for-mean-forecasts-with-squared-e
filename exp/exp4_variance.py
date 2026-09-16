"""Experiment 4: Score decomposition for financial variance forecasts.

NOTE: Real analysis requires high-frequency ESc1 E-mini S&P 500 futures data
from a commercial data provider.  This script uses *synthetic* daily returns
and realized-variance proxies to demonstrate the full framework.

Models compared
---------------
- HAR  : Heterogeneous AR model fitted to sqrt(RV) via OLS
- GARCH: GARCH(1,1) variance forecasts from the ``arch`` library

Saves
-----
- results/exp4_mcb_dsc_scatter.png
- results/exp4_decomposition_table.csv
- results/exp4_summary.json
"""

import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import scipy.optimize
import statsmodels.api as sm

from src.decomposition import squared_error, compute_decomposition
from src.recalibration import fit_recalibration_se, compute_reference_mean
from src.hac import estimate_hac_covariance
from src.inference import (
    build_score_process_se,
    build_xi_matrix,
    boundary_pvalue_se,
    equal_mcb_test,
    equal_dsc_test,
    dm_test,
)
from src.visualization import plot_mcb_dsc_scatter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_synthetic_returns(
    T_train: int = 2000,
    T_eval: int = 3617,
    seed: int = 42,
) -> tuple:
    """Generate synthetic daily returns and realized variance from GARCH(1,1).

    NOTE: Using synthetic data; real analysis requires ESc1 intraday data
    from a commercial data provider.

    The latent GARCH(1,1) variance follows:
        h_t = omega + alpha * r_{t-1}^2 + beta * h_{t-1}
    with omega=0.01, alpha=0.09, beta=0.90.

    Realized variance is approximated as the squared return scaled by a
    multiplier to mimic intraday summation.

    Returns
    -------
    (returns_eval, rv_eval, dates_eval) : tuple of ndarrays/DatetimeIndex
    """
    rng = np.random.default_rng(seed)
    T_total = T_train + T_eval + 100  # extra burn-in

    # GARCH(1,1) parameters
    omega, alpha_g, beta_g = 0.01, 0.09, 0.90
    h = np.zeros(T_total)
    r = np.zeros(T_total)
    h[0] = omega / (1.0 - alpha_g - beta_g)
    for t in range(1, T_total):
        h[t] = omega + alpha_g * r[t - 1] ** 2 + beta_g * h[t - 1]
        r[t] = np.sqrt(h[t]) * rng.standard_normal()

    # Discard burn-in; last T_train + T_eval observations
    r = r[-(T_train + T_eval):]
    h = h[-(T_train + T_eval):]

    # Realized variance proxy: use h (true variance) + noise to simulate RV
    noise = rng.gamma(shape=5.0, scale=1.0 / 5.0, size=len(r))
    rv = h * noise  # Gamma-distributed around h (mimics RV micro-structure noise)

    returns_eval = r[T_train:]
    rv_eval = rv[T_train:]

    dates_eval = pd.date_range("2003-01-02", periods=T_eval, freq="B")
    return returns_eval, rv_eval, dates_eval


# ---------------------------------------------------------------------------
# HAR model
# ---------------------------------------------------------------------------

def fit_har(rv_train: np.ndarray, rv_eval: np.ndarray) -> np.ndarray:
    """Fit HAR model on sqrt(RV) and generate one-step-ahead forecasts.

    HAR: sqrt(RV_t) = beta_0 + beta_1*sqrt(RV_{t-1})
                     + beta_2*mean(sqrt(RV_{t-5:t-1}))
                     + beta_3*mean(sqrt(RV_{t-22:t-1})) + e_t

    Estimation uses all available training data; evaluation uses rolling
    forecasts with fixed coefficients (re-fitting every period is too slow
    for this synthetic demonstration).

    Parameters
    ----------
    rv_train : ndarray, shape (T_train,)
    rv_eval  : ndarray, shape (T_eval,)

    Returns
    -------
    forecasts : ndarray, shape (T_eval,)
        One-step-ahead HAR variance forecasts (squared back from sqrt scale).
    """
    srV = np.sqrt(np.maximum(rv_train, 1e-12))

    # Build design matrix for training
    burn = 22
    y_reg = srV[burn:]
    lag1 = srV[burn - 1:-1]
    lag_w = np.array([srV[i - 5:i].mean() for i in range(burn, len(srV))])
    lag_m = np.array([srV[i - 22:i].mean() for i in range(burn, len(srV))])

    X_reg = sm.add_constant(np.column_stack([lag1, lag_w, lag_m]))
    coef = sm.OLS(y_reg, X_reg).fit().params

    # Rolling forecasts on evaluation set (using combined train+eval history)
    rv_all = np.concatenate([rv_train, rv_eval])
    srV_all = np.sqrt(np.maximum(rv_all, 1e-12))
    T_train = len(rv_train)
    T_eval = len(rv_eval)

    forecasts = np.empty(T_eval)
    for i in range(T_eval):
        t = T_train + i
        lag1_t = srV_all[t - 1]
        lag_w_t = srV_all[t - 5:t].mean()
        lag_m_t = srV_all[t - 22:t].mean()
        w = np.array([1.0, lag1_t, lag_w_t, lag_m_t])
        srV_hat = float(coef @ w)
        forecasts[i] = srV_hat ** 2  # back to variance scale

    return np.maximum(forecasts, 1e-12)


# ---------------------------------------------------------------------------
# GARCH(1,1)
# ---------------------------------------------------------------------------

def fit_garch_variance(
    returns_train: np.ndarray,
    returns_eval: np.ndarray,
) -> np.ndarray:
    """Fit GARCH(1,1) and generate one-step-ahead variance forecasts.

    Parameters
    ----------
    returns_train : ndarray, shape (T_train,)
    returns_eval  : ndarray, shape (T_eval,)

    Returns
    -------
    forecasts : ndarray, shape (T_eval,)
        One-step-ahead conditional variance forecasts.
    """
    from arch import arch_model

    returns_pct = returns_train * 100.0
    am = arch_model(returns_pct, vol="Garch", p=1, q=1, dist="normal", rescale=False)
    try:
        res = am.fit(disp="off", show_warning=False)
    except Exception as exc:
        logger.warning("GARCH fit failed: %s; using squared returns as fallback", exc)
        returns_all = np.concatenate([returns_train, returns_eval])
        return (returns_all[len(returns_train):] ** 2)

    # Generate one-step forecasts over evaluation period
    returns_all_pct = np.concatenate([returns_pct, returns_eval * 100.0])
    forecasts = np.empty(len(returns_eval))

    omega_g = res.params["omega"]
    alpha_g = res.params.get("alpha[1]", res.params.get("a[1]", 0.09))
    beta_g = res.params.get("beta[1]", res.params.get("b[1]", 0.90))
    h = float(res.conditional_volatility[-1] ** 2)

    for i in range(len(returns_eval)):
        t = len(returns_train) + i
        r_prev = returns_all_pct[t - 1]
        h = omega_g + alpha_g * r_prev ** 2 + beta_g * h
        forecasts[i] = h / (100.0 ** 2)  # convert back from pct scale

    return np.maximum(forecasts, 1e-12)


# ---------------------------------------------------------------------------
# QLIKE score and recalibration
# ---------------------------------------------------------------------------

def qlike_score(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """QLIKE scoring rule  S_QL(x, y) = y/x - log(y/x) - 1.

    Requires x > 0 and y > 0.

    Parameters
    ----------
    x : ndarray  Variance forecast.
    y : ndarray  Realized variance.

    Returns
    -------
    ndarray  Element-wise QLIKE scores.
    """
    ratio = y / x
    return ratio - np.log(ratio) - 1.0


def fit_recalibration_qlike(
    X: np.ndarray,
    Y: np.ndarray,
) -> tuple:
    """Score-consistent recalibration under QLIKE.

    Minimises  (1/T) sum [Y_t/(a + b*X_t) - log(Y_t/(a + b*X_t)) - 1]
    subject to  a + b*X_t > 0  for all t.

    Uses L-BFGS-B with explicit inequality constraints enforced via interior
    penalty when the naive unconstrained solution is infeasible.

    Parameters
    ----------
    X : ndarray, shape (T,)  Original variance forecast.
    Y : ndarray, shape (T,)  Realized variance (positive).

    Returns
    -------
    theta_hat : ndarray, shape (2,)  [a, b]
    Xc        : ndarray, shape (T,)  Recalibrated fitted values a + b*X
    """
    T = len(X)
    min_gap = 1e-8  # minimum fitted value

    def objective(theta):
        a, b = theta
        fit = a + b * X
        if np.any(fit <= 0):
            return 1e15
        return float(np.mean(qlike_score(fit, Y)))

    def gradient(theta):
        a, b = theta
        fit = a + b * X
        if np.any(fit <= 0):
            return np.array([1e10, 1e10])
        dS_dfit = -Y / fit ** 2 + 1.0 / fit  # d/d(fit) [y/fit - log(y/fit) - 1]
        ga = float(np.mean(dS_dfit))
        gb = float(np.mean(dS_dfit * X))
        return np.array([ga, gb])

    # Initial guess: a=0, b=1 (identity recalibration)
    theta0 = np.array([0.0, 1.0])

    # Constraints: a + b*X_t >= min_gap for all t
    constraints = [
        {"type": "ineq", "fun": lambda th: th[0] + th[1] * X - min_gap}
    ]

    result = scipy.optimize.minimize(
        objective,
        theta0,
        jac=gradient,
        method="SLSQP",
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 500},
    )

    theta_hat = result.x
    Xc = np.maximum(theta_hat[0] + theta_hat[1] * X, min_gap)
    return theta_hat, Xc


# ---------------------------------------------------------------------------
# Decomposition analysis
# ---------------------------------------------------------------------------

def run_variance_analysis(
    rv_eval: np.ndarray,
    forecasts: dict,
    score: str = "se",
) -> dict:
    """Run decomposition analysis for all model pairs under squared error.

    Parameters
    ----------
    rv_eval : ndarray, shape (T_eval,)
        Realized variance series.
    forecasts : dict
        Mapping model_name -> forecast array, shape (T_eval,).
    score : {'se', 'qlike'}
        Scoring rule to use.

    Returns
    -------
    dict with decomposition and inference results for each model pair.
    """
    if score == "se":
        score_fn = squared_error
        recalib_fn = fit_recalibration_se
        ref_fn = compute_reference_mean
    elif score == "qlike":
        score_fn = qlike_score
        recalib_fn = fit_recalibration_qlike
        ref_fn = lambda Y: float(np.exp(np.mean(np.log(Y))))  # geometric mean
    else:
        raise ValueError(f"Unknown score: {score}")

    Y = rv_eval
    r_hat = ref_fn(Y)
    model_names = list(forecasts.keys())

    decomps = {}
    recalibs = {}
    for name, X in forecasts.items():
        _, Xc = recalib_fn(X, Y)
        decomps[name] = compute_decomposition(score_fn, X, Y, Xc, r_hat)
        recalibs[name] = Xc

    results = {"score": score, "r_hat": float(r_hat), "decompositions": decomps}
    pairwise = {}

    for i, n1 in enumerate(model_names):
        for n2 in model_names[i + 1:]:
            X1, Xc1 = forecasts[n1], recalibs[n1]
            X2, Xc2 = forecasts[n2], recalibs[n2]
            d1, d2 = decomps[n1], decomps[n2]
            T = len(Y)

            Z = build_score_process_se(X1, Y, Xc1, r_hat, X2, Xc2)
            Omega = estimate_hac_covariance(Z)
            Xi = build_xi_matrix()

            bp1_mcb = boundary_pvalue_se(T, d1["MCB"], X1, Xc1, Y, r_hat, "mcb")
            bp2_mcb = boundary_pvalue_se(T, d2["MCB"], X2, Xc2, Y, r_hat, "mcb")
            bp1_dsc = boundary_pvalue_se(T, d1["DSC"], X1, Xc1, Y, r_hat, "dsc")
            bp2_dsc = boundary_pvalue_se(T, d2["DSC"], X2, Xc2, Y, r_hat, "dsc")

            p_mcb = equal_mcb_test(T, d1["MCB"], d2["MCB"], Xi, Omega, bp1_mcb, bp2_mcb)
            p_dsc = equal_dsc_test(T, d1["DSC"], d2["DSC"], Xi, Omega, bp1_dsc, bp2_dsc)
            mcb_dsc = np.array([d1["MCB"], d1["DSC"], d2["MCB"], d2["DSC"]])
            p_dm = dm_test(T, Xi, Omega, mcb_dsc)

            pairwise[f"{n1}_vs_{n2}"] = {
                "p_mcb": float(p_mcb),
                "p_dsc": float(p_dsc),
                "p_dm": float(p_dm),
            }

    results["pairwise"] = pairwise
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    logger.info("=== Experiment 4: Financial Variance Forecast Evaluation ===")
    logger.info("NOTE: Using synthetic data; real analysis requires ESc1 "
                "intraday data from commercial provider.")

    T_train = 2000
    T_eval = 3617
    returns_eval, rv_eval, dates_eval = generate_synthetic_returns(T_train, T_eval)

    # Reconstruct training returns for GARCH/HAR fitting
    _, rv_all, _ = generate_synthetic_returns(T_train + T_eval + 100, 1, seed=42)
    returns_all, _, _ = generate_synthetic_returns(T_train + T_eval + 100, 1, seed=42)

    rng = np.random.default_rng(42)
    T_total = T_train + T_eval + 100
    omega, alpha_g, beta_g = 0.01, 0.09, 0.90
    h_all = np.zeros(T_total)
    r_all = np.zeros(T_total)
    h_all[0] = omega / (1.0 - alpha_g - beta_g)
    for t in range(1, T_total):
        h_all[t] = omega + alpha_g * r_all[t - 1] ** 2 + beta_g * h_all[t - 1]
        r_all[t] = np.sqrt(h_all[t]) * rng.standard_normal()
    noise_all = rng.gamma(shape=5.0, scale=1.0 / 5.0, size=T_total)
    rv_all_series = h_all * noise_all
    rv_train = rv_all_series[-(T_train + T_eval):-T_eval]
    returns_train = r_all[-(T_train + T_eval):-T_eval]

    logger.info("Fitting HAR model ...")
    har_forecasts = fit_har(rv_train, rv_eval)

    logger.info("Fitting GARCH(1,1) model ...")
    garch_forecasts = fit_garch_variance(returns_train, returns_eval)

    forecasts = {"HAR": har_forecasts, "GARCH": garch_forecasts}

    logger.info("Running variance analysis (squared error) ...")
    results_se = run_variance_analysis(rv_eval, forecasts, score="se")

    # Build decomposition table
    rows = []
    for name, d in results_se["decompositions"].items():
        rows.append({
            "model": name,
            "S": round(d["S"], 6),
            "MCB": round(d["MCB"], 6),
            "DSC": round(d["DSC"], 6),
            "UNC": round(d["UNC"], 6),
        })
    decomp_df = pd.DataFrame(rows)
    csv_path = results_dir / "exp4_decomposition_table.csv"
    decomp_df.to_csv(csv_path, index=False)
    logger.info("Saved decomposition table to %s", csv_path)
    logger.info("\n%s", decomp_df.to_string(index=False))

    # MCB–DSC scatter
    mcb_vals = np.array([results_se["decompositions"][n]["MCB"] for n in forecasts])
    dsc_vals = np.array([results_se["decompositions"][n]["DSC"] for n in forecasts])
    score_vals = np.array([results_se["decompositions"][n]["S"] for n in forecasts])
    unc = results_se["decompositions"]["HAR"]["UNC"]

    scatter_path = str(results_dir / "exp4_mcb_dsc_scatter.png")
    plot_mcb_dsc_scatter(
        mcb_vals=mcb_vals,
        dsc_vals=dsc_vals,
        labels=list(forecasts.keys()),
        unc=unc,
        score_vals=score_vals,
        title="Variance Forecasts: MCB vs DSC (SE loss)",
        save_path=scatter_path,
    )
    logger.info("Saved MCB–DSC scatter to %s", scatter_path)

    # Summary JSON
    summary = {
        "note": ("Using synthetic data; real analysis requires ESc1 intraday "
                 "data from commercial provider."),
        "T_train": T_train,
        "T_eval": T_eval,
        "score": "squared_error",
        "decompositions": {
            name: {k: round(v, 6) for k, v in d.items()}
            for name, d in results_se["decompositions"].items()
        },
        "pairwise": {
            pair: {k: round(v, 6) for k, v in pv.items()}
            for pair, pv in results_se["pairwise"].items()
        },
    }
    summary_path = results_dir / "exp4_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary to %s", summary_path)
    logger.info("=== Experiment 4 complete ===")


if __name__ == "__main__":
    main()
