"""Experiment 2: Monte Carlo simulation for quantile forecasts (check loss).

For each (scenario, k, alpha) combination, runs N_MC replications and records
rejection rates for the equal-MCB, equal-DSC, and DM tests.  Results are
saved as 'results/exp2_rejection_rates.csv' and a rejection-rate plot as
'results/exp2_rejection_rates.png'.
"""

import sys
import logging
import functools
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from src.dgp import generate_ar1_predictors, generate_target, construct_quantile_forecasts
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
from src.scenarios import get_quantile_scenarios, population_components_cl
from src.visualization import plot_rejection_rates_quantile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
T = 500
N_MC = 5000
BETA = 0.25
NOMINAL_LEVEL = 0.10

K_GRID = np.arange(0.0, 1.05, 0.05)
ALPHA_LIST = [0.01, 0.05, 0.10, 0.25, 0.50]


def _run_one_replication(
    seed: int,
    scenario: dict,
    k: float,
    alpha: float,
) -> dict | None:
    """Run a single Monte Carlo replication.

    Returns
    -------
    dict with p-values, or None if an error occurred.
    """
    try:
        rng_seed_v = seed * 3
        rng_seed_y = seed * 3 + 1

        V = generate_ar1_predictors(T, beta=BETA, seed=rng_seed_v)

        # Scale gamma by k for Group B scenarios
        if scenario["group"] == "B":
            gamma = k * scenario["gamma"]
        else:
            gamma = scenario["gamma"]

        Y = generate_target(V, gamma, seed=rng_seed_y)

        delta_0, delta, xi_0, xi = scenario["get_params"](k, alpha)
        X1, X2 = construct_quantile_forecasts(V, delta_0, delta, xi_0, xi, alpha)

        theta1, Xc1 = fit_recalibration_quantile(X1, Y, alpha)
        theta2, Xc2 = fit_recalibration_quantile(X2, Y, alpha)
        r_hat = compute_reference_quantile(Y, alpha)

        cl_fn = functools.partial(check_loss, alpha=alpha)
        d1 = compute_decomposition(cl_fn, X1, Y, Xc1, r_hat)
        d2 = compute_decomposition(cl_fn, X2, Y, Xc2, r_hat)

        Z = build_score_process_cl(X1, Y, Xc1, r_hat, X2, Xc2, alpha)
        Omega = estimate_hac_covariance(Z)
        Xi = build_xi_matrix()

        bp1_mcb = boundary_pvalue_quantile_mcb(T, d1["MCB"], X1, Xc1, Y, alpha)
        bp2_mcb = boundary_pvalue_quantile_mcb(T, d2["MCB"], X2, Xc2, Y, alpha)
        bp1_dsc = boundary_pvalue_quantile_dsc(T, d1["DSC"], Xc1, Y, alpha, r_hat)
        bp2_dsc = boundary_pvalue_quantile_dsc(T, d2["DSC"], Xc2, Y, alpha, r_hat)

        p_mcb = equal_mcb_test(T, d1["MCB"], d2["MCB"], Xi, Omega, bp1_mcb, bp2_mcb)
        p_dsc = equal_dsc_test(T, d1["DSC"], d2["DSC"], Xi, Omega, bp1_dsc, bp2_dsc)
        mcb_dsc_hats = np.array([d1["MCB"], d1["DSC"], d2["MCB"], d2["DSC"]])
        p_dm = dm_test(T, Xi, Omega, mcb_dsc_hats)

        return {
            "rej_mcb": float(p_mcb < NOMINAL_LEVEL),
            "rej_dsc": float(p_dsc < NOMINAL_LEVEL),
            "rej_dm": float(p_dm < NOMINAL_LEVEL),
            "mcb1": d1["MCB"],
            "dsc1": d1["DSC"],
            "mcb2": d2["MCB"],
            "dsc2": d2["DSC"],
            "unc": d1["UNC"],
        }
    except Exception as exc:
        logger.debug("Replication seed=%d failed: %s", seed, exc)
        return None


def run_simulation() -> pd.DataFrame:
    """Run the full Monte Carlo simulation and return results DataFrame."""
    scenarios = get_quantile_scenarios()
    rows = []
    total_configs = len(scenarios) * len(K_GRID) * len(ALPHA_LIST)
    config_idx = 0

    for scenario in scenarios:
        sc_id = scenario["id"]
        for k in K_GRID:
            k = float(k)
            for alpha in ALPHA_LIST:
                config_idx += 1
                logger.info(
                    "[%d/%d] Scenario %s, k=%.2f, alpha=%.2f",
                    config_idx, total_configs, sc_id, k, alpha,
                )

                rej_mcb_list, rej_dsc_list, rej_dm_list = [], [], []
                mcb1_list, dsc1_list, mcb2_list, dsc2_list, unc_list = [], [], [], [], []
                n_errors = 0

                for rep in range(N_MC):
                    seed = config_idx * N_MC + rep
                    result = _run_one_replication(seed, scenario, k, alpha)
                    if result is None:
                        n_errors += 1
                        continue
                    rej_mcb_list.append(result["rej_mcb"])
                    rej_dsc_list.append(result["rej_dsc"])
                    rej_dm_list.append(result["rej_dm"])
                    mcb1_list.append(result["mcb1"])
                    dsc1_list.append(result["dsc1"])
                    mcb2_list.append(result["mcb2"])
                    dsc2_list.append(result["dsc2"])
                    unc_list.append(result["unc"])

                if n_errors > 0:
                    logger.warning("Scenario %s k=%.2f alpha=%.2f: %d/%d errors",
                                   sc_id, k, alpha, n_errors, N_MC)

                n_valid = len(rej_mcb_list)
                if n_valid == 0:
                    logger.error("All replications failed for Scenario %s k=%.2f alpha=%.2f",
                                 sc_id, k, alpha)
                    continue

                # Population components for comparison
                if scenario["group"] == "B":
                    gamma_pop = k * scenario["gamma"]
                else:
                    gamma_pop = scenario["gamma"]
                delta_0, delta, xi_0, xi = scenario["get_params"](k, alpha)
                pop = population_components_cl(delta_0, delta, gamma_pop, alpha)

                rows.append({
                    "scenario_id": sc_id,
                    "group": scenario["group"],
                    "k": k,
                    "alpha": alpha,
                    "rej_mcb": np.mean(rej_mcb_list),
                    "rej_dsc": np.mean(rej_dsc_list),
                    "rej_dm": np.mean(rej_dm_list),
                    "mc_mcb1": np.mean(mcb1_list),
                    "mc_dsc1": np.mean(dsc1_list),
                    "mc_mcb2": np.mean(mcb2_list),
                    "mc_dsc2": np.mean(dsc2_list),
                    "mc_unc": np.mean(unc_list),
                    "pop_mcb1": pop["MCB"],
                    "pop_dsc1": pop["DSC"],
                    "pop_unc": pop["UNC"],
                    "n_valid": n_valid,
                })

    return pd.DataFrame(rows)


def main() -> None:
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    logger.info("Starting Experiment 2: quantile simulation (N_MC=%d, T=%d)", N_MC, T)
    results_df = run_simulation()

    csv_path = results_dir / "exp2_rejection_rates.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info("Saved rejection rates to %s", csv_path)

    png_path = str(results_dir / "exp2_rejection_rates.png")
    plot_rejection_rates_quantile(results_df, K_GRID, ALPHA_LIST, save_path=png_path)
    logger.info("Saved rejection-rate plot to %s", png_path)

    # Summary: compare MC averages with population formulas
    logger.info("=== Population vs MC average components (alpha=0.10) ===")
    sub = results_df[results_df["alpha"] == 0.10]
    for sc_id in sub["scenario_id"].unique():
        sdf = sub[sub["scenario_id"] == sc_id]
        for _, row in sdf.iterrows():
            logger.info(
                "Sc=%s k=%.2f | pop MCB=%.4f mc MCB1=%.4f | pop DSC=%.4f mc DSC1=%.4f",
                row["scenario_id"], row["k"],
                row["pop_mcb1"], row["mc_mcb1"],
                row["pop_dsc1"], row["mc_dsc1"],
            )


if __name__ == "__main__":
    main()
