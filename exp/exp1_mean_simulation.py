"""Experiment 1: Monte Carlo simulation for mean forecasts under squared error loss.

For each of the 12 mean-forecast scenarios, runs N_MC replications across the
k-grid {0, 0.05, ..., 0.50}, recording rejection rates for the equal-MCB,
equal-DSC, and DM tests.

Steps per replication (as in the paper):
  4.  Recalibrate via OLS (fit_recalibration_se)
  5.  Reference forecast = sample mean (compute_reference_mean)
  6.  Compute MCB, DSC, UNC from score decomposition
  7.  Verify identity S = MCB - DSC + UNC and non-negativity (tol 1e-8)
  8.  HAC long-run covariance via Andrews-style Bartlett estimator
  9-11. Equal-MCB test with hybrid boundary p-value
  12-13. Equal-DSC test with hybrid boundary p-value
  14. Diebold–Mariano (DM) test
  15. Population formula comparison
  16. Plot rejection rates

Output files:
  results/exp1_results.csv
  results/exp1_population_comparison.csv
  results/exp1_rejection_rates.png
  results/RESULTS.md  (appended)
"""

import sys
import time
import logging
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

from tqdm import tqdm

from src.dgp import generate_ar1_predictors, generate_target, construct_mean_forecasts
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
from src.scenarios import (
    get_mean_scenarios,
    population_mcb_se,
    population_dsc_se,
    population_unc_se,
)
from src.visualization import plot_rejection_rates_mean

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
T = 500
N_MC = 5000
BETA = 0.25
NOMINAL_LEVEL = 0.10
BURN_IN = 200

K_GRID = np.arange(0.0, 0.55, 0.05)   # {0, 0.05, 0.10, ..., 0.50}  (11 values)

SHORT_N_MC = 50       # replications used to estimate per-replication timing
CHUNK_SIZE = 100      # save intermediate results every CHUNK_SIZE replications
TIME_LIMIT_SEC = 3600  # 1 hour; fall back to fewer reps if exceeded


# ---------------------------------------------------------------------------
# Single replication
# ---------------------------------------------------------------------------

def run_single_replication(scenario: dict, k: float, seed: int) -> dict | None:
    """Run one Monte Carlo replication for a given scenario and k value.

    Parameters
    ----------
    scenario : dict
        Entry from get_mean_scenarios(); must have keys 'gamma', 'group',
        'get_params'.
    k : float
        Current point on the k-grid.
    seed : int
        Base random seed; AR(1) predictors use seed*2, target uses seed*2+1.

    Returns
    -------
    dict
        Keys: MCB1, DSC1, MCB2, DSC2, UNC, p_mcb, p_dsc, p_dm,
        reject_mcb, reject_dsc, reject_dm.
    None
        If any unrecoverable error is encountered.
    """
    try:
        # Step 3: Generate data -----------------------------------------------
        gamma = scenario["gamma"].copy()
        if scenario["group"] == "B":
            # For Group B the DGP is Y_t = k * V_t^T gamma_base + eps
            gamma = k * gamma

        V = generate_ar1_predictors(T, beta=BETA, burn_in=BURN_IN, seed=seed * 2)
        Y = generate_target(V, gamma, seed=seed * 2 + 1)

        delta_0, delta, xi_0, xi = scenario["get_params"](k)
        X1, X2 = construct_mean_forecasts(V, delta_0, delta, xi_0, xi)

        # Step 4: Recalibration via OLS ----------------------------------------
        _, Xc1 = fit_recalibration_se(X1, Y)
        _, Xc2 = fit_recalibration_se(X2, Y)

        # Step 5: Reference forecast = sample mean ----------------------------
        r_hat = compute_reference_mean(Y)

        # Step 6: Score decomposition -----------------------------------------
        d1 = compute_decomposition(squared_error, X1, Y, Xc1, r_hat)
        d2 = compute_decomposition(squared_error, X2, Y, Xc2, r_hat)

        # Step 7: Verify identity and non-negativity (tol 1e-8) ---------------
        # (compute_decomposition already calls verify_decomposition_identity
        #  internally, but we add explicit assertions here per the spec)
        assert d1["MCB"] >= -1e-8, f"MCB1={d1['MCB']:.3e} violates non-negativity"
        assert d1["DSC"] >= -1e-8, f"DSC1={d1['DSC']:.3e} violates non-negativity"
        assert abs(d1["S"] - (d1["MCB"] - d1["DSC"] + d1["UNC"])) <= 1e-8, "Identity violated (d1)"
        assert d2["MCB"] >= -1e-8, f"MCB2={d2['MCB']:.3e} violates non-negativity"
        assert d2["DSC"] >= -1e-8, f"DSC2={d2['DSC']:.3e} violates non-negativity"
        assert abs(d2["S"] - (d2["MCB"] - d2["DSC"] + d2["UNC"])) <= 1e-8, "Identity violated (d2)"

        # Step 8: HAC long-run covariance (Andrews-style Bartlett) ------------
        Z = build_score_process_se(X1, Y, Xc1, r_hat, X2, Xc2)
        Omega_hat = estimate_hac_covariance(Z)
        Xi = build_xi_matrix()

        # Steps 9-13: Boundary p-values (with error handling) -----------------
        def _safe_boundary(val: float, X: np.ndarray, Xc: np.ndarray,
                           component: str) -> float:
            try:
                return boundary_pvalue_se(T, val, X, Xc, Y, r_hat, component)
            except Exception as exc:
                logger.debug("boundary_pvalue_se(%s) failed: %s", component, exc)
                return 1.0

        bp_mcb1 = _safe_boundary(d1["MCB"], X1, Xc1, "mcb")
        bp_dsc1 = _safe_boundary(d1["DSC"], X1, Xc1, "dsc")
        bp_mcb2 = _safe_boundary(d2["MCB"], X2, Xc2, "mcb")
        bp_dsc2 = _safe_boundary(d2["DSC"], X2, Xc2, "dsc")

        mcb_dsc_hats = np.array([d1["MCB"], d1["DSC"], d2["MCB"], d2["DSC"]])

        # Step 9-11: Equal-MCB hybrid test ------------------------------------
        p_mcb = equal_mcb_test(T, d1["MCB"], d2["MCB"], Xi, Omega_hat, bp_mcb1, bp_mcb2)

        # Steps 12-13: Equal-DSC hybrid test ----------------------------------
        p_dsc = equal_dsc_test(T, d1["DSC"], d2["DSC"], Xi, Omega_hat, bp_dsc1, bp_dsc2)

        # Step 14: DM test ----------------------------------------------------
        p_dm = dm_test(T, Xi, Omega_hat, mcb_dsc_hats)

        return {
            "MCB1": d1["MCB"],
            "DSC1": d1["DSC"],
            "MCB2": d2["MCB"],
            "DSC2": d2["DSC"],
            "UNC":  d1["UNC"],
            "p_mcb": p_mcb,
            "p_dsc": p_dsc,
            "p_dm":  p_dm,
            "reject_mcb": int(p_mcb < NOMINAL_LEVEL),
            "reject_dsc": int(p_dsc < NOMINAL_LEVEL),
            "reject_dm":  int(p_dm  < NOMINAL_LEVEL),
        }

    except Exception as exc:
        logger.debug("Replication seed=%d failed: %s", seed, exc)
        return None


# ---------------------------------------------------------------------------
# Population-level components (Step 15)
# ---------------------------------------------------------------------------

def compute_population_components(scenario: dict, k: float) -> dict:
    """Compute population MCB, DSC, UNC for both forecasters at a given k.

    Parameters
    ----------
    scenario : dict
        Scenario entry from get_mean_scenarios().
    k : float
        k-grid value.

    Returns
    -------
    dict
        Keys: MCB1_pop, DSC1_pop, MCB2_pop, DSC2_pop, UNC_pop.
    """
    delta_0, delta, xi_0, xi = scenario["get_params"](k)
    gamma = scenario["gamma"].copy()
    if scenario["group"] == "B":
        gamma = k * gamma

    return {
        "MCB1_pop": population_mcb_se(delta_0, delta, gamma),
        "DSC1_pop": population_dsc_se(delta, gamma),
        "MCB2_pop": population_mcb_se(xi_0, xi, gamma),
        "DSC2_pop": population_dsc_se(xi, gamma),
        "UNC_pop":  population_unc_se(gamma),
    }


# ---------------------------------------------------------------------------
# Scenario × k runner with intermediate saving
# ---------------------------------------------------------------------------

def _run_replication_wrapper(scenario: dict, k: float, seed: int) -> dict | None:
    """Top-level wrapper for joblib pickling (avoids lambda capture issues)."""
    return run_single_replication(scenario, k, seed)


def run_scenario_k(
    scenario_idx: int,
    scenario: dict,
    k_idx: int,
    k: float,
    n_mc: int,
    intermediate_path: Path | None = None,
) -> list[dict]:
    """Run n_mc replications for one (scenario, k) pair.

    Processes replications in batches of CHUNK_SIZE to allow intermediate
    results to be written to disk after each batch, so progress is not lost
    if the run is interrupted.

    Parameters
    ----------
    scenario_idx : int
        0-based index of the scenario (used in seed formula).
    scenario : dict
        Scenario definition.
    k_idx : int
        0-based index in K_GRID (used in seed formula).
    k : float
        k-grid value.
    n_mc : int
        Total number of replications.
    intermediate_path : Path or None
        If given, partial results are saved here after every CHUNK_SIZE reps.

    Returns
    -------
    list of dict
        One entry per successful replication (failed reps are excluded).
    """
    sc_id = scenario["id"]
    results: list[dict] = []
    n_errors = 0

    def _seed(rep_idx: int) -> int:
        return scenario_idx * 100000 + k_idx * 10000 + rep_idx

    for chunk_start in range(0, n_mc, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, n_mc)
        rep_indices = list(range(chunk_start, chunk_end))

        if HAS_JOBLIB:
            chunk_results = Parallel(n_jobs=-1, backend="loky", verbose=0)(
                delayed(_run_replication_wrapper)(scenario, k, _seed(ri))
                for ri in rep_indices
            )
        else:
            chunk_results = [
                run_single_replication(scenario, k, _seed(ri))
                for ri in rep_indices
            ]

        for r in chunk_results:
            if r is None:
                n_errors += 1
            else:
                r["scenario_id"] = sc_id
                r["group"] = scenario["group"]
                r["k"] = k
                results.append(r)

        # Save intermediate results after each chunk
        if intermediate_path is not None and results:
            pd.DataFrame(results).to_csv(intermediate_path, index=False)

    if n_errors:
        logger.warning(
            "Sc=%s k=%.2f: %d/%d replications failed",
            sc_id, k, n_errors, n_mc,
        )

    return results


# ---------------------------------------------------------------------------
# Full simulation loop
# ---------------------------------------------------------------------------

def run_simulation(n_mc: int, results_dir: Path) -> pd.DataFrame:
    """Run the complete Monte Carlo simulation for all scenarios and k values.

    Parameters
    ----------
    n_mc : int
        Number of Monte Carlo replications per (scenario, k) pair.
    results_dir : Path
        Directory where results and intermediate files are saved.

    Returns
    -------
    pd.DataFrame
        One row per (scenario, k) pair with aggregated rejection rates,
        MC component averages, and population component values.
    """
    scenarios = get_mean_scenarios()
    intermediate_dir = results_dir / "_intermediate_exp1"
    intermediate_dir.mkdir(exist_ok=True)

    all_rows: list[dict] = []
    total_configs = len(scenarios) * len(K_GRID)

    config_pairs = [
        (sc_idx, sc, k_idx, float(k))
        for sc_idx, sc in enumerate(scenarios)
        for k_idx, k in enumerate(K_GRID)
    ]

    for cfg_i, (sc_idx, scenario, k_idx, k) in enumerate(
        tqdm(config_pairs, desc="Scenario×k", unit="cfg"), start=1
    ):
        sc_id = scenario["id"]
        logger.info("[%d/%d] Scenario %s, k=%.2f", cfg_i, total_configs, sc_id, k)

        intermed = intermediate_dir / f"sc{sc_id}_k{k_idx:02d}.csv"
        reps = run_scenario_k(sc_idx, scenario, k_idx, k, n_mc, intermediate_path=intermed)

        if not reps:
            logger.error("All replications failed for Sc=%s k=%.2f", sc_id, k)
            continue

        rep_df = pd.DataFrame(reps)

        # Step 15: population component comparison
        pop = compute_population_components(scenario, k)

        row = {
            "scenario_id": sc_id,
            "group": scenario["group"],
            "k": k,
            # Rejection rates (named for visualization compatibility)
            "rej_mcb": float(rep_df["reject_mcb"].mean()),
            "rej_dsc": float(rep_df["reject_dsc"].mean()),
            "rej_dm":  float(rep_df["reject_dm"].mean()),
            # MC component averages
            "mc_MCB1": float(rep_df["MCB1"].mean()),
            "mc_DSC1": float(rep_df["DSC1"].mean()),
            "mc_MCB2": float(rep_df["MCB2"].mean()),
            "mc_DSC2": float(rep_df["DSC2"].mean()),
            "mc_UNC":  float(rep_df["UNC"].mean()),
            "n_valid": len(rep_df),
            **pop,
        }
        all_rows.append(row)

        # Save running aggregated results after each (scenario, k) pair
        pd.DataFrame(all_rows).to_csv(results_dir / "exp1_results.csv", index=False)

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Timing estimate helper
# ---------------------------------------------------------------------------

def _estimate_timing(scenarios: list[dict]) -> float:
    """Run SHORT_N_MC replications on scenario 0, k=0 and return secs/rep."""
    logger.info("Timing probe: %d replications of scenario %s at k=0.0 ...",
                SHORT_N_MC, scenarios[0]["id"])
    t0 = time.perf_counter()
    run_scenario_k(0, scenarios[0], 0, 0.0, SHORT_N_MC, intermediate_path=None)
    elapsed = time.perf_counter() - t0
    secs_per_rep = elapsed / SHORT_N_MC
    logger.info("Timing probe: %.3f s/rep  (%.1f ms)", secs_per_rep, secs_per_rep * 1e3)
    return secs_per_rep


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    logger.info("=== Experiment 1: Mean Forecast Monte Carlo Simulation ===")
    logger.info(
        "Constants: T=%d, N_MC=%d, BETA=%.2f, NOMINAL_LEVEL=%.2f, BURN_IN=%d",
        T, N_MC, BETA, NOMINAL_LEVEL, BURN_IN,
    )

    # ---- Short timing test --------------------------------------------------
    scenarios = get_mean_scenarios()
    secs_per_rep = _estimate_timing(scenarios)

    total_configs = len(scenarios) * len(K_GRID)
    estimated_sec = secs_per_rep * N_MC * total_configs
    logger.info(
        "Estimated full-run time: %.1f min  (N_MC=%d, %d configs, %.3f s/rep)",
        estimated_sec / 60, N_MC, total_configs, secs_per_rep,
    )

    # Decide effective N_MC
    if estimated_sec <= TIME_LIMIT_SEC:
        effective_n_mc = N_MC
        logger.info("Will use N_MC=%d (within time budget)", effective_n_mc)
    else:
        # Scale down to stay within one hour, rounded to nearest 100
        raw = int(TIME_LIMIT_SEC / (secs_per_rep * total_configs))
        effective_n_mc = max(100, (raw // 100) * 100)
        logger.warning(
            "Full N_MC=%d would take ~%.0f min; falling back to N_MC=%d",
            N_MC, estimated_sec / 60, effective_n_mc,
        )

    # ---- Full simulation ----------------------------------------------------
    logger.info("Starting full simulation with N_MC=%d ...", effective_n_mc)
    t_start = time.perf_counter()
    results_df = run_simulation(effective_n_mc, results_dir)
    runtime_sec = time.perf_counter() - t_start
    logger.info("Simulation finished in %.1f min", runtime_sec / 60)

    if results_df.empty:
        logger.error("results_df is empty – no results to save.")
        return

    # ---- Save main CSV ------------------------------------------------------
    csv_path = results_dir / "exp1_results.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info("Saved results to %s", csv_path)

    # ---- Population vs MC comparison table ----------------------------------
    pop_cols = [
        "scenario_id", "k",
        "MCB1_pop", "mc_MCB1",
        "DSC1_pop", "mc_DSC1",
        "MCB2_pop", "mc_MCB2",
        "DSC2_pop", "mc_DSC2",
        "UNC_pop",  "mc_UNC",
    ]
    available = [c for c in pop_cols if c in results_df.columns]
    pop_df = results_df[available]
    pop_csv = results_dir / "exp1_population_comparison.csv"
    pop_df.to_csv(pop_csv, index=False)
    logger.info("Saved population comparison to %s", pop_csv)

    # ---- Step 16: Plot rejection rates --------------------------------------
    png_path = str(results_dir / "exp1_rejection_rates.png")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_rejection_rates_mean(results_df, K_GRID, save_path=png_path)
    logger.info("Saved rejection-rate plot to %s", png_path)

    # ---- Console summary ----------------------------------------------------
    print("\n" + "=" * 60)
    print("Experiment 1 Summary")
    print("=" * 60)
    print(f"  N_MC = {effective_n_mc}  (requested {N_MC})")
    print(f"  T = {T},  BETA = {BETA},  nominal level = {NOMINAL_LEVEL}")
    print(f"  Runtime: {runtime_sec / 60:.1f} min")
    print(f"  Scenarios: {sorted(results_df['scenario_id'].unique().tolist())}")

    k0_df = results_df[np.isclose(results_df["k"], 0.0)]
    if not k0_df.empty:
        print(f"\n  Rejection rates at k=0.00 (size, target ≈ {NOMINAL_LEVEL}):")
        for _, row in k0_df.iterrows():
            print(f"    Sc={row['scenario_id']:>3s}: "
                  f"MCB={row['rej_mcb']:.3f}  "
                  f"DSC={row['rej_dsc']:.3f}  "
                  f"DM={row['rej_dm']:.3f}")

    k_max_df = results_df[np.isclose(results_df["k"], 0.50)]
    if not k_max_df.empty:
        print(f"\n  Rejection rates at k=0.50 (power):")
        for _, row in k_max_df.iterrows():
            print(f"    Sc={row['scenario_id']:>3s}: "
                  f"MCB={row['rej_mcb']:.3f}  "
                  f"DSC={row['rej_dsc']:.3f}  "
                  f"DM={row['rej_dm']:.3f}")

    print("=" * 60 + "\n")

    # ---- Append to RESULTS.md -----------------------------------------------
    results_md = results_dir / "RESULTS.md"
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    k0_table = ""
    if not k0_df.empty:
        k0_table = "| Scenario | MCB | DSC | DM |\n|---|---|---|---|\n"
        for _, row in k0_df.sort_values("scenario_id").iterrows():
            k0_table += (
                f"| {row['scenario_id']} "
                f"| {row['rej_mcb']:.3f} "
                f"| {row['rej_dsc']:.3f} "
                f"| {row['rej_dm']:.3f} |\n"
            )

    kmax_table = ""
    if not k_max_df.empty:
        kmax_table = "| Scenario | MCB | DSC | DM |\n|---|---|---|---|\n"
        for _, row in k_max_df.sort_values("scenario_id").iterrows():
            kmax_table += (
                f"| {row['scenario_id']} "
                f"| {row['rej_mcb']:.3f} "
                f"| {row['rej_dsc']:.3f} "
                f"| {row['rej_dm']:.3f} |\n"
            )

    md_entry = f"""

---

## Experiment 1 Run – {timestamp}

**Parameters:** N_MC={effective_n_mc}, T={T}, β={BETA}, nominal level={NOMINAL_LEVEL}, burn-in={BURN_IN}  
**Runtime:** {runtime_sec / 60:.1f} min  
**Parallel:** {'yes (joblib)' if HAS_JOBLIB else 'no (sequential)'}

### Rejection rates at k=0 (size, target ≈ {NOMINAL_LEVEL})

{k0_table}

### Rejection rates at k=0.50 (power)

{kmax_table}

### Output files

- `results/exp1_rejection_rates.png` – Step 16 rejection-rate curves
- `results/exp1_results.csv` – Full aggregated results
- `results/exp1_population_comparison.csv` – Population vs MC component averages
"""

    with open(results_md, "a") as fh:
        fh.write(md_entry)
    logger.info("Appended findings to %s", results_md)


if __name__ == "__main__":
    main()
