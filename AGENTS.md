# Repository Knowledge Base

## Project Summary
This repository implements score decomposition-based inference for forecast comparison as described in a quantitative finance paper. The decomposition S = MCB - DSC + UNC allows evaluation of forecast miscalibration and discrimination separately.

## Key Mathematical Concepts

### Score Decomposition (Squared Error)
- S_SE(x,y) = (x-y)^2
- MCB_hat_i = S_hat_i - S_hat_i^C (always >= 0)
- DSC_hat_i = S_hat^R - S_hat_i^C (always >= 0)
- UNC_hat = S_hat^R
- Identity: S_hat_i = MCB_hat_i - DSC_hat_i + UNC_hat

### Score Decomposition (Check Loss)
- S_CL,alpha(x,y) = (1{y<=x} - alpha)*(x-y)
- Same decomposition structure applies

### Recalibration
- Squared error: OLS regression Y on (1, X) using statsmodels
- Check loss: Quantile regression at level alpha using statsmodels QuantReg
- Reference forecast r_hat: sample mean (SE) or empirical quantile (CL)

### Statistical Tests
- Equal-MCB: Hybrid IU/UI with Gaussian branch + boundary p-values via Imhof
- Equal-DSC: Same structure
- DM test: Standard HAC-robust score difference test
- Non-smooth (check loss): VQR-type Wald tests for boundary p-values

### HAC Covariance
- Bartlett (Newey-West) kernel with Andrews (1991) automatic bandwidth
- Implemented in src/hac.py
- Approximates R's sandwich::vcovHAC

### Imhof Integration
- Implemented in src/imhof.py using scipy.integrate.quad
- Used for generalized chi-square p-values in boundary tests
- **CRITICAL**: Correct formula uses arctan(lambda_j * u) NOT arctan(2*lambda_j*u)
  The integration variable u in Imhof = 2*t where t is the CF argument; after
  substitution the chi^2 CF becomes (1-i*u*lambda_j)^{-1/2}, giving
  theta(u) = (1/2)*sum arctan(lambda_j*u) - q*u/2 and
  rho(u) = prod(1+lambda_j^2*u^2)^{1/4}
- k=1 case: uses exact formula P(lambda*chi^2_1 > q) = chi2.sf(q/lambda, df=1)
  to avoid conditionally convergent integral (amplitude ~ u^{-3/2})
- k>=2 case: adaptive Gauss-Kronrod on [0, u_max] with asymptotic breakpoints
  at zero-crossings spaced by half-period = 2*pi/|q_stat|

## Design Choices (Documented)
- AR(1) initialization: burn-in of 200 steps (not stationary initialization)
- Empirical quantile convention: np.quantile with 'midpoint' interpolation
- HAC bandwidth: Andrews (1991) AR(1) plug-in rule with T^(1/3) fallback

## File Organization
- `src/`: Core modules (imported by exp/ scripts)
- `exp/`: Experiment scripts (self-contained runners)
  - `exp2_quantile_simulation.py`: Monte Carlo for quantile forecasts (check loss). Constants: T=500, N_MC=5000, K_GRID=np.arange(0,1.05,0.05), alpha_list=[0.01,0.05,0.10,0.25,0.50]. Group B scenarios scale gamma by k.
  - `exp3_inflation.py`: US CPI inflation forecast comparison (SPF vs Michigan). Uses FRED data with synthetic fallback. Aligned window 1982Q3–2020Q4.
  - `exp4_variance.py`: Variance forecast evaluation (HAR vs GARCH). Synthetic data only; note about ESc1 requirement. Includes QLIKE score and constrained recalibration.
  - `exp5_var_forecasts.py`: VaR forecast evaluation (HS, GARCH, GJR). Synthetic data only; includes UC/CC/DQ/DQX/VQR backtests at alpha=0.01 and 0.05.
- `tests/`: pytest unit and integration tests (47 tests, all passing)
  - `test_dgp.py`, `test_decomposition.py`, `test_inference.py`, `test_scenarios.py`
- `results/`: Output figures, tables, and RESULTS.md summary

## Running Instructions
- All exp/ and tests/ files use `sys.path.insert(0, str(Path(__file__).parent.parent))` for portability
- Simulation experiments (exp1, exp2): Fully self-contained, no external data needed
- Empirical experiments (exp3-5): Use synthetic data by default; FRED download attempted for exp3 (pandas_datareader)
- Run tests: `cd /workspace/project && python -m pytest tests/ -v`
- Exp scripts use `matplotlib.use("Agg")` for headless execution

## Key Implementation Notes
- exp2: check_loss needs `functools.partial(check_loss, alpha=alpha)` as score_fn for compute_decomposition
- exp2: Group B quantile scenarios (B4, B5) scale gamma by k: `gamma = k * scenario['gamma']`
- exp4/exp5: GARCH parameters accessed as `res.params['alpha[1]']`, `res.params['beta[1]']`
- exp5: VaR forecasts are the alpha-quantile of returns (negative for small alpha)
- QLIKE recalibration in exp4: constrained optimization via scipy.optimize.minimize SLSQP
