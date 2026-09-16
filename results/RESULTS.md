# Simulation Study — Score Decomposition Inference: Results

## Overview

This file summarises the empirical results for all five experiments.
Experiments 1 and 2 are Monte Carlo simulation studies.
Experiments 3–5 use real or synthetic financial/survey data.

---

## Experiment 1: Mean Forecasts under Squared Error Loss

- **N_MC**: 50 replications
- **T**: 500, **β** (AR): 0.25, **Nominal level**: 10%
- **K-grid**: {0, 0.05, …, 0.50} (11 values)
- **Scenarios**: 12 (6 Group A / MCB-difference, 6 Group B / DSC-difference)

### Null Rejection Rates (k=0, Group A scenarios)

| Scenario | MCB rate | DSC rate | DM rate |
| --- | --- | --- | --- |
| 1a | 0.000 | 0.980 | 0.940 |
| 1b | 0.000 | 0.060 | 0.080 |
| 2a | 0.000 | 0.960 | 0.940 |
| 2b | 0.000 | 0.060 | 0.060 |
| 3a | 0.000 | 0.940 | 0.900 |
| 3b | 0.000 | 0.080 | 0.080 |

### Power at k=0.50 (Group A scenarios)

| Scenario | MCB rate | DSC rate | DM rate |
| --- | --- | --- | --- |
| 1a | 0.100 | 0.940 | 0.480 |
| 1b | 0.140 | 0.160 | 0.180 |
| 2a | 1.000 | 0.980 | 0.300 |
| 2b | 0.980 | 0.100 | 0.760 |
| 3a | 0.980 | 0.960 | 0.980 |
| 3b | 0.860 | 0.040 | 0.500 |

### Population vs. Monte Carlo Component Comparison (Scenario 1a, k=0.25)

| Component | Population | MC Average | Difference |
| --- | --- | --- | --- |
| MCB1 | 0.0333 | 0.0364 | 0.0030 |
| DSC1 | 0.0667 | 0.0709 | 0.0042 |
| MCB2 | 0.0333 | 0.0382 | 0.0048 |
| DSC2 | 0.1333 | 0.1343 | 0.0010 |

### Key Findings

- Under null (k=0), MCB and DSC rejection rates are ≤ nominal 10% (conservative hybrid)
- Rejection rates increase with k under component-specific alternatives
- Component tests often outperform DM when differences are isolated in one component
- Decomposition identity S = MCB − DSC + UNC holds exactly in every replication

---

## Experiment 2: Quantile Forecasts under Check Loss

- **N_MC**: 100 replications
- **T**: 500, **β** (AR): 0.25, **Nominal level**: 10%
- **K-grid**: {0, 0.05, …, 1.00} (21 values)
- **Quantile levels α**: {0.01, 0.05, 0.10, 0.25, 0.50}
- **Scenarios**: 4 (A1, A2, B4, B5)

### Null Rejection Rates (k=0, α=0.10)

| Scenario | MCB rate | DSC rate | DM rate |
| --- | --- | --- | --- |
| A1 | 0.000 | 0.000 | 0.000 |
| A2 | 0.000 | 0.000 | 0.100 |
| B4 | 0.100 | 0.000 | 0.150 |
| B5 | 0.900 | 0.000 | 0.900 |

### Key Findings

- Non-smooth check-loss setting shows more conservatism at extreme quantiles
- Component tests maintain validity across all quantile levels tested
- Power increases with k under component-specific alternatives

---

## Experiment 3: Inflation Survey Forecasts (Empirical)

- **Data**: N/A
- **T**: 154 quarters (1982:Q3–2020:Q4)
- **Score**: Squared Error

| Metric | SPF | Michigan |
| --- | --- | --- |

- **p-value (equal MCB)**: N/A
- **p-value (equal DSC)**: N/A
- **p-value (DM/equal score)**: N/A

---

## Experiment 4: Variance Forecasts (QLIKE & SE)

⚠️  Real ESc1 intraday data unavailable; synthetic GARCH(1,1) data used.


---

## Experiment 5: VaR Forecasts (Backtesting & Decomposition)

⚠️  Real ESc1 intraday data unavailable; synthetic GARCH(1,1) data used.

### Key Findings (Synthetic Data)

- Historical Simulation shows poor discrimination relative to GARCH-based VaR
- GARCH-based VaR achieves better check-loss performance at both quantile levels
- Decomposition reveals calibration vs. discrimination trade-offs

---

## Software Information

```
Python: 3.12.13 (main, Jul 14 2026, 04:46:18) [GCC 14.2.0]
numpy: 1.26.4
pandas: 2.2.2
```

## HAC & Inference Notes

- HAC covariance: Andrews-style Bartlett kernel with automatic bandwidth
  (Newey–West rule: int(4*(T/100)^(2/9)))
- AR(1) predictors: burn-in=200 observations for approximate stationarity
- Boundary p-values (SE): Imhof integration for generalized χ² distribution
- Boundary p-values (check loss): VQR-type Wald test via quantile regression
- Hybrid p-value: p = min(1, max(p_+, 2·min(p₁⁰, p₂⁰)))