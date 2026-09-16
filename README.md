# Score Decomposition Inference for Forecasts

A comprehensive simulation and empirical study implementing score decomposition-based statistical inference for forecast comparison under squared error and check loss, as described in the quantitative finance literature.

## Repository Structure

```
repo/
├── data/              # Data loading scripts and cached data
├── src/               # Core implementation modules
│   ├── dgp.py         # Data generating process
│   ├── decomposition.py  # Score decomposition (MCB, DSC, UNC)
│   ├── recalibration.py  # Score-consistent recalibration
│   ├── hac.py         # HAC covariance estimation
│   ├── imhof.py       # Imhof generalized chi-square integration
│   ├── inference.py   # Statistical tests
│   ├── scenarios.py   # Simulation scenario parameters
│   └── visualization.py  # Plotting utilities
├── exp/               # Experiment scripts
│   ├── exp1_mean_simulation.py    # Exp 1: Mean forecasts, squared error
│   ├── exp2_quantile_simulation.py # Exp 2: Quantile forecasts, check loss
│   ├── exp3_inflation.py          # Exp 3: US inflation survey forecasts
│   ├── exp4_variance.py           # Exp 4: Variance forecasts
│   └── exp5_var_forecasts.py      # Exp 5: VaR forecasts and backtesting
├── tests/             # pytest tests
├── results/           # Output artifacts
│   └── RESULTS.md     # Summary of findings
└── AGENTS.md          # Repository knowledge base
```

## Setup

```bash
pip install -r requirements.txt
```

## Running Experiments

```bash
# Run simulation studies
python exp/exp1_mean_simulation.py
python exp/exp2_quantile_simulation.py

# Run empirical applications (requires data)
python exp/exp3_inflation.py
python exp/exp4_variance.py
python exp/exp5_var_forecasts.py
```

## Running Tests

```bash
pytest tests/ -v
```

## Key Methodology

The score decomposition identity:
**S = MCB - DSC + UNC**

where:
- **S**: Average score (loss)
- **MCB**: Miscalibration (always ≥ 0, measures calibration error)
- **DSC**: Discrimination (always ≥ 0, measures information content)
- **UNC**: Uncertainty (constant across forecasters)

## Environment Variables

- `MASSIVE_API_KEY`: API key for financial data (required for exp_4 and exp_5)
