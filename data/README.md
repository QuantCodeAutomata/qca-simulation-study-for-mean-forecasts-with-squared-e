# Data Directory

This directory contains data loading scripts and cached datasets.

## Data Sources

### Experiment 3: US Inflation Survey Forecasts
- SPF (Survey of Professional Forecasters): Available from Philadelphia Fed
- Michigan Survey of Consumers: Available from FRED
- CPI data: From FRED or BLS (2023:Q3 vintage)
- Evaluation sample: 1982:Q3 to 2020:Q4 (T=154 quarters)

### Experiments 4 & 5: S&P 500 E-mini Futures
- Intraday prices: Refinitiv ESc1 (requires data provider access)
- Sample: 2000-01-01 to 2022-01-01
- 5-minute frequency for realized variance computation
- Training: 2000-01-01 to 2007-12-31
- Evaluation: 2008-01-01 to 2022-01-01 (T=3617)

## Data Loading

See `exp/exp3_inflation.py` for data loading functions for inflation data.
See `exp/exp4_variance.py` and `exp/exp5_var_forecasts.py` for financial data loading.

## Notes

- Raw market data files (CSV, Parquet) are gitignored and should not be committed
- Use environment variable `MASSIVE_API_KEY` for financial data access
