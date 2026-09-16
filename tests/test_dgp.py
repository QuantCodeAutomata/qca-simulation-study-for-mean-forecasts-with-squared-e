"""Unit tests for src/dgp.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.dgp import (
    generate_ar1_predictors,
    generate_target,
    construct_mean_forecasts,
    construct_quantile_forecasts,
)


def test_ar1_shape():
    V = generate_ar1_predictors(T=100, seed=42)
    assert V.shape == (100, 4)


def test_ar1_stationarity():
    # Check approximate stationarity: variance per column ~ 1/(1-0.25^2) = 16/15
    V = generate_ar1_predictors(T=10000, beta=0.25, seed=1)
    expected_var = 1 / (1 - 0.25 ** 2)
    for col in range(4):
        assert abs(np.var(V[:, col]) - expected_var) < 0.1, (
            f"Column {col} variance off"
        )


def test_ar1_independence():
    # Columns should be approximately uncorrelated
    V = generate_ar1_predictors(T=10000, seed=2)
    corr = np.corrcoef(V.T)
    off_diag = np.abs(corr - np.eye(4)).max()
    assert off_diag < 0.05


def test_target_shape():
    V = generate_ar1_predictors(T=50, seed=3)
    gamma = np.array([0.25, 0.25, 0.25, 0.0])
    Y = generate_target(V, gamma, seed=4)
    assert Y.shape == (50,)


def test_mean_forecasts_shape():
    V = generate_ar1_predictors(T=100, seed=5)
    X1, X2 = construct_mean_forecasts(
        V, 0.0, np.ones(4) * 0.25, 0.0, np.ones(4) * 0.25
    )
    assert X1.shape == (100,)
    assert X2.shape == (100,)


def test_quantile_forecasts_z_alpha_shift():
    # Quantile forecasts should shift by z_alpha relative to mean forecasts
    from scipy.stats import norm

    V = generate_ar1_predictors(T=100, seed=6)
    delta_0, delta = 0.0, np.ones(4) * 0.25
    xi_0, xi = 0.0, np.ones(4) * 0.25

    X1_q, X2_q = construct_quantile_forecasts(V, delta_0, delta, xi_0, xi, alpha=0.05)
    X1_m, X2_m = construct_mean_forecasts(V, delta_0, delta, xi_0, xi)

    z_alpha = norm.ppf(0.05)
    np.testing.assert_allclose(X1_q - X1_m, z_alpha, atol=1e-10)
    np.testing.assert_allclose(X2_q - X2_m, z_alpha, atol=1e-10)


def test_seed_reproducibility():
    V1 = generate_ar1_predictors(T=100, seed=42)
    V2 = generate_ar1_predictors(T=100, seed=42)
    np.testing.assert_array_equal(V1, V2)


def test_target_seed_reproducibility():
    V = generate_ar1_predictors(T=100, seed=0)
    gamma = np.array([0.1, 0.2, 0.3, 0.4])
    Y1 = generate_target(V, gamma, seed=7)
    Y2 = generate_target(V, gamma, seed=7)
    np.testing.assert_array_equal(Y1, Y2)


def test_mean_forecasts_intercept():
    # With all-zero slope vectors, X1 = delta_0, X2 = xi_0
    V = generate_ar1_predictors(T=50, seed=8)
    X1, X2 = construct_mean_forecasts(V, 3.0, np.zeros(4), -1.5, np.zeros(4))
    np.testing.assert_allclose(X1, 3.0, atol=1e-14)
    np.testing.assert_allclose(X2, -1.5, atol=1e-14)


def test_ar1_burn_in_independence():
    # Different burn-in values should give same T returned observations (shape only)
    V1 = generate_ar1_predictors(T=50, burn_in=100, seed=1)
    V2 = generate_ar1_predictors(T=50, burn_in=500, seed=1)
    assert V1.shape == V2.shape == (50, 4)
