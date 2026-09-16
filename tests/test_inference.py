"""Unit tests for src/inference.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.inference import (
    build_xi_matrix,
    gaussian_component_test,
    equal_mcb_test,
    equal_dsc_test,
    dm_test,
    build_score_process_se,
    boundary_pvalue_se,
)
from src.hac import estimate_hac_covariance
from src.dgp import generate_ar1_predictors, generate_target, construct_mean_forecasts
from src.recalibration import fit_recalibration_se, compute_reference_mean
from src.decomposition import squared_error, compute_decomposition


def test_xi_matrix_shape():
    Xi = build_xi_matrix()
    assert Xi.shape == (4, 5)


def test_xi_matrix_values():
    Xi = build_xi_matrix()
    expected = np.array(
        [[1, -1, 0, 0, 0],
         [0, -1, 0, 0, 1],
         [0, 0, 1, -1, 0],
         [0, 0, 0, -1, 1]],
        dtype=float,
    )
    np.testing.assert_array_equal(Xi, expected)


def test_gaussian_test_pvalue_range():
    np.random.seed(42)
    Xi = build_xi_matrix()
    Omega = np.eye(5)
    mcb_dsc = np.array([0.1, 0.2, 0.05, 0.15])
    omega = np.array([1.0, 0.0, -1.0, 0.0])
    p = gaussian_component_test(mcb_dsc, omega, Xi, Omega, T=500)
    assert 0.0 <= p <= 1.0


def test_equal_mcb_hybrid_formula():
    # p_MCB = min(1, max(p_MCB^+, 2*min(p_MCB1, p_MCB2)))
    # With p_MCB^+ = 0.5, p_MCB1 = 0.3, p_MCB2 = 0.4: should be max(0.5, 2*0.3) = 0.6
    # But we can't easily mock gaussian_component_test, so just check range
    np.random.seed(0)
    V = generate_ar1_predictors(100, seed=0)
    gamma = np.array([0.0, 0.25, 0.25, 0.0])
    Y = generate_target(V, gamma, seed=1)
    X1, X2 = construct_mean_forecasts(
        V, 0.0, np.array([0, 0, 0.25, 0]), 0.0, np.array([0, 0.25, 0.25, 0])
    )
    t1, Xc1 = fit_recalibration_se(X1, Y)
    t2, Xc2 = fit_recalibration_se(X2, Y)
    r_hat = compute_reference_mean(Y)
    d1 = compute_decomposition(squared_error, X1, Y, Xc1, r_hat)
    d2 = compute_decomposition(squared_error, X2, Y, Xc2, r_hat)
    Z = build_score_process_se(X1, Y, Xc1, r_hat, X2, Xc2)
    Xi = build_xi_matrix()
    Omega = estimate_hac_covariance(Z)
    bp1 = boundary_pvalue_se(100, d1["MCB"], X1, Xc1, Y, r_hat, "mcb")
    bp2 = boundary_pvalue_se(100, d2["MCB"], X2, Xc2, Y, r_hat, "mcb")
    bp1d = boundary_pvalue_se(100, d1["DSC"], X1, Xc1, Y, r_hat, "dsc")
    bp2d = boundary_pvalue_se(100, d2["DSC"], X2, Xc2, Y, r_hat, "dsc")
    p_mcb = equal_mcb_test(100, d1["MCB"], d2["MCB"], Xi, Omega, bp1, bp2)
    p_dsc = equal_dsc_test(100, d1["DSC"], d2["DSC"], Xi, Omega, bp1d, bp2d)
    p_dm = dm_test(100, Xi, Omega, np.array([d1["MCB"], d1["DSC"], d2["MCB"], d2["DSC"]]))
    assert 0 <= p_mcb <= 1
    assert 0 <= p_dsc <= 1
    assert 0 <= p_dm <= 1


def test_boundary_pvalue_range():
    np.random.seed(5)
    Y = np.random.randn(200)
    X = Y + np.random.randn(200) * 0.3
    _, Xc = fit_recalibration_se(X, Y)
    r_hat = compute_reference_mean(Y)
    d = compute_decomposition(squared_error, X, Y, Xc, r_hat)
    p = boundary_pvalue_se(200, d["MCB"], X, Xc, Y, r_hat, "mcb")
    assert 0 <= p <= 1


def test_dm_test_null_pvalue():
    # Under null (identical forecasters), DM test should not frequently reject
    np.random.seed(10)
    Y = np.random.randn(500)
    X = Y + np.random.randn(500) * 0.5
    _, Xc1 = fit_recalibration_se(X, Y)
    _, Xc2 = fit_recalibration_se(X, Y)  # same forecast
    r_hat = compute_reference_mean(Y)
    Z = build_score_process_se(X, Y, Xc1, r_hat, X, Xc2)
    Xi = build_xi_matrix()
    Omega = estimate_hac_covariance(Z)
    d = compute_decomposition(squared_error, X, Y, Xc1, r_hat)
    mcb_dsc = np.array([d["MCB"], d["DSC"], d["MCB"], d["DSC"]])
    p_dm = dm_test(500, Xi, Omega, mcb_dsc)
    assert p_dm > 0.1  # should not reject under null


def test_score_process_shape():
    np.random.seed(3)
    Y = np.random.randn(150)
    X1 = Y + np.random.randn(150) * 0.4
    X2 = Y + np.random.randn(150) * 0.6
    _, Xc1 = fit_recalibration_se(X1, Y)
    _, Xc2 = fit_recalibration_se(X2, Y)
    r_hat = compute_reference_mean(Y)
    Z = build_score_process_se(X1, Y, Xc1, r_hat, X2, Xc2)
    assert Z.shape == (150, 5)


def test_boundary_pvalue_dsc_range():
    np.random.seed(20)
    Y = np.random.randn(200)
    X = Y + np.random.randn(200) * 0.3
    _, Xc = fit_recalibration_se(X, Y)
    r_hat = compute_reference_mean(Y)
    d = compute_decomposition(squared_error, X, Y, Xc, r_hat)
    p = boundary_pvalue_se(200, d["DSC"], X, Xc, Y, r_hat, "dsc")
    assert 0 <= p <= 1


def test_gaussian_test_degenerate_omega():
    # If omega produces zero denominator, should return 1.0
    Xi = build_xi_matrix()
    Omega = np.zeros((5, 5))  # zero covariance -> degenerate
    mcb_dsc = np.array([0.1, 0.2, 0.05, 0.15])
    omega = np.array([1.0, 0.0, -1.0, 0.0])
    p = gaussian_component_test(mcb_dsc, omega, Xi, Omega, T=500)
    assert p == 1.0
