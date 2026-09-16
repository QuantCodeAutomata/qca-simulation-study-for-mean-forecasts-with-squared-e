"""Unit tests for src/decomposition.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.decomposition import (
    squared_error,
    check_loss,
    compute_decomposition,
    verify_decomposition_identity,
)
from src.recalibration import fit_recalibration_se, compute_reference_mean


def test_squared_error_perfect():
    x = np.array([1.0, 2.0, 3.0])
    assert np.allclose(squared_error(x, x), 0.0)


def test_squared_error_values():
    x = np.array([0.0, 1.0])
    y = np.array([1.0, 3.0])
    expected = np.array([1.0, 4.0])
    np.testing.assert_allclose(squared_error(x, y), expected)


def test_check_loss_nonnegative():
    # check loss is always >= 0 for (1{y<=x}-alpha)*(x-y)
    np.random.seed(42)
    x = np.random.randn(1000)
    y = np.random.randn(1000)
    cl = check_loss(x, y, alpha=0.1)
    assert np.all(cl >= -1e-10)


def test_check_loss_alpha_05_symmetric():
    # At alpha=0.5, check loss should be 0.5*|x-y|
    x = np.array([0.0, 2.0])
    y = np.array([1.0, 0.0])
    cl = check_loss(x, y, alpha=0.5)
    expected = 0.5 * np.abs(x - y)
    np.testing.assert_allclose(cl, expected, atol=1e-10)


def test_decomposition_identity():
    np.random.seed(42)
    Y = np.random.randn(200)
    X = Y + np.random.randn(200) * 0.5
    _, Xc = fit_recalibration_se(X, Y)
    r_hat = compute_reference_mean(Y)
    d = compute_decomposition(squared_error, X, Y, Xc, r_hat)
    assert abs(d["S"] - (d["MCB"] - d["DSC"] + d["UNC"])) < 1e-8


def test_decomposition_nonneg():
    np.random.seed(99)
    Y = np.random.randn(100)
    X = np.random.randn(100)
    _, Xc = fit_recalibration_se(X, Y)
    r_hat = compute_reference_mean(Y)
    d = compute_decomposition(squared_error, X, Y, Xc, r_hat)
    assert d["MCB"] >= -1e-8
    assert d["DSC"] >= -1e-8


def test_decomposition_perfect_calibration():
    # If X = Y, MCB should be 0 (perfect calibration)
    Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    X = Y.copy()
    _, Xc = fit_recalibration_se(X, Y)
    r_hat = compute_reference_mean(Y)
    d = compute_decomposition(squared_error, X, Y, Xc, r_hat)
    assert abs(d["MCB"]) < 1e-8


def test_unc_equals_sr():
    np.random.seed(7)
    Y = np.random.randn(100)
    X = np.random.randn(100)
    _, Xc = fit_recalibration_se(X, Y)
    r_hat = compute_reference_mean(Y)
    d = compute_decomposition(squared_error, X, Y, Xc, r_hat)
    assert abs(d["UNC"] - d["S_R"]) < 1e-10


def test_verify_decomposition_identity_raises():
    bad = {"S": 1.0, "MCB": 0.5, "DSC": 0.2, "UNC": 0.5}
    # S = 1.0, MCB - DSC + UNC = 0.5 - 0.2 + 0.5 = 0.8 -> should raise
    with pytest.raises(AssertionError):
        verify_decomposition_identity(bad, tol=1e-8)


def test_decomposition_keys():
    np.random.seed(0)
    Y = np.random.randn(50)
    X = np.random.randn(50)
    _, Xc = fit_recalibration_se(X, Y)
    r_hat = compute_reference_mean(Y)
    d = compute_decomposition(squared_error, X, Y, Xc, r_hat)
    for key in ("S", "S_C", "S_R", "MCB", "DSC", "UNC"):
        assert key in d


def test_check_loss_specific_values():
    # x > y, so 1{y<=x} = 1: loss = (1-alpha)*(x-y)
    x = np.array([2.0])
    y = np.array([1.0])
    alpha = 0.1
    expected = (1.0 - alpha) * (x - y)
    np.testing.assert_allclose(check_loss(x, y, alpha), expected, atol=1e-14)

    # x < y, so 1{y<=x} = 0: loss = (0-alpha)*(x-y) = alpha*(y-x)
    x2 = np.array([1.0])
    y2 = np.array([2.0])
    expected2 = alpha * (y2 - x2)
    np.testing.assert_allclose(check_loss(x2, y2, alpha), expected2, atol=1e-14)
