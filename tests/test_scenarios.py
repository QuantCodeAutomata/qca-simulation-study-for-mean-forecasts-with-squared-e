"""Unit tests for src/scenarios.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.scenarios import (
    get_mean_scenarios,
    get_quantile_scenarios,
    population_mcb_se,
    population_dsc_se,
    population_unc_se,
    population_components_cl,
)


def test_mean_scenarios_count():
    assert len(get_mean_scenarios()) == 12


def test_mean_scenario_structure():
    for s in get_mean_scenarios():
        assert "id" in s and "group" in s and "gamma" in s and "get_params" in s
        assert s["group"] in ("A", "B")
        assert s["gamma"].shape == (4,)


def test_mean_scenario_params_shape():
    for s in get_mean_scenarios():
        delta_0, delta, xi_0, xi = s["get_params"](0.1)
        assert isinstance(delta_0, (float, int, np.floating))
        assert len(delta) == 4
        assert len(xi) == 4


def test_quantile_scenarios_count():
    assert len(get_quantile_scenarios()) == 4


def test_quantile_scenario_structure():
    for s in get_quantile_scenarios():
        assert "id" in s and "group" in s and "gamma" in s and "get_params" in s
        assert s["group"] in ("A", "B")
        assert s["gamma"].shape == (4,)


def test_quantile_scenario_params_shape():
    for s in get_quantile_scenarios():
        delta_0, delta, xi_0, xi = s["get_params"](0.1, 0.05)
        assert isinstance(delta_0, (float, int, np.floating))
        assert len(delta) == 4
        assert len(xi) == 4


def test_population_mcb_se_zero_delta():
    # When delta = 0, MCB = delta_0^2
    delta_0 = 0.5
    mcb = population_mcb_se(delta_0, np.zeros(4), np.array([0.25] * 4))
    assert abs(mcb - delta_0 ** 2) < 1e-10


def test_population_mcb_se_perfect():
    # When delta = gamma and delta_0 = 0, MCB = 0
    gamma = np.array([0.25, 0.25, 0.25, 0.0])
    mcb = population_mcb_se(0.0, gamma, gamma)
    assert abs(mcb) < 1e-10


def test_population_unc_se():
    gamma = np.array([0.0, 0.25, 0.25, 0.0])
    varsigma = 16 / 15
    expected = varsigma * np.dot(gamma, gamma) + 1
    result = population_unc_se(gamma)
    assert abs(result - expected) < 1e-10


def test_population_dsc_nonneg():
    delta = np.array([0.25, 0.0, 0.25, 0.0])
    gamma = np.array([0.25, 0.25, 0.0, 0.0])
    dsc = population_dsc_se(delta, gamma)
    assert dsc >= 0


def test_population_cl_components_valid():
    delta_0 = 0.0
    delta = np.array([0.25, 0.0, 0.25, 0.0])
    gamma = np.array([0.25, 0.25, 0.25, 0.0])
    result = population_components_cl(delta_0, delta, gamma, alpha=0.1)
    assert "MCB" in result and "DSC" in result and "UNC" in result
    assert result["DSC"] >= -1e-10
    assert result["UNC"] >= 0


def test_population_dsc_se_zero_delta():
    # When delta = 0, DSC = 0
    gamma = np.array([0.25, 0.25, 0.25, 0.0])
    dsc = population_dsc_se(np.zeros(4), gamma)
    assert abs(dsc) < 1e-10


def test_population_mcb_se_zero_intercept_zero_slope():
    # When delta = gamma and delta_0 = 0: MCB should be 0
    gamma = np.array([0.1, 0.2, 0.3, 0.4])
    mcb = population_mcb_se(0.0, gamma, gamma)
    assert abs(mcb) < 1e-10


def test_population_unc_se_zero_gamma():
    # When gamma = 0, UNC = 1 (only the noise variance)
    gamma = np.zeros(4)
    result = population_unc_se(gamma)
    assert abs(result - 1.0) < 1e-10


def test_population_cl_unc_formula():
    # UNC_bar = sigma_Y * phi(z_alpha)
    import scipy.stats
    gamma = np.array([0.25, 0.25, 0.25, 0.0])
    alpha = 0.1
    varsigma = 16 / 15
    sigma_Y = np.sqrt(varsigma * np.dot(gamma, gamma) + 1.0)
    z_alpha = scipy.stats.norm.ppf(alpha)
    expected_unc = sigma_Y * scipy.stats.norm.pdf(z_alpha)
    result = population_components_cl(0.0, gamma, gamma, alpha=alpha)
    assert abs(result["UNC"] - expected_unc) < 1e-10


def test_mean_scenario_ids_unique():
    ids = [s["id"] for s in get_mean_scenarios()]
    assert len(ids) == len(set(ids))


def test_quantile_scenario_ids_unique():
    ids = [s["id"] for s in get_quantile_scenarios()]
    assert len(ids) == len(set(ids))
