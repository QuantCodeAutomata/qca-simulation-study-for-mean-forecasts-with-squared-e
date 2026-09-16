"""Simulation scenario parameters module.

Defines the 12 mean-forecast (exp_1) and 4 quantile-forecast (exp_2)
scenarios from Tables 3 and 4 of the paper, together with functions for
computing population-level MCB, DSC, and UNC values.

All population formulas assume:
  - V_t  iid AR(1) with unit innovations, stationary variance 1/(1-beta^2)
  - Because the formulas are expressed in terms of the population gram matrix
    of V_t, we use  varsigma = E[V_t^T delta * V_t^T delta] / (delta^T delta)
    = 1 / (1 - beta^2) = 16/15  for beta = 0.25 (the default AR coefficient).
"""

import numpy as np
import scipy.optimize
import scipy.stats


# ---------------------------------------------------------------------------
# Population-level components under squared error
# ---------------------------------------------------------------------------

def population_mcb_se(
    delta_0: float,
    delta: np.ndarray,
    gamma: np.ndarray,
    varsigma: float = 16 / 15,
) -> float:
    """Population MCB for a mean forecast under squared error.

    With  Y_t = V_t^T gamma + eps  and  X_{1t} = delta_0 + V_t^T delta:

        MCB_bar = delta_0^2 + varsigma * (delta^T delta - delta^T gamma)^2 / (delta^T delta)

    When  delta^T delta = 0  (degenerate forecast), MCB_bar = delta_0^2.

    Parameters
    ----------
    delta_0 : float
        Forecast intercept.
    delta : np.ndarray, shape (4,)
        Forecast slope vector.
    gamma : np.ndarray, shape (4,)
        True DGP coefficient vector.
    varsigma : float
        Stationary second moment scaling; default 16/15 for beta=0.25.

    Returns
    -------
    float
    """
    dd = float(np.dot(delta, delta))
    if dd == 0.0:
        return delta_0 ** 2
    dg = float(np.dot(delta, gamma))
    return delta_0 ** 2 + varsigma * (dd - dg) ** 2 / dd


def population_dsc_se(
    delta: np.ndarray,
    gamma: np.ndarray,
    varsigma: float = 16 / 15,
) -> float:
    """Population DSC for a mean forecast under squared error.

        DSC_bar = varsigma * (delta^T gamma)^2 / (delta^T delta)

    When  delta^T delta = 0, DSC_bar = 0.

    Parameters
    ----------
    delta : np.ndarray, shape (4,)
    gamma : np.ndarray, shape (4,)
    varsigma : float

    Returns
    -------
    float
    """
    dd = float(np.dot(delta, delta))
    if dd == 0.0:
        return 0.0
    dg = float(np.dot(delta, gamma))
    return varsigma * dg ** 2 / dd


def population_unc_se(
    gamma: np.ndarray,
    varsigma: float = 16 / 15,
) -> float:
    """Population UNC for mean forecasts under squared error.

        UNC_bar = varsigma * gamma^T gamma + 1

    (variance of Y_t = V_t^T gamma + eps, eps iid N(0,1)).

    Parameters
    ----------
    gamma : np.ndarray, shape (4,)
    varsigma : float

    Returns
    -------
    float
    """
    return varsigma * float(np.dot(gamma, gamma)) + 1.0


def population_components_cl(
    delta_0: float,
    delta: np.ndarray,
    gamma: np.ndarray,
    alpha: float,
    varsigma: float = 16 / 15,
) -> dict:
    """Population MCB, DSC, UNC for a quantile forecast under check loss.

    Derived in Proposition C.1 of the paper.  Notation:

        sigma_Y^2       = varsigma * gamma^T gamma + 1
        sigma_{1|X}^2   = sigma_Y^2 - varsigma * (delta^T gamma)^2 / (delta^T delta)
                          [= sigma_Y^2  when  delta^T delta = 0]
        z_alpha         = Phi^{-1}(alpha)
        m_1             = -(delta_0 + z_alpha)
        s_1^2           = 1 + varsigma * (gamma^T gamma + delta^T delta - 2 * delta^T gamma)
        kappa_1         = m_1 / s_1

        MCB_bar = m_1 * (Phi(kappa_1) + alpha - 1) + s_1 * phi(kappa_1) - sigma_{1|X} * phi(z_alpha)
        DSC_bar = phi(z_alpha) * (sigma_Y - sigma_{1|X})
        UNC_bar = sigma_Y * phi(z_alpha)

    Parameters
    ----------
    delta_0 : float
    delta   : np.ndarray, shape (4,)
    gamma   : np.ndarray, shape (4,)
    alpha   : float
    varsigma : float

    Returns
    -------
    dict with keys 'MCB', 'DSC', 'UNC'
    """
    z_alpha = scipy.stats.norm.ppf(alpha)
    phi = scipy.stats.norm.pdf
    Phi = scipy.stats.norm.cdf

    gg = float(np.dot(gamma, gamma))
    dd = float(np.dot(delta, delta))
    dg = float(np.dot(delta, gamma))

    sigma_Y2 = varsigma * gg + 1.0
    sigma_Y = np.sqrt(sigma_Y2)

    if dd == 0.0:
        sigma_1X2 = sigma_Y2
    else:
        sigma_1X2 = sigma_Y2 - varsigma * dg ** 2 / dd
    sigma_1X = np.sqrt(max(sigma_1X2, 0.0))

    m1 = -(delta_0 + z_alpha)
    s1_sq = 1.0 + varsigma * (gg + dd - 2.0 * dg)
    s1 = np.sqrt(max(s1_sq, 0.0))

    kappa1 = m1 / s1 if s1 > 0 else (np.sign(m1) * np.inf)

    MCB = (
        m1 * (Phi(kappa1) + alpha - 1.0)
        + s1 * phi(kappa1)
        - sigma_1X * phi(z_alpha)
    )
    DSC = phi(z_alpha) * (sigma_Y - sigma_1X)
    UNC = sigma_Y * phi(z_alpha)

    return {"MCB": float(MCB), "DSC": float(DSC), "UNC": float(UNC)}


# ---------------------------------------------------------------------------
# Mean forecast scenarios (exp_1)
# ---------------------------------------------------------------------------

def get_mean_scenarios() -> list[dict]:
    """Return the 12 mean-forecast scenario definitions (Table 3).

    Each entry is a dict:
        'id'         : scenario label string (e.g. '1a')
        'group'      : 'A' (MCB-difference) or 'B' (DSC-difference)
        'gamma'      : np.ndarray shape (4,) – DGP coefficient
        'get_params' : callable(k) -> (delta_0, delta, xi_0, xi)
                       where delta, xi are np.ndarray shape (4,)

    Notes on parametrization
    ------------------------
    X1 = delta_0 + V^T delta,  X2 = xi_0 + V^T xi.
    V_t has four components; all delta/xi vectors are in R^4.
    Intercepts delta_0, xi_0 default to 0 unless listed below.

    Group A – scenarios that differ in MCB
    Group B – scenarios that differ in DSC
    """
    s = np.sqrt

    scenarios = [
        # ---- Group A: MCB-difference scenarios ----
        {
            "id": "1a",
            "group": "A",
            "gamma": np.array([0.0, 1 / 4, 1 / 4, 0.0]),
            "get_params": lambda k: (
                0.0,
                np.array([0.0, 0.0, 1 / 4 + k / s(2), 0.0]),
                0.0,
                np.array([0.0, 1 / 4 + k / 2, 1 / 4 + k / 2, 0.0]),
            ),
        },
        {
            "id": "1b",
            "group": "A",
            "gamma": np.array([1 / 4, 1 / 4, 1 / 4, 0.0]),
            "get_params": lambda k: (
                0.0,
                np.array([1 / 4 + k / 2, 0.0, 1 / 4 + k / 2, 0.0]),
                0.0,
                np.array([0.0, 1 / 4 + k / 2, 1 / 4 + k / 2, 0.0]),
            ),
        },
        {
            "id": "2a",
            "group": "A",
            "gamma": np.array([0.0, 1 / 4, 1 / 4, 0.0]),
            "get_params": lambda k: (
                0.0,
                np.array([0.0, 0.0, 1 / 4, 0.0]),
                0.0,
                np.array([0.0, 1 / 4 + k / 2, 1 / 4 + k / 2, 0.0]),
            ),
        },
        {
            "id": "2b",
            "group": "A",
            "gamma": np.array([1 / 4, 1 / 4, 1 / 4, 0.0]),
            "get_params": lambda k: (
                0.0,
                np.array([1 / 4, 0.0, 1 / 4, 0.0]),
                0.0,
                np.array([0.0, 1 / 4 + k / 2, 1 / 4 + k / 2, 0.0]),
            ),
        },
        {
            "id": "3a",
            "group": "A",
            "gamma": np.array([0.0, 1 / 4, 1 / 4, 0.0]),
            "get_params": lambda k: (
                0.0,
                np.array([0.0, 0.0, 1 / 4 + k, 0.0]),
                0.0,
                np.array([0.0, 1 / 4 + k / 2, 1 / 4 + k / 2, 0.0]),
            ),
        },
        {
            "id": "3b",
            "group": "A",
            "gamma": np.array([1 / 4, 1 / 4, 1 / 4, 0.0]),
            "get_params": lambda k: (
                0.0,
                np.array([1 / 4 + k / s(2), 0.0, 1 / 4 + k / s(2), 0.0]),
                0.0,
                np.array([0.0, 1 / 4 + k / 2, 1 / 4 + k / 2, 0.0]),
            ),
        },
        # ---- Group B: DSC-difference scenarios ----
        {
            "id": "4a",
            "group": "B",
            "gamma": np.array([0.0, 0.0, 0.0, 1.0]),   # scaled by k via get_params
            "get_params": lambda k: (
                0.0,
                np.array([1 / (4 * s(2)) + k / 2, 0.0, 0.0, 1 / (4 * s(2)) + k / 2]),
                0.0,
                np.array([0.0, 1 / 4 + k / 2, 0.0, 1 / 4 + k / 2]),
            ),
        },
        {
            "id": "4b",
            "group": "B",
            "gamma": np.array([0.0, 0.0, 0.0, 1.0]),
            "get_params": lambda k: (
                0.0,
                np.array([1 / 4 + k / 2, 0.0, 0.0, 1 / 4 + k / 2]),
                0.0,
                np.array([0.0, 1 / 4 + k / 2, 0.0, 1 / 4 + k / 2]),
            ),
        },
        {
            "id": "5a",
            "group": "B",
            "gamma": np.array([0.0, 0.0, 0.0, 1.0]),
            "get_params": lambda k: (
                0.0,
                np.array([0.0, 0.0, 1 / 4, 0.0]),
                0.0,
                np.array([0.0, 1 / 4 + k / 2, 0.0, 1 / 4 + k / 2]),
            ),
        },
        {
            "id": "5b",
            "group": "B",
            "gamma": np.array([0.0, 0.0, 0.0, 1.0]),
            "get_params": lambda k: (
                0.0,
                np.array([1 / 4, 0.0, 1 / 4, 0.0]),
                0.0,
                np.array([0.0, 1 / 4 + k / 2, 0.0, 1 / 4 + k / 2]),
            ),
        },
        {
            "id": "6a",
            "group": "B",
            "gamma": np.array([0.0, 0.0, 0.0, 1.0]),
            "get_params": lambda k: (
                0.0,
                np.array([0.0, 0.0, 0.0, 1 / 4 + k]),
                0.0,
                np.array([0.0, 1 / 4 + k / 2, 0.0, 1 / 4 + k / 2]),
            ),
        },
        {
            "id": "6b",
            "group": "B",
            "gamma": np.array([0.0, 0.0, 0.0, 1.0]),
            "get_params": lambda k: (
                1 / s(15),
                np.array([0.0, 0.0, 0.0, 1 / 4 + k]),
                0.0,
                np.array([0.0, 1 / 4 + k / 2, 0.0, 1 / 4 + k / 2]),
            ),
        },
    ]

    # For Group B scenarios 4a–6b the DGP is  Y_t = k * N_t + eps_t,
    # so gamma needs to scale with k.  We override gamma via a wrapper so that
    # the scenario dict still stores the base (k=1) gamma and the caller
    # supplies k through get_params.  Callers must pass the *actual* gamma
    # as  k * scenario['gamma']  for Group B.
    return scenarios


# ---------------------------------------------------------------------------
# Quantile forecast scenarios (exp_2)
# ---------------------------------------------------------------------------

def solve_xi0_equal_mcb(
    k: float,
    alpha: float,
    varsigma: float = 16 / 15,
) -> float:
    """Solve xi_0 numerically so that MCB_2 (scenario B5 quantile) equals MCB_1.

    Scenario B5 has:
      X1: delta_0 = 1/2, delta = (1/4, 0, 1/4, 0), gamma = (0,0,0,k)
      X2: xi = (0, 1/4+k/2, 0, 1/4+k/2), xi_0 unknown

    We find xi_0 such that population_components_cl gives equal MCB for both
    forecasts.  Uses scipy.optimize.brentq.

    Parameters
    ----------
    k : float
    alpha : float
    varsigma : float

    Returns
    -------
    float  xi_0 value
    """
    gamma = np.array([0.0, 0.0, 0.0, k])
    delta_B5 = np.array([1 / 4, 0.0, 1 / 4, 0.0])
    delta_0_B5 = 0.5
    xi_B5 = np.array([0.0, 1 / 4 + k / 2, 0.0, 1 / 4 + k / 2])

    mcb1 = population_components_cl(delta_0_B5, delta_B5, gamma, alpha, varsigma)["MCB"]

    def objective(xi_0: float) -> float:
        return (
            population_components_cl(xi_0, xi_B5, gamma, alpha, varsigma)["MCB"] - mcb1
        )

    # Search over a wide range
    try:
        xi_0_sol = scipy.optimize.brentq(objective, -10.0, 10.0, xtol=1e-10, maxiter=500)
    except ValueError:
        # If bracketing fails, return 0
        xi_0_sol = 0.0

    return float(xi_0_sol)


def get_quantile_scenarios() -> list[dict]:
    """Return the 4 quantile-forecast scenario definitions (Table 4 / exp_2).

    Each entry is a dict:
        'id'         : scenario label string
        'group'      : 'A' or 'B'
        'gamma'      : np.ndarray shape (4,) – base DGP coefficient
                       (for B scenarios the caller scales by k)
        'get_params' : callable(k, alpha) -> (delta_0, delta, xi_0, xi)

    Notes
    -----
    A1, A2: Group A (MCB-difference).
    B4, B5: Group B (DSC-difference).
    """
    s = np.sqrt

    def _z(alpha):
        return scipy.stats.norm.ppf(alpha)

    scenarios = [
        # ---- Group A ----
        {
            "id": "A1",
            "group": "A",
            "gamma": np.array([1 / 4, 1 / 4, 1 / 4, 0.0]),
            "get_params": lambda k, alpha: (
                ((4 / s(15)) - 1) * _z(alpha),
                np.array([1 / 4 + k / 2, 0.0, 1 / 4 + k / 2, 0.0]),
                ((4 / s(15)) - 1) * _z(alpha),
                np.array([0.0, 1 / 4 + k / 2, 1 / 4 + k / 2, 0.0]),
            ),
        },
        {
            "id": "A2",
            "group": "A",
            "gamma": np.array([1 / 4, 1 / 4, 1 / 4, 0.0]),
            "get_params": lambda k, alpha: (
                ((4 / s(15)) - 1) * _z(alpha),
                np.array([1 / 4, 0.0, 1 / 4, 0.0]),
                ((4 / s(15)) - 1) * _z(alpha),
                np.array([0.0, 1 / 4 + k / 2, 1 / 4 + k / 2, 0.0]),
            ),
        },
        # ---- Group B ----
        {
            "id": "B4",
            "group": "B",
            "gamma": np.array([0.0, 0.0, 0.0, 1.0]),   # scaled by k by caller
            "get_params": lambda k, alpha: (
                (s(1 + 8 * k ** 2 / 15) - 1) * _z(alpha),
                np.array([1 / 4 + k / 2, 0.0, 0.0, 1 / 4 + k / 2]),
                (s(1 + 8 * k ** 2 / 15) - 1) * _z(alpha),
                np.array([0.0, 1 / 4 + k / 2, 0.0, 1 / 4 + k / 2]),
            ),
        },
        {
            "id": "B5",
            "group": "B",
            "gamma": np.array([0.0, 0.0, 0.0, 1.0]),   # scaled by k by caller
            "get_params": lambda k, alpha: (
                0.5,
                np.array([1 / 4, 0.0, 1 / 4, 0.0]),
                solve_xi0_equal_mcb(k, alpha),
                np.array([0.0, 1 / 4 + k / 2, 0.0, 1 / 4 + k / 2]),
            ),
        },
    ]
    return scenarios
