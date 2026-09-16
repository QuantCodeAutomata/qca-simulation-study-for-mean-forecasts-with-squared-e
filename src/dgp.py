"""Data Generating Process module.

Provides functions to simulate AR(1) predictor processes, generate targets,
and construct both mean and quantile forecasts as described in the paper.
"""

import numpy as np
import scipy.stats


def generate_ar1_predictors(
    T: int,
    beta: float = 0.25,
    burn_in: int = 200,
    seed: int | None = None,
) -> np.ndarray:
    """Generate V_t = (K_t, L_t, M_t, N_t)^T as four mutually independent AR(1) processes.

    Each process follows  X_t = beta * X_{t-1} + e_t,  e_t iid N(0,1).
    The stationary variance is 1 / (1 - beta^2).

    Initialization strategy: start from 0 and run `burn_in` additional steps
    (rather than drawing from the exact stationary distribution) so that the
    series has reached approximate stationarity before the first retained
    observation.  burn_in=200 is large enough for |beta| <= 0.9 and avoids
    introducing dependence on the analytic stationary variance formula.

    Parameters
    ----------
    T:
        Number of time steps to return.
    beta:
        AR(1) coefficient (|beta| < 1 required for stationarity).
    burn_in:
        Number of initial steps discarded to achieve approximate stationarity.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    V : np.ndarray, shape (T, 4)
        Rows are time steps; columns are K, L, M, N.
    """
    rng = np.random.default_rng(seed)
    total = T + burn_in

    # Draw all innovations at once: shape (total, 4)
    innovations = rng.standard_normal((total, 4))

    V_all = np.zeros((total, 4))
    for t in range(1, total):
        V_all[t] = beta * V_all[t - 1] + innovations[t]

    # Discard burn-in
    return V_all[burn_in:]


def generate_target(
    V: np.ndarray,
    gamma: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    """Generate Y_t = V_t^T gamma + e_{Y,t},  e_{Y,t} iid N(0,1).

    Parameters
    ----------
    V : np.ndarray, shape (T, 4)
        Predictor matrix from :func:`generate_ar1_predictors`.
    gamma : np.ndarray, shape (4,)
        Coefficient vector.
    seed : int or None
        Random seed.

    Returns
    -------
    Y : np.ndarray, shape (T,)
    """
    rng = np.random.default_rng(seed)
    return V @ gamma + rng.standard_normal(len(V))


def construct_mean_forecasts(
    V: np.ndarray,
    delta_0: float,
    delta: np.ndarray,
    xi_0: float,
    xi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct linear mean forecasts from V.

    X_{1t} = delta_0 + V_t^T delta
    X_{2t} = xi_0   + V_t^T xi

    Parameters
    ----------
    V : np.ndarray, shape (T, 4)
    delta_0, xi_0 : float
        Intercepts.
    delta, xi : np.ndarray, shape (4,)
        Slope vectors.

    Returns
    -------
    (X1, X2) : tuple of np.ndarray, each shape (T,)
    """
    X1 = delta_0 + V @ delta
    X2 = xi_0 + V @ xi
    return X1, X2


def construct_quantile_forecasts(
    V: np.ndarray,
    delta_0: float,
    delta: np.ndarray,
    xi_0: float,
    xi: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct quantile forecasts at level alpha.

    X_{1t}^alpha = delta_0 + V_t^T delta + z_alpha
    X_{2t}^alpha = xi_0   + V_t^T xi   + z_alpha

    where z_alpha = Phi^{-1}(alpha) (standard normal quantile).

    Parameters
    ----------
    V : np.ndarray, shape (T, 4)
    delta_0, xi_0 : float
        Intercepts (before adding z_alpha).
    delta, xi : np.ndarray, shape (4,)
        Slope vectors.
    alpha : float
        Quantile level in (0, 1).

    Returns
    -------
    (X1, X2) : tuple of np.ndarray, each shape (T,)
    """
    z_alpha = scipy.stats.norm.ppf(alpha)
    X1 = delta_0 + V @ delta + z_alpha
    X2 = xi_0 + V @ xi + z_alpha
    return X1, X2
