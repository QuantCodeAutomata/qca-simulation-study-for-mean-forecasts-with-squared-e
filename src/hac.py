"""HAC covariance estimation module.

Implements the Bartlett (Newey–West) heteroskedasticity-and-autocorrelation
consistent long-run covariance estimator with the Andrews (1991) AR(1)
plug-in bandwidth selection rule.
"""

import numpy as np


def newey_west_bandwidth(T: int, Z: np.ndarray) -> int:
    """Automatic bandwidth for the Bartlett kernel (Andrews 1991 plug-in rule).

    For each column z of Z, fit an AR(1) by OLS and obtain rho_hat.  Average
    the per-column rho_hat values, then apply the Andrews (1991) formula for
    the optimal Bartlett bandwidth:

        rho_bar = mean over columns of rho_hat_j
        alpha2  = rho_bar^2 / (1 - rho_bar^2)^2
        alpha1  = (1 + rho_bar^2) / (1 - rho_bar^2)^2   [for Bartlett kernel]
        S_hat   = 4 * alpha2 / alpha1                    [= 4 * rho_bar^2]
        m_star  = 1.1447 * (S_hat * T)^{1/3}

    Falls back to floor(T^{1/3}) when rho_bar is too close to 1 or the
    formula yields a non-positive bandwidth.

    Parameters
    ----------
    T : int
        Sample size.
    Z : np.ndarray, shape (T, k)
        Multivariate time series (demeaned or raw; demeaning is done inside).

    Returns
    -------
    int
        Non-negative integer bandwidth (lag truncation point).
    """
    if Z.ndim == 1:
        Z = Z[:, np.newaxis]

    Z_dm = Z - Z.mean(axis=0)  # demean for AR(1) estimation

    rho_hats = []
    for j in range(Z_dm.shape[1]):
        z = Z_dm[:, j]
        if np.std(z) < 1e-14:          # constant column – no autocorrelation
            rho_hats.append(0.0)
            continue
        # OLS: regress z[1:] on z[:-1] without intercept after demeaning
        y_reg = z[1:]
        x_reg = z[:-1]
        denom = x_reg @ x_reg
        rho = float(x_reg @ y_reg / denom) if denom > 0 else 0.0
        # Clip to [-0.999, 0.999] for numerical stability
        rho = np.clip(rho, -0.999, 0.999)
        rho_hats.append(rho)

    rho_bar = float(np.mean(rho_hats))

    try:
        denom1 = (1.0 - rho_bar ** 2) ** 2
        alpha2 = rho_bar ** 2 / denom1                  # sum of squared spectral density weights
        alpha1 = (1.0 + rho_bar ** 2) / denom1          # derivative-based weight
        # For the Bartlett kernel the Andrews formula simplifies to
        #   m* = 1.1447 * (4 * alpha2 / alpha1 * T)^{1/3}
        # which equals  1.1447 * (4 * rho_bar^2 / (1 + rho_bar^2) * T)^{1/3}
        ratio = 4.0 * alpha2 / alpha1
        if ratio <= 0:
            raise ValueError("Non-positive ratio in bandwidth formula")
        m_star = int(1.1447 * (ratio * T) ** (1.0 / 3.0))
    except (ZeroDivisionError, ValueError):
        m_star = int(T ** (1.0 / 3.0))

    # Ensure at least 1 and no more than T-1
    return max(1, min(m_star, T - 1))


def estimate_hac_covariance(
    Z: np.ndarray,
    bandwidth: int | None = None,
) -> np.ndarray:
    """Long-run (HAC) covariance matrix using the Bartlett (Newey–West) kernel.

    Computes the Newey–West estimator

        Omega_hat = Gamma_0 + sum_{h=1}^{m} w_h (Gamma_h + Gamma_h^T)

    where  Gamma_h = (1/T) Z_dm^T_{[1:T-h]} Z_dm_{[1+h:T]}  is the
    sample autocovariance at lag h, and  w_h = 1 - h/(m+1)  is the
    Bartlett weight.

    Approximates Andrews (1991) and the R ``sandwich::vcovHAC`` default with
    automatic bandwidth selection.  Returns Omega_hat (NOT divided by T),
    analogous to the long-run variance used in sandwich standard errors.

    Parameters
    ----------
    Z : np.ndarray, shape (T, k)
        Multivariate time series; each row is one observation.
    bandwidth : int or None
        Lag truncation point m.  If None, selected via
        :func:`newey_west_bandwidth`.

    Returns
    -------
    np.ndarray, shape (k, k)
        HAC covariance matrix Omega_hat.
    """
    if Z.ndim == 1:
        Z = Z[:, np.newaxis]

    T, k = Z.shape
    Z_dm = Z - Z.mean(axis=0)

    if bandwidth is None:
        bandwidth = newey_west_bandwidth(T, Z)

    m = min(bandwidth, T - 1)

    # Gamma_0
    Omega = (Z_dm.T @ Z_dm) / T

    # Add weighted autocovariances
    for h in range(1, m + 1):
        w = 1.0 - h / (m + 1.0)          # Bartlett weight
        Gamma_h = (Z_dm[h:].T @ Z_dm[:-h]) / T
        Omega += w * (Gamma_h + Gamma_h.T)

    return Omega


def estimate_hac_covariance_scalar(
    z: np.ndarray,
    bandwidth: int | None = None,
) -> float:
    """HAC long-run variance for a univariate series z.

    Convenience wrapper around :func:`estimate_hac_covariance` for the
    scalar (k=1) case.

    Parameters
    ----------
    z : np.ndarray, shape (T,)
    bandwidth : int or None

    Returns
    -------
    float
        HAC variance estimate (scalar).
    """
    Omega = estimate_hac_covariance(z[:, np.newaxis], bandwidth=bandwidth)
    return float(Omega[0, 0])
