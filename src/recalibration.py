"""Score-consistent recalibration module.

Estimates the recalibration regression  W_t^T theta  that minimises the
expected score, and provides reference-forecast estimators.
"""

import numpy as np
import statsmodels.api as sm


def fit_recalibration_se(
    X: np.ndarray,
    Y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate recalibration parameters via OLS (squared-error consistent).

    Minimises  (1/T) sum_t (theta_0 + theta_1 * X_t - Y_t)^2  with respect
    to theta = (theta_0, theta_1)^T using W_t = (1, X_t)^T.

    Parameters
    ----------
    X : np.ndarray, shape (T,)
        Original forecast series.
    Y : np.ndarray, shape (T,)
        Realized values.

    Returns
    -------
    theta_hat : np.ndarray, shape (2,)
        Estimated intercept and slope [theta_0, theta_1].
    Xc : np.ndarray, shape (T,)
        Recalibrated fitted values  W^T theta_hat = theta_0 + theta_1 * X.
    """
    W = sm.add_constant(X, has_constant="add")
    result = sm.OLS(Y, W).fit()
    theta_hat = result.params
    Xc = W @ theta_hat
    return theta_hat, Xc


def fit_recalibration_quantile(
    X: np.ndarray,
    Y: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate recalibration parameters via quantile regression.

    Minimises the check loss at level `alpha` over  theta = (theta_0, theta_1)^T
    with  W_t = (1, X_t)^T.  Uses statsmodels :class:`~statsmodels.regression.quantile_regression.QuantReg`.

    Parameters
    ----------
    X : np.ndarray, shape (T,)
        Original quantile forecast series.
    Y : np.ndarray, shape (T,)
        Realized values.
    alpha : float
        Quantile level in (0, 1).

    Returns
    -------
    theta_hat : np.ndarray, shape (2,)
        Estimated intercept and slope [theta_0, theta_1].
    Xc : np.ndarray, shape (T,)
        Recalibrated fitted values  W^T theta_hat.
    """
    W = sm.add_constant(X, has_constant="add")
    result = sm.QuantReg(Y, W).fit(q=alpha, disp=False)
    theta_hat = result.params
    Xc = W @ theta_hat
    return theta_hat, Xc


def compute_reference_mean(Y: np.ndarray) -> float:
    """Estimate the reference forecast as the sample mean of Y.

    Under squared error, the optimal constant forecast is E[Y], estimated
    consistently by the sample mean.

    Parameters
    ----------
    Y : np.ndarray, shape (T,)

    Returns
    -------
    float
        Sample mean of Y.
    """
    return float(np.mean(Y))


def compute_reference_quantile(Y: np.ndarray, alpha: float) -> float:
    """Estimate the reference forecast as the empirical alpha-quantile of Y.

    Under check loss at level alpha, the optimal constant forecast is the
    alpha-quantile of Y's marginal distribution.

    Implementation note: 'midpoint' interpolation is used (average of the two
    surrounding order statistics when the quantile falls between observations).
    This matches R's default ``type=2`` behaviour and avoids stochastic
    tie-breaking, making results exactly reproducible across platforms.

    Parameters
    ----------
    Y : np.ndarray, shape (T,)
    alpha : float
        Quantile level in (0, 1).

    Returns
    -------
    float
        Empirical alpha-quantile of Y.
    """
    return float(np.quantile(Y, alpha, method="midpoint"))
