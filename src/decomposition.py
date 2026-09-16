"""Score decomposition module.

Implements elementary scoring functions and the Murphy–Diebold–Mariano
decomposition  S = MCB - DSC + UNC  for both squared error and check loss.
"""

import numpy as np


def squared_error(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Squared-error scoring function  S_SE(x, y) = (x - y)^2.

    Parameters
    ----------
    x : np.ndarray
        Forecast values.
    y : np.ndarray
        Realized values.

    Returns
    -------
    np.ndarray
        Element-wise squared errors.
    """
    return (x - y) ** 2


def check_loss(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Check (pinball) loss at level alpha.

    S_{CL,alpha}(x, y) = (1{y <= x} - alpha) * (x - y)

    Parameters
    ----------
    x : np.ndarray
        Quantile forecast values.
    y : np.ndarray
        Realized values.
    alpha : float
        Quantile level in (0, 1).

    Returns
    -------
    np.ndarray
        Element-wise check-loss scores.
    """
    return (np.where(y <= x, 1.0, 0.0) - alpha) * (x - y)


def compute_decomposition(
    score_fn,
    X: np.ndarray,
    Y: np.ndarray,
    Xc: np.ndarray,
    r_hat: float,
) -> dict:
    """Compute the Murphy score decomposition: S = MCB - DSC + UNC.

    Given a proper scoring rule S and forecast X, the decomposition is:

        S_hat       = mean S(X_t, Y_t)                     (mean score)
        S_hat_C     = mean S(Xc_t, Y_t)                    (recalibrated score)
        S_hat_R     = mean S(r_hat, Y_t)                   (reference score)
        MCB_hat     = S_hat   - S_hat_C  >= 0              (miscalibration)
        DSC_hat     = S_hat_R - S_hat_C  >= 0              (discrimination)
        UNC_hat     = S_hat_R                              (uncertainty)

    The identity  S_hat = MCB_hat - DSC_hat + UNC_hat  is verified.

    Parameters
    ----------
    score_fn : callable(x, y) -> array
        Scoring function; must accept array inputs.
    X : np.ndarray, shape (T,)
        Original forecast.
    Y : np.ndarray, shape (T,)
        Realized values.
    Xc : np.ndarray, shape (T,)
        Recalibrated fitted values W_t^T theta_hat.
    r_hat : float
        Scalar reference forecast (e.g. sample mean or empirical quantile).

    Returns
    -------
    dict with keys:
        S     : mean score of X
        S_C   : mean score of recalibrated forecast Xc
        S_R   : mean score of reference forecast r_hat
        MCB   : miscalibration component (S - S_C)
        DSC   : discrimination component (S_R - S_C)
        UNC   : uncertainty component (S_R)
    """
    S_bar = float(np.mean(score_fn(X, Y)))
    S_C_bar = float(np.mean(score_fn(Xc, Y)))
    S_R_bar = float(np.mean(score_fn(np.full_like(Y, r_hat), Y)))

    MCB = S_bar - S_C_bar
    DSC = S_R_bar - S_C_bar
    UNC = S_R_bar

    decomp = {
        "S": S_bar,
        "S_C": S_C_bar,
        "S_R": S_R_bar,
        "MCB": MCB,
        "DSC": DSC,
        "UNC": UNC,
    }

    verify_decomposition_identity(decomp)
    return decomp


def verify_decomposition_identity(decomp: dict, tol: float = 1e-8) -> bool:
    """Verify  S = MCB - DSC + UNC  and  MCB >= 0,  DSC >= 0.

    Parameters
    ----------
    decomp : dict
        Output of :func:`compute_decomposition`.
    tol : float
        Numerical tolerance for equality and non-negativity checks.

    Returns
    -------
    bool
        True if all checks pass.

    Raises
    ------
    AssertionError
        If the identity or non-negativity conditions are violated.
    """
    S = decomp["S"]
    MCB = decomp["MCB"]
    DSC = decomp["DSC"]
    UNC = decomp["UNC"]

    lhs = S
    rhs = MCB - DSC + UNC
    assert abs(lhs - rhs) <= tol, (
        f"Decomposition identity violated: S={S:.6g} != MCB-DSC+UNC={rhs:.6g} "
        f"(diff={abs(lhs - rhs):.2e})"
    )
    assert MCB >= -tol, f"MCB={MCB:.6g} < 0 (non-negativity violated)"
    assert DSC >= -tol, f"DSC={DSC:.6g} < 0 (non-negativity violated)"
    return True
