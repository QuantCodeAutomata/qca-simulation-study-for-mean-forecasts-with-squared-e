"""Statistical inference module.

Implements hypothesis tests for the score-decomposition components:
  - Equal-MCB test  (H0: MCB_1 = MCB_2)
  - Equal-DSC test  (H0: DSC_1 = DSC_2)
  - Diebold–Mariano (DM) equal-score test  (H0: S_1 = S_2)

Both squared-error (mean forecast) and check-loss (quantile forecast)
variants are provided.  Boundary p-values use the Imhof generalized
chi-square distribution for the MCB/DSC=0 boundary cases.
"""

import numpy as np
import scipy.stats
import statsmodels.api as sm
from typing import Literal

from .decomposition import squared_error, check_loss
from .hac import estimate_hac_covariance
from .imhof import generalized_chisq_pvalue


# ---------------------------------------------------------------------------
# Squared-error (mean forecast) inference
# ---------------------------------------------------------------------------

def build_score_process_se(
    X1: np.ndarray,
    Y: np.ndarray,
    Xc1: np.ndarray,
    r_hat: float,
    X2: np.ndarray,
    Xc2: np.ndarray,
) -> np.ndarray:
    """Build the 5-vector score process Z_t for HAC estimation (squared error).

    Z_t = (S(X1_t, Y_t), S(Xc1_t, Y_t), S(X2_t, Y_t), S(Xc2_t, Y_t), S(r_hat, Y_t))

    where S is squared error.

    Parameters
    ----------
    X1, X2 : np.ndarray, shape (T,)
        Original forecasts for models 1 and 2.
    Y : np.ndarray, shape (T,)
        Realized values.
    Xc1, Xc2 : np.ndarray, shape (T,)
        Recalibrated (fitted) forecasts.
    r_hat : float
        Reference forecast scalar.

    Returns
    -------
    np.ndarray, shape (T, 5)
    """
    R = np.full_like(Y, r_hat)
    return np.column_stack([
        squared_error(X1, Y),
        squared_error(Xc1, Y),
        squared_error(X2, Y),
        squared_error(Xc2, Y),
        squared_error(R, Y),
    ])


def build_xi_matrix() -> np.ndarray:
    """Return the 4×5 selection matrix Xi.

    Xi maps Z_t = (S1, S1C, S2, S2C, SR) to the component differences:
        row 0: MCB1 = S1  - S1C  ->  (1, -1,  0,  0,  0)
        row 1: DSC1 = SR  - S1C  ->  (0, -1,  0,  0,  1)
        row 2: MCB2 = S2  - S2C  ->  (0,  0,  1, -1,  0)
        row 3: DSC2 = SR  - S2C  ->  (0,  0,  0, -1,  1)

    Applying  Xi @ Z_t.T  gives a (4, T) matrix whose row means are
    (MCB1_hat, DSC1_hat, MCB2_hat, DSC2_hat).

    Returns
    -------
    np.ndarray, shape (4, 5)
    """
    return np.array([
        [1, -1,  0,  0,  0],
        [0, -1,  0,  0,  1],
        [0,  0,  1, -1,  0],
        [0,  0,  0, -1,  1],
    ], dtype=float)


def gaussian_component_test(
    mcb_dsc_hats: np.ndarray,
    omega: np.ndarray,
    Xi: np.ndarray,
    Omega_hat: np.ndarray,
    T: int,
) -> float:
    """Gaussian positive-component branch p-value (two-sided Wald test).

    Computes the Wald statistic

        W = T * (omega^T xi_bar)^2 / (omega^T Xi Omega_hat Xi^T omega)

    where  xi_bar = (MCB1, DSC1, MCB2, DSC2)^T,  and returns the two-sided
    p-value from a chi^2(1) distribution (equivalently, two-sided N(0,1)).

    Parameters
    ----------
    mcb_dsc_hats : np.ndarray, shape (4,)
        Estimated component vector (MCB1, DSC1, MCB2, DSC2).
    omega : np.ndarray, shape (4,)
        Linear combination selector.
    Xi : np.ndarray, shape (4, 5)
        Score-process selection matrix from :func:`build_xi_matrix`.
    Omega_hat : np.ndarray, shape (5, 5)
        HAC long-run covariance of Z_t.
    T : int
        Sample size.

    Returns
    -------
    float
        Two-sided p-value in [0, 1].
    """
    numerator = float(omega @ mcb_dsc_hats) ** 2
    sandwich = float(omega @ Xi @ Omega_hat @ Xi.T @ omega)
    if sandwich <= 0.0:
        return 1.0
    wald = T * numerator / sandwich
    # chi^2(1) two-sided p-value
    return float(1.0 - scipy.stats.chi2.cdf(wald, df=1))


def boundary_pvalue_se(
    T: int,
    mcb_or_dsc_hat: float,
    X: np.ndarray,
    Xc: np.ndarray,
    Y: np.ndarray,
    r_hat: float,
    component: Literal["mcb", "dsc"],
) -> float:
    """Boundary p-value for H0: MCB_i = 0 or H0: DSC_i = 0 (squared error).

    Under squared error  S(x,y) = (x-y)^2  with W_t = (1, X_t)^T:

        S'(x,y)  = 2(x-y)           (gradient with respect to x)
        S''(x,y) = 2                 (Hessian with respect to x, scalar)

    The M-estimator sandwich theory gives an asymptotic distribution of the
    form  T * stat ~_d  sum_j lambda_j chi^2_1  under the null.

    For the MCB=0 test (H0: theta = (0,1)^T):
        Upsilon = E[S'' W W^T] ≈ 2/T * W^T W
        Pi      = HAC covariance of  {S'(Xc_t, Y_t) * W_t}
                = HAC covariance of  {2*(Xc_t - Y_t) * W_t}
        Weights lambda_j = eigenvalues of Upsilon^{-1/2} Pi Upsilon^{-1/2}

    For the DSC=0 test (H0: DSC = 0):
        Uses residuals from the reference-score process.
        Upsilon_R ≈ 2/T * 1^T 1 = 2  (scalar, since W=1)
        Pi_R = HAC variance of  {2*(r_hat - Y_t)}
        lambda = Pi_R / Upsilon_R

    Parameters
    ----------
    T : int
        Sample size.
    mcb_or_dsc_hat : float
        Estimated MCB or DSC.
    X : np.ndarray, shape (T,)
        Original forecast (for MCB) or recalibrated (Xc) if component='dsc'.
    Xc : np.ndarray, shape (T,)
        Recalibrated fitted forecast.
    Y : np.ndarray, shape (T,)
        Realized values.
    r_hat : float
        Reference forecast scalar.
    component : {'mcb', 'dsc'}
        Which boundary case to test.

    Returns
    -------
    float
        Imhof p-value P(Q > T * mcb_or_dsc_hat) in [0, 1].
    """
    test_stat = T * mcb_or_dsc_hat

    if component == "mcb":
        # W_t = (1, X_t)^T
        W = sm.add_constant(X, has_constant="add")  # shape (T, 2)
        # Upsilon = 2 * (W^T W) / T
        Upsilon = 2.0 * (W.T @ W) / T
        # Score gradient process: S'(Xc_t, Y_t) * W_t = 2*(Xc_t - Y_t) * W_t
        grad_process = 2.0 * (Xc - Y)[:, np.newaxis] * W   # (T, 2)
        Pi = estimate_hac_covariance(grad_process)           # (2, 2)

    else:  # component == "dsc"
        # Scalar case: W=1, Xc = r_hat (constant reference)
        # Upsilon_R = 2
        Upsilon = np.array([[2.0]])
        # Score gradient of reference: 2*(r_hat - Y_t)
        grad_process = (2.0 * (r_hat - Y))[:, np.newaxis]  # (T, 1)
        Pi = estimate_hac_covariance(grad_process)           # (1, 1)

    # Compute eigenvalues of Upsilon^{-1/2} Pi Upsilon^{-1/2}
    # Equivalent to eigenvalues of Upsilon^{-1} Pi (same non-zero eigenvalues)
    try:
        Upsilon_inv = np.linalg.inv(Upsilon)
        A = Upsilon_inv @ Pi
        eigenvalues = np.linalg.eigvalsh(0.5 * (A + A.T))   # symmetrize for stability
    except np.linalg.LinAlgError:
        return 1.0

    return generalized_chisq_pvalue(test_stat, eigenvalues)


def equal_mcb_test(
    T: int,
    MCB1: float,
    MCB2: float,
    Xi: np.ndarray,
    Omega_hat: np.ndarray,
    boundary_p1: float,
    boundary_p2: float,
) -> float:
    """Hybrid equal-MCB p-value for H0: MCB_1 = MCB_2.

    Uses the selection vector  omega_MCB = (1, 0, -1, 0)^T  to test the
    difference MCB_1 - MCB_2 = 0 via a Wald test, subject to the boundary
    correction:

        p_MCB = min(1, max(p_MCB^+, 2 * min(boundary_p1, boundary_p2)))

    Parameters
    ----------
    T : int
        Sample size.
    MCB1, MCB2 : float
        Estimated MCB components.
    Xi : np.ndarray, shape (4, 5)
    Omega_hat : np.ndarray, shape (5, 5)
        HAC long-run covariance of Z_t.
    boundary_p1, boundary_p2 : float
        Boundary p-values for H0: MCB_1=0 and H0: MCB_2=0.

    Returns
    -------
    float
        Hybrid p-value in [0, 1].
    """
    omega = np.array([1.0, 0.0, -1.0, 0.0])
    mcb_dsc_hats = np.array([MCB1, 0.0, MCB2, 0.0])   # only MCB entries matter
    p_plus = gaussian_component_test(mcb_dsc_hats, omega, Xi, Omega_hat, T)
    p_boundary = 2.0 * min(boundary_p1, boundary_p2)
    return float(min(1.0, max(p_plus, p_boundary)))


def equal_dsc_test(
    T: int,
    DSC1: float,
    DSC2: float,
    Xi: np.ndarray,
    Omega_hat: np.ndarray,
    boundary_p1: float,
    boundary_p2: float,
) -> float:
    """Hybrid equal-DSC p-value for H0: DSC_1 = DSC_2.

    Uses  omega_DSC = (0, 1, 0, -1)^T.

        p_DSC = min(1, max(p_DSC^+, 2 * min(boundary_p1, boundary_p2)))

    Parameters
    ----------
    T : int
        Sample size.
    DSC1, DSC2 : float
        Estimated DSC components.
    Xi : np.ndarray, shape (4, 5)
    Omega_hat : np.ndarray, shape (5, 5)
    boundary_p1, boundary_p2 : float
        Boundary p-values for H0: DSC_1=0 and H0: DSC_2=0.

    Returns
    -------
    float
        Hybrid p-value in [0, 1].
    """
    omega = np.array([0.0, 1.0, 0.0, -1.0])
    mcb_dsc_hats = np.array([0.0, DSC1, 0.0, DSC2])   # only DSC entries matter
    p_plus = gaussian_component_test(mcb_dsc_hats, omega, Xi, Omega_hat, T)
    p_boundary = 2.0 * min(boundary_p1, boundary_p2)
    return float(min(1.0, max(p_plus, p_boundary)))


def dm_test(
    T: int,
    Xi: np.ndarray,
    Omega_hat: np.ndarray,
    mcb_dsc_hats: np.ndarray,
) -> float:
    """Diebold–Mariano equal-score test (H0: S_1 = S_2).

    The mean-score difference is

        S_bar_1 - S_bar_2 = (MCB_1 - DSC_1) - (MCB_2 - DSC_2)
                          = MCB_1 - DSC_1 - MCB_2 + DSC_2

    which corresponds to the selector  omega_S = (1, -1, -1, 1)^T  applied to
    (MCB_1, DSC_1, MCB_2, DSC_2).

    Parameters
    ----------
    T : int
        Sample size.
    Xi : np.ndarray, shape (4, 5)
    Omega_hat : np.ndarray, shape (5, 5)
    mcb_dsc_hats : np.ndarray, shape (4,)
        (MCB1, DSC1, MCB2, DSC2).

    Returns
    -------
    float
        Two-sided p-value in [0, 1].
    """
    omega = np.array([1.0, -1.0, -1.0, 1.0])
    return gaussian_component_test(mcb_dsc_hats, omega, Xi, Omega_hat, T)


# ---------------------------------------------------------------------------
# Check-loss (quantile forecast) inference
# ---------------------------------------------------------------------------

def build_score_process_cl(
    X1: np.ndarray,
    Y: np.ndarray,
    Xc1: np.ndarray,
    r_hat: float,
    X2: np.ndarray,
    Xc2: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Build the 5-vector score process Z_t under check loss.

    Z_t = (S_CL(X1_t, Y_t), S_CL(Xc1_t, Y_t),
           S_CL(X2_t, Y_t), S_CL(Xc2_t, Y_t), S_CL(r_hat, Y_t))

    Parameters
    ----------
    X1, X2 : np.ndarray, shape (T,)
    Y : np.ndarray, shape (T,)
    Xc1, Xc2 : np.ndarray, shape (T,)
    r_hat : float
        Reference quantile forecast.
    alpha : float
        Quantile level.

    Returns
    -------
    np.ndarray, shape (T, 5)
    """
    def cl(x, y):
        return check_loss(x, y, alpha)

    R = np.full_like(Y, r_hat)
    return np.column_stack([
        cl(X1, Y),
        cl(Xc1, Y),
        cl(X2, Y),
        cl(Xc2, Y),
        cl(R, Y),
    ])


def boundary_pvalue_quantile_mcb(
    T: int,
    mcb_hat: float,
    X: np.ndarray,
    Xc: np.ndarray,
    Y: np.ndarray,
    alpha: float,
) -> float:
    """Boundary p-value for H0: MCB_i = 0 under check loss (VQR-type Wald test).

    Under the null MCB_i = 0, the recalibration regression satisfies
    theta = (0, 1)^T in the quantile regression of Y on W_t = (1, X_t)^T.
    This is tested via a HAC Wald statistic for the restriction H: theta = (0,1).

    The restricted residuals are  u_t = (alpha - 1{Y_t <= X_t}) * 1  (score
    gradient evaluated at the null).  The HAC covariance of the gradient
    process and the Hessian Upsilon are combined to form the Wald statistic,
    distributed chi^2 under the null (interior case) or via Imhof at the boundary.

    Parameters
    ----------
    T : int
        Sample size.
    mcb_hat : float
        Estimated MCB.
    X : np.ndarray, shape (T,)
        Original quantile forecast.
    Xc : np.ndarray, shape (T,)
        Recalibrated quantile forecast.
    Y : np.ndarray, shape (T,)
        Realized values.
    alpha : float
        Quantile level.

    Returns
    -------
    float
        P-value for H0: MCB_i = 0 in [0, 1].
    """
    W = sm.add_constant(X, has_constant="add")  # (T, 2)

    # Score gradient (subgradient) of check loss at Xc_t:
    # psi_t(theta) = (1{Y_t <= Xc_t} - alpha) * W_t
    indicator = (Y <= Xc).astype(float)
    grad_process = (indicator - alpha)[:, np.newaxis] * W  # (T, 2)

    # HAC covariance of gradient process
    Pi = estimate_hac_covariance(grad_process)  # (2, 2)

    # Hessian (density at 0): Upsilon ≈ f_eps(0) * E[W W^T]
    # Under standard normal errors, f_eps(0) = phi(0) = 1/sqrt(2*pi) but we
    # use a kernel density estimate at 0 for the residuals eps_t = Y_t - Xc_t
    eps = Y - Xc
    bw = 1.06 * np.std(eps) * T ** (-0.2)  # Silverman bandwidth
    bw = max(bw, 1e-6)
    f0 = float(np.mean(scipy.stats.norm.pdf(eps / bw) / bw))
    Upsilon = f0 * (W.T @ W) / T  # (2, 2)

    try:
        Upsilon_inv = np.linalg.inv(Upsilon)
    except np.linalg.LinAlgError:
        return 1.0

    # Wald statistic for H0: theta = (0,1), i.e. restriction R*theta = r
    # where R = I_2, r = (0,1). We use the mean gradient evaluated at null.
    # Under H0 the mean gradient ~_d N(0, Pi/T) -> Wald ~ chi^2(2) at interior
    # but at the MCB=0 boundary we compute via Imhof.
    A = Upsilon_inv @ Pi
    eigenvalues = np.linalg.eigvalsh(0.5 * (A + A.T))
    test_stat = T * mcb_hat
    return generalized_chisq_pvalue(test_stat, eigenvalues)


def boundary_pvalue_quantile_dsc(
    T: int,
    dsc_hat: float,
    Xc: np.ndarray,
    Y: np.ndarray,
    alpha: float,
    r_hat: float,
) -> float:
    """Boundary p-value for H0: DSC_i = 0 under check loss (VQR-type Wald test).

    Under the null DSC_i = 0, the reference-score process should be no better
    than the recalibrated score; equivalently, the optimal intercept in the
    quantile regression of Y on a constant equals r_hat.

    We test H0 via the HAC Wald statistic for the slope of the reference
    check-loss gradient process against the null, using a scalar formulation:

        Upsilon_R = f_eps_R(0)  (density at residual 0 for reference)
        Pi_R      = HAC variance of  {(1{Y_t <= r_hat} - alpha)}

    The eigenvalue lambda = Pi_R / Upsilon_R^2 is used with Imhof.

    Parameters
    ----------
    T : int
    dsc_hat : float
        Estimated DSC.
    Xc : np.ndarray, shape (T,)
        Recalibrated forecast (used to obtain residuals for bandwidth).
    Y : np.ndarray, shape (T,)
    alpha : float
    r_hat : float
        Reference quantile forecast.

    Returns
    -------
    float
        P-value for H0: DSC_i = 0 in [0, 1].
    """
    # Gradient process at reference r_hat (constant forecast)
    indicator_R = (Y <= r_hat).astype(float)
    grad_R = (indicator_R - alpha)  # (T,) scalar process

    # HAC variance of gradient
    Pi_R = estimate_hac_covariance(grad_R[:, np.newaxis])  # (1,1)

    # Density at 0 for the reference residuals eps_R = Y - r_hat
    eps_R = Y - r_hat
    bw = 1.06 * np.std(eps_R) * len(Y) ** (-0.2)
    bw = max(bw, 1e-6)
    f0_R = float(np.mean(scipy.stats.norm.pdf(eps_R / bw) / bw))

    # Upsilon_R = f0_R (scalar Hessian at reference)
    Upsilon_R = max(f0_R, 1e-10)

    # Eigenvalue: lambda = Pi_R / Upsilon_R^2
    lambda_val = float(Pi_R[0, 0]) / (Upsilon_R ** 2)
    eigenvalues = np.array([lambda_val])

    test_stat = T * dsc_hat
    return generalized_chisq_pvalue(test_stat, eigenvalues)
