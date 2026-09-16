"""Imhof (1961) integration for generalized chi-square CDF / p-values.

Computes P(Q > q) where Q = sum_j lambda_j * chi^2_1 via characteristic
function inversion.  No third-party library is used for the core formula;
only scipy.integrate.quad is used for the numerical quadrature.
"""

import numpy as np
import scipy.integrate
import scipy.stats


def imhof_pvalue(
    q_stat: float,
    lambdas: np.ndarray,
    epsabs: float = 1.49e-8,
    epsrel: float = 1.49e-8,
    limit: int = 500,
) -> float:
    """Compute P(Q > q_stat) via Imhof's (1961) characteristic function inversion.

    For Q = sum_j lambda_j * chi^2_1 (each chi^2_1 independent), the
    characteristic function is

        phi(t) = prod_j (1 - 2 i t lambda_j)^{-1/2}

    and the tail probability is obtained via (Imhof 1961, eq. 3):

        P(Q > q) = 1/2 + (1/pi) * integral_0^inf  sin(theta(u, q)) / (u * rho(u))  du

    where (with h_j = 1 degrees of freedom per chi^2 component):

        theta(u, q) = (1/2) * sum_j arctan(lambda_j * u) - q * u / 2
        rho(u)      = prod_j (1 + lambda_j^2 * u^2)^{1/4}

    Note: this uses arctan(lambda_j * u), derived by substituting s=2t in the
    CF inversion integral (so the chi^2 CF becomes (1-i*s*lambda_j)^{-1/2}).
    The characteristic function in the paper notation is phi(t) with t=s/2.

    Numerical issues: the integrand oscillates and decays to zero; when
    |theta(u,q)| is very large (> 1e15) we set the integrand to 0.

    Special case: when J=1 and lambda > 0, uses the exact chi-squared formula
    P(lambda * chi^2_1 > q) = P(chi^2_1 > q/lambda) to avoid the conditionally
    convergent integral (amplitude ~ u^{-3/2} for J=1 makes numerical quadrature
    slow and unreliable with finite subdivision budgets).

    For J >= 2, the amplitude decays as u^{-(J/2+1)}, which is absolutely
    integrable, and the adaptive quadrature with a finite upper limit suffices.

    Parameters
    ----------
    q_stat : float
        Value of the test statistic Q.
    lambdas : np.ndarray, shape (J,)
        Positive or mixed-sign weights of the chi^2_1 components.
    epsabs, epsrel : float
        Absolute and relative tolerances for scipy.integrate.quad.
    limit : int
        Maximum number of subintervals for the integrator.

    Returns
    -------
    float
        Estimated tail probability P(Q > q_stat), clipped to [0, 1].
    """
    lambdas = np.asarray(lambdas, dtype=float)
    k = len(lambdas)

    # -- Exact formula for a single positive eigenvalue -----------------------
    # Q = lambda * chi^2_1, so P(Q > q) = P(chi^2_1 > q/lambda).
    # This avoids the slow conditional convergence of the Imhof integral
    # when k=1 (integrand amplitude ~ u^{-3/2}).
    if k == 1:
        lam = float(lambdas[0])
        if lam <= 0.0:
            return 0.0 if q_stat > 0.0 else 1.0
        scaled = q_stat / lam
        if scaled <= 0.0:
            return 1.0
        return float(scipy.stats.chi2.sf(scaled, df=1))

    # -- General Imhof integration (k >= 2) -----------------------------------
    # For k >= 2 the integrand amplitude decays as u^{-(k/2+1)} (absolutely
    # integrable), so we use adaptive Gaussian quadrature on a piecewise grid.
    #
    # Correct Imhof (1961) formula (derived by substituting s=2t in the CF
    # inversion integral so that the CF of chi^2_h becomes (1-is*lambda)^{-h/2}):
    #
    #   theta(u) = (1/2)*sum_j h_j * arctan(lambda_j * u) - q * u / 2
    #   rho(u)   = prod_j (1 + lambda_j^2 * u^2)^{h_j/4}
    #
    # For h_j=1 (chi^2_1) the asymptotic form is
    #   theta(u) -> k*pi/2 - q*u/2  as u -> inf
    # so zeros of sin(theta) are spaced by half-period = 2*pi / |q_stat|.

    def _theta(u: float) -> float:
        return 0.5 * np.sum(np.arctan(lambdas * u)) - q_stat * u / 2.0

    def _log_rho(u: float) -> float:
        return 0.25 * np.sum(np.log1p(lambdas ** 2 * u ** 2))

    def _integrand(u: float) -> float:
        if u == 0.0:
            return 0.0
        t = _theta(u)
        if abs(t) > 1e15:
            return 0.0
        return np.sin(t) / (u * np.exp(_log_rho(u)))

    if abs(q_stat) < 1e-10:
        # Near-zero test statistic: integrand nearly constant; single interval
        val, _ = scipy.integrate.quad(
            _integrand, 0.0, 100.0,
            epsabs=epsabs, epsrel=epsrel, limit=limit,
        )
        p = 0.5 + val / np.pi
        return float(np.clip(p, 0.0, 1.0))

    # Asymptotic half-period of sin(theta) oscillations: 2*pi / |q_stat|
    half_period = 2.0 * np.pi / abs(q_stat)

    # First asymptotic zero-crossing beyond u=0:
    #   theta(u) ≈ k*pi/2 - q*u/2 = 0  =>  u* = k*pi / q_stat
    u_first_zero = k * np.pi / abs(q_stat)

    # Upper integration limit (where the amplitude 1/rho(u) is negligible)
    u_max = max(200.0, u_first_zero + 50.0 * half_period)
    u_max = min(u_max, 1e5)

    # Breakpoints at the asymptotic zero-crossings of sin(theta) improve the
    # accuracy of adaptive Gauss-Kronrod on this oscillatory integrand.
    breakpoints = []
    u_zero = u_first_zero
    while u_zero < u_max:
        breakpoints.append(u_zero)
        u_zero += half_period
    if len(breakpoints) > limit - 10:
        breakpoints = breakpoints[: limit - 10]

    val, _ = scipy.integrate.quad(
        _integrand,
        0.0,
        u_max,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=max(limit, len(breakpoints) + 20),
        points=breakpoints if breakpoints else None,
    )

    p = 0.5 + val / np.pi
    return float(np.clip(p, 0.0, 1.0))


def generalized_chisq_pvalue(
    test_stat: float,
    eigenvalues: np.ndarray,
) -> float:
    """P(Q > test_stat) for Q = sum_j lambda_j chi^2_1 via Imhof integration.

    Filters near-zero eigenvalues whose absolute value is less than
    1e-10 * max(|eigenvalues|) before calling :func:`imhof_pvalue`.

    Parameters
    ----------
    test_stat : float
        Observed value of the test statistic.
    eigenvalues : np.ndarray, shape (J,)
        Eigenvalues (lambda_j) of the weighting matrix.

    Returns
    -------
    float
        Tail probability P(Q > test_stat) in [0, 1].
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    max_abs = np.max(np.abs(eigenvalues))
    if max_abs == 0.0:
        # Degenerate case: Q = 0 almost surely
        return 0.0 if test_stat > 0.0 else 1.0

    threshold = 1e-10 * max_abs
    lambdas_filtered = eigenvalues[np.abs(eigenvalues) >= threshold]

    if len(lambdas_filtered) == 0:
        return 0.0 if test_stat > 0.0 else 1.0

    return imhof_pvalue(test_stat, lambdas_filtered)
