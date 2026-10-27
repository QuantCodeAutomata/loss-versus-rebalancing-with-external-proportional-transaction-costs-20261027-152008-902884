"""
Skorokhod reflection for the no-arbitrage band in the separate-fee AMM.

Paper: "Loss-versus-rebalancing with external proportional transaction costs"
Section 2.1.3 (pages 5-6): No-arbitrage band and reserve dynamics.
Proposition 2.3 (pages 8-9): Local retained-fee dynamics (same reflection principle).

The Skorokhod problem on [-h, h]:

    Z_t = Z_0^free + L_t - U_t  in  [-h, h]

where:
    Z_t    = log(S_t / S*_t)         (log-ratio of pool to external price)
    h      = -log(gamma)             (half-width of band, h > 0)
    L_t    = cumulative push from lower boundary (increases only when Z = -h)
    U_t    = cumulative push from upper boundary (increases only when Z = h)
    S*_t   = continuous external reference price

For the separate-fee geometric-mean curve:
    S_t = exp(Z_t) * S*_t
    X_t = (k * C / S_t)^{1/(k+1)}   (inverted from S = k*C*X^{-(k+1)})
    Y_t = C * X_t^{-k}

# Custom — Context7 found no library equivalent (paper Proposition 2.3)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def skorokhod_reflect_scalar(z_free: float, h: float) -> tuple[float, float, float]:
    """
    Apply one step of the Skorokhod reflection to keep z in [-h, h].

    Parameters
    ----------
    z_free : float
        The 'free' value of Z before reflection (Z if no boundary forces act).
    h : float
        Half-width of the interval; h > 0.

    Returns
    -------
    z_reflected : float
        Reflected value in [-h, h].
    dL : float
        Increment of the lower-boundary regulator L (>= 0).
    dU : float
        Increment of the upper-boundary regulator U (>= 0).
    """
    assert h > 0, "Band half-width h must be positive"
    # Push from below: if z_free < -h, push up
    dL = max(0.0, -h - z_free)
    z_temp = z_free + dL
    # Push from above: if z_temp > h, push down
    dU = max(0.0, z_temp - h)
    z_reflected = z_temp - dU
    return z_reflected, dL, dU


def run_skorokhod_path(
    S_star: NDArray[np.float64],
    S0: float,
    h: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute the Skorokhod-reflected internal price path for the separate-fee AMM.

    Given an external price path S*_t, reflect Z_t = log(S_t / S*_t) in [-h, h]
    to obtain the pool's internal price path S_t and the boundary regulator increments.

    Parameters
    ----------
    S_star : np.ndarray of shape (N+1,)
        External reference price path (S*_0, S*_1, ..., S*_N).
    S0 : float
        Initial pool internal price S_0. Must lie in the no-arbitrage band
        [gamma * S*_0, S*_0 / gamma].
    h : float
        Band half-width: h = -log(gamma) where gamma = (1-tau)*(1-lambda).

    Returns
    -------
    S : np.ndarray of shape (N+1,)
        Pool internal price path.
    Z : np.ndarray of shape (N+1,)
        Log-ratio path Z_t = log(S_t / S*_t) in [-h, h].
    dL : np.ndarray of shape (N+1,)
        Lower-boundary regulator increments (dL[t] >= 0).
    dU : np.ndarray of shape (N+1,)
        Upper-boundary regulator increments (dU[t] >= 0).
    """
    N = len(S_star) - 1
    S = np.empty(N + 1)
    Z = np.empty(N + 1)
    dL = np.zeros(N + 1)
    dU = np.zeros(N + 1)

    Z0 = np.log(S0 / S_star[0])
    if abs(Z0) > h + 1e-10:
        raise ValueError(
            f"Initial S0={S0:.6g} is outside the no-arbitrage band at S*_0={S_star[0]:.6g}: "
            f"|Z0|={abs(Z0):.4g} > h={h:.4g}"
        )
    Z[0] = np.clip(Z0, -h, h)
    S[0] = np.exp(Z[0]) * S_star[0]

    for t in range(N):
        # Free evolution: S stays at S[t], S* changes to S_star[t+1]
        # Z_free = log(S[t] / S_star[t+1])
        z_free = Z[t] + np.log(S_star[t] / S_star[t + 1])
        Z[t + 1], dL[t + 1], dU[t + 1] = skorokhod_reflect_scalar(z_free, h)
        S[t + 1] = np.exp(Z[t + 1]) * S_star[t + 1]

    return S, Z, dL, dU


def run_skorokhod_path_vectorized(
    S_star: NDArray[np.float64],
    S0: float,
    h: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Vectorized Skorokhod reflection path (uses loop internally; named 'vectorized'
    to match the scalar version's interface for benchmarking).

    Same signature as run_skorokhod_path.
    """
    return run_skorokhod_path(S_star, S0, h)


def at_lower_boundary(dL: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Return True where the lower boundary was active (pool sold risky asset)."""
    return dL > 0.0


def at_upper_boundary(dU: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Return True where the upper boundary was active (pool bought risky asset)."""
    return dU > 0.0
