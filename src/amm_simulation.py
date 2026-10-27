"""
AMM pool simulation for the separate-fee mechanism (Uniswap v3 type).

Paper: "Loss-versus-rebalancing with external proportional transaction costs"
Sections 2.1.1 – 2.1.4 (pages 3-6): Separate fee mechanism dynamics.
Section 3.1 (pages 10-11): Common formulation in total holdings.
Section 3.2 (pages 11-12): Benchmark and zero LVR (Theorem 3.2).

Key equations implemented:
  Reserve curve:        Y = f(X) = C * X^{-k}                           (Section 2.1.1)
  Internal price:       S = -f'(X) = k * C * X^{-(k+1)} = k*Y/X
  Fee accounts:         dXf = (τ/(1-τ)) dX↑                             (eq 6)
                        dYf = (τ/(1-τ)) S dX↓
  Total inventory:      Q = X + Xf                                       (Section 3.1)
  Total Q increments:   dQ↑ = dX↑/(1-τ),  dQ↓ = dX↓                    (eq 8)
  LP cash flow:         dC^LP = A dQ↓ − B dQ↑                           (eq 19)
  LP liquidation:       V^LP = C^LP + S^b Q                              (eq 21)
  Benchmark cash flow:  dC^R = S^b dQ↓ − S^a dQ↑                        (eq 22)
  Benchmark wealth:     V^R  = C^R  + S^b Q                              (eq 23)
  LVR:                  LVR^tc = V^R − V^LP = C^R − C^LP                 (Def 3.1)

# Custom — Context7 found no library equivalent (paper Section 2.1, Theorem 3.2)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from src.reserve_curve import GeometricMeanCurve
from src.skorokhod import run_skorokhod_path


@dataclass
class SeparateFeeSimResult:
    """Container for all simulation time-series."""

    # External price path
    S_star: NDArray[np.float64]      # shape (N+1,)

    # Pool active reserves
    X: NDArray[np.float64]           # risky active reserve
    Y: NDArray[np.float64]           # numeraire active reserve
    S: NDArray[np.float64]           # pool internal price

    # Fee accounts
    Xf: NDArray[np.float64]          # risky fee account
    Yf: NDArray[np.float64]          # numeraire fee account

    # Total risky inventory Q = X + Xf
    Q: NDArray[np.float64]

    # Jordan decomposition increments (cumulative)
    dX_up: NDArray[np.float64]       # pool buys risky (risky sold to pool, X increases)
    dX_down: NDArray[np.float64]     # pool sells risky (X decreases)

    # Total inventory increments
    dQ_up: NDArray[np.float64]       # = dX_up / (1-tau)
    dQ_down: NDArray[np.float64]     # = dX_down

    # LP numeraire account C^LP = Y + Yf
    C_LP: NDArray[np.float64]

    # Benchmark numeraire account C^R
    C_R: NDArray[np.float64]

    # Liquidation wealths
    V_LP: NDArray[np.float64]        # V^LP = C^LP + S^b * Q
    V_R: NDArray[np.float64]         # V^R  = C^R  + S^b * Q

    # LVR^tc = V^R - V^LP = C^R - C^LP
    LVR_tc: NDArray[np.float64]

    # Boundary regulators
    Z: NDArray[np.float64]           # Z_t = log(S_t / S*_t) in [-h, h]
    dL: NDArray[np.float64]          # lower-boundary regulator increments
    dU: NDArray[np.float64]          # upper-boundary regulator increments

    # Metadata
    curve: GeometricMeanCurve = field(repr=False)
    num_lower_hits: int = 0
    num_upper_hits: int = 0


def simulate_separate_fee_amm(
    S_star: NDArray[np.float64],
    curve: GeometricMeanCurve,
    X0: float,
    C_R0: float | None = None,
) -> SeparateFeeSimResult:
    """
    Simulate the separate-fee AMM (Uniswap v3 type) under continuous pure arbitrage.

    The pool's internal price is constrained to the no-arbitrage band
    [gamma * S*, S*/gamma] via a Skorokhod reflection (paper eq (4) / Proposition 2.3).

    Parameters
    ----------
    S_star : np.ndarray of shape (N+1,)
        External reference price path.
    curve : GeometricMeanCurve
        Reserve curve instance containing (k, C, tau, lam).
    X0 : float
        Initial active risky reserve X_0.
    C_R0 : float, optional
        Initial benchmark cash C^R_0. Defaults to C^LP_0 = Y_0 + Yf_0 = Y_0
        (same initial wealth as LP). Paper Section 3.2: C^R_0 = C^LP_0.

    Returns
    -------
    SeparateFeeSimResult
        Full time-series of all simulation quantities.
    """
    N = len(S_star) - 1
    tau = curve.tau
    h = curve.h
    k = curve.k
    C_const = curve.C

    # ── Initial state ──────────────────────────────────────────────────────────
    Y0 = curve.Y_from_X(X0)
    S0 = curve.S_from_X(X0)

    # Validate initial position inside the no-arbitrage band
    lb, ub = curve.no_arb_band(S_star[0])
    if not (lb * (1 - 1e-8) <= S0 <= ub * (1 + 1e-8)):
        raise ValueError(
            f"Initial S0={S0:.4g} outside no-arbitrage band [{lb:.4g}, {ub:.4g}]"
        )

    # ── Skorokhod reflection for internal price path ───────────────────────────
    # Computes S_t = exp(Z_t) * S*_t for all t, reflecting Z in [-h, h].
    # Paper Proposition 2.3 / eq (4): continuous pure-arbitrage reserve path.
    S_path, Z_path, dL_path, dU_path = run_skorokhod_path(S_star, S0, h)

    # ── Active reserves from internal price ────────────────────────────────────
    X_path = curve.X_from_S(S_path)     # X_t = (kC / S_t)^{1/(k+1)}
    Y_path = curve.Y_from_X(X_path)     # Y_t = C * X_t^{-k}

    # ── Allocate output arrays ─────────────────────────────────────────────────
    dX_up = np.zeros(N + 1)
    dX_down = np.zeros(N + 1)
    dQ_up = np.zeros(N + 1)
    dQ_down = np.zeros(N + 1)
    Xf = np.zeros(N + 1)
    Yf = np.zeros(N + 1)
    C_LP = np.zeros(N + 1)
    C_R = np.zeros(N + 1)

    # Initial conditions
    Xf[0] = 0.0
    Yf[0] = 0.0
    C_LP[0] = Y0 + 0.0   # Y + Yf at t=0
    C_R[0] = C_R0 if C_R0 is not None else C_LP[0]  # paper: C^R_0 = C^LP_0

    # ── Step through time ───────────────────────────────────────────────────────
    for t in range(1, N + 1):
        S_t = S_path[t]
        S_star_t = S_star[t]

        # Reserve changes
        delta_X = X_path[t] - X_path[t - 1]
        if delta_X > 0.0:
            # Pool buys risky asset (trader sells to pool): X increases
            dX_up[t] = delta_X
        else:
            # Pool sells risky asset (trader buys from pool): X decreases
            dX_down[t] = -delta_X

        # ── Fee accounts (paper eq (6)) ────────────────────────────────────────
        # dXf = (tau/(1-tau)) * dX↑  [risky fee when trader sells to pool]
        d_Xf = (tau / (1.0 - tau)) * dX_up[t]
        # dYf = (tau/(1-tau)) * S * dX↓  [numeraire fee when trader buys from pool]
        # Here S is the pool's internal price at time t (post-trade).
        d_Yf = (tau / (1.0 - tau)) * S_t * dX_down[t]

        Xf[t] = Xf[t - 1] + d_Xf
        Yf[t] = Yf[t - 1] + d_Yf

        # ── Total inventory increments (paper eq (8)) ──────────────────────────
        dQ_up[t] = dX_up[t] / (1.0 - tau)   # dQ↑ = dX↑ / (1-tau)
        dQ_down[t] = dX_down[t]              # dQ↓ = dX↓

        # ── LP numeraire (paper eq (9): d(Y + Yf) = A*dX↓ - S*dX↑) ───────────
        # Equivalently from eq (19): dC^LP = A*dQ↓ - B*dQ↑
        A_t = S_t / (1.0 - tau)             # pool ask
        B_t = (1.0 - tau) * S_t            # pool bid
        d_C_LP = A_t * dQ_down[t] - B_t * dQ_up[t]
        C_LP[t] = C_LP[t - 1] + d_C_LP

        # ── Benchmark numeraire (paper eq (22)) ────────────────────────────────
        # dC^R = S^b * dQ↓ - S^a * dQ↑
        Sb_t = curve.external_bid(S_star_t)   # (1-lambda)*S*
        Sa_t = curve.external_ask(S_star_t)   # S*/(1-lambda)
        d_C_R = Sb_t * dQ_down[t] - Sa_t * dQ_up[t]
        C_R[t] = C_R[t - 1] + d_C_R

    # ── Derived quantities ────────────────────────────────────────────────────
    Q = X_path + Xf
    Sb_path = curve.external_bid(S_star)      # (1-lambda)*S* at each t
    V_LP = C_LP + Sb_path * Q                 # paper eq (21)
    V_R = C_R + Sb_path * Q                   # paper eq (23)
    LVR_tc = V_R - V_LP                       # paper Definition 3.1

    num_lower = int(np.sum(dL_path > 0))
    num_upper = int(np.sum(dU_path > 0))

    return SeparateFeeSimResult(
        S_star=S_star,
        X=X_path,
        Y=Y_path,
        S=S_path,
        Xf=Xf,
        Yf=Yf,
        Q=Q,
        dX_up=dX_up,
        dX_down=dX_down,
        dQ_up=dQ_up,
        dQ_down=dQ_down,
        C_LP=C_LP,
        C_R=C_R,
        V_LP=V_LP,
        V_R=V_R,
        LVR_tc=LVR_tc,
        Z=Z_path,
        dL=dL_path,
        dU=dU_path,
        curve=curve,
        num_lower_hits=num_lower,
        num_upper_hits=num_upper,
    )


def generate_gbm_price(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    N: int,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """
    Generate a geometric Brownian motion external price path.

    S*_{t+1} = S*_t * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z)

    The GBM produces a continuous path (in the limit N -> inf),
    satisfying the paper's requirement of continuous external prices.

    Parameters
    ----------
    S0, mu, sigma : float
        Initial price, drift, volatility.
    T : float
        Time horizon.
    N : int
        Number of time steps.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray of shape (N+1,)
        Simulated external price path.
    """
    rng = np.random.default_rng(seed)
    dt = T / N
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(N)
    log_prices = np.empty(N + 1)
    log_prices[0] = np.log(S0)
    np.cumsum(increments, out=log_prices[1:])
    log_prices[1:] += log_prices[0]
    return np.exp(log_prices)


def generate_deterministic_path(
    S0: float,
    path_type: str,
    T: float,
    N: int,
    amplitude: float = 0.2,
) -> NDArray[np.float64]:
    """
    Generate deterministic external price paths for controlled verification.

    Parameters
    ----------
    S0 : float
        Initial price.
    path_type : str
        One of: 'monotone_up', 'monotone_down', 'oscillating'.
    T, N : float, int
        Time horizon and steps.
    amplitude : float
        Relative amplitude for oscillating path.

    Returns
    -------
    np.ndarray of shape (N+1,)
        Deterministic price path.
    """
    t = np.linspace(0.0, T, N + 1)
    if path_type == "monotone_up":
        return S0 * np.exp(0.5 * t)
    elif path_type == "monotone_down":
        return S0 * np.exp(-0.5 * t)
    elif path_type == "oscillating":
        return S0 * (1.0 + amplitude * np.sin(4.0 * np.pi * t / T))
    else:
        raise ValueError(f"Unknown path_type: {path_type!r}")
