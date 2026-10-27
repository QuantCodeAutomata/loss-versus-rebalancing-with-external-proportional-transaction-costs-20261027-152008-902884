"""
Reserve curve implementation for the geometric-mean AMM with separate fees.

Paper: "Loss-versus-rebalancing with external proportional transaction costs"
Section 2.1.1 (pages 3-4): Reserve curve and execution prices.
Section 2.2.4 (pages 9-10): Geometric-mean market maker example.

The reserve curve for the separate-fee mechanism is:
    Y = f(X) = C * X^{-k}     (paper equation implied by Section 2.1.1 and 2.2.4)

Internal price:
    S = -f'(X) = k * C * X^{-(k+1)} = k * Y / X   (paper eq, Section 2.1.1)

Marginal execution prices:
    A = S / (1 - tau)    [pool ask: cost to buy from pool]    (paper eq, Section 2.1.1)
    B = (1 - tau) * S   [pool bid: received when selling to pool]  (paper eq, Section 2.1.1)

No-arbitrage band:
    gamma * S* <= S <= S* / gamma
    where gamma = (1 - tau) * (1 - lambda)              (paper eq (4))
"""

from __future__ import annotations

import numpy as np


class GeometricMeanCurve:
    """
    Geometric-mean reserve curve for the separate-fee AMM.

    Reserve curve: Y = f(X) = C * X^{-k}
    Internal price: S = k * C * X^{-(k+1)} = k * Y / X

    Parameters
    ----------
    k : float
        Geometric-mean exponent (k > 0). k=1 is the constant-product (Uniswap v2) curve.
    C : float
        Initial curve level (C > 0). Determines the product X^k * Y = C.
    tau : float
        Pool fee fraction in [0, 1). Paper eq (1) / Section 2.1.1.
    lam : float
        External proportional transaction cost parameter in [0, 1). Paper eq (1).
    """

    def __init__(self, k: float, C: float, tau: float, lam: float) -> None:
        assert k > 0, "k must be positive"
        assert C > 0, "C must be positive"
        assert 0 <= tau < 1, "tau must be in [0, 1)"
        assert 0 <= lam < 1, "lam must be in [0, 1)"
        assert tau + lam > 0, "Theorem 3.2 requires tau + lambda > 0"

        self.k = k
        self.C = C
        self.tau = tau
        self.lam = lam
        # gamma = (1 - tau) * (1 - lambda): no-arbitrage band parameter (paper eq (4))
        self.gamma = (1.0 - tau) * (1.0 - lam)
        # h = -log(gamma): half-width of the Skorokhod band (paper Proposition 2.3)
        self.h = -np.log(self.gamma)

    def Y_from_X(self, X: float | np.ndarray) -> float | np.ndarray:
        """Compute Y = f(X) = C * X^{-k}.  Paper Section 2.1.1."""
        return self.C * X ** (-self.k)

    def X_from_S(self, S: float | np.ndarray) -> float | np.ndarray:
        """
        Invert S = k * C * X^{-(k+1)} to get X = (k * C / S)^{1 / (k+1)}.

        Used to recover active risky reserves from the internal price.
        """
        return (self.k * self.C / S) ** (1.0 / (self.k + 1.0))

    def S_from_X(self, X: float | np.ndarray) -> float | np.ndarray:
        """Compute internal price S = k * C * X^{-(k+1)} = k * Y / X."""
        return self.k * self.C * X ** (-(self.k + 1.0))

    def ask_price(self, S: float | np.ndarray) -> float | np.ndarray:
        """
        Pool ask price A = S / (1 - tau).

        Cost (in numeraire) per unit of risky asset bought from the pool.
        Paper Section 2.1.1: A := lim_{z->0} w(z)/z = S / (1 - tau).
        """
        return S / (1.0 - self.tau)

    def bid_price(self, S: float | np.ndarray) -> float | np.ndarray:
        """
        Pool bid price B = (1 - tau) * S.

        Numeraire received per unit of risky asset sold to the pool.
        Paper Section 2.1.1: B := lim_{z->0} v(z)/z = (1 - tau) * S.
        """
        return (1.0 - self.tau) * S

    def external_bid(self, S_star: float | np.ndarray) -> float | np.ndarray:
        """External bid price S^b = (1 - lambda) * S*. Paper eq (1)."""
        return (1.0 - self.lam) * S_star

    def external_ask(self, S_star: float | np.ndarray) -> float | np.ndarray:
        """External ask price S^a = S* / (1 - lambda). Paper eq (1)."""
        return S_star / (1.0 - self.lam)

    def no_arb_band(self, S_star: float | np.ndarray) -> tuple:
        """
        No-arbitrage band boundaries [gamma * S*, S* / gamma].

        Paper eq (4): gamma * S* <= S <= S* / gamma.
        Returns (lower_bound, upper_bound).
        """
        return self.gamma * S_star, S_star / self.gamma

    def verify_no_arb(self, S: float | np.ndarray, S_star: float | np.ndarray) -> np.ndarray:
        """Return True where S lies within the no-arbitrage band."""
        lb, ub = self.no_arb_band(S_star)
        return (S >= lb * (1 - 1e-12)) & (S <= ub * (1 + 1e-12))

    def curve_constant(self, X: float | np.ndarray) -> float | np.ndarray:
        """Compute X^k * Y = C (should remain constant for separate fees)."""
        return X ** self.k * self.Y_from_X(X)

    def f_prime(self, X: float | np.ndarray) -> float | np.ndarray:
        """f'(X) = -k * C * X^{-(k+1)} = -S. Paper: S = -f'(X)."""
        return -self.k * self.C * X ** (-(self.k + 1.0))

    def f_double_prime(self, X: float | np.ndarray) -> float | np.ndarray:
        """f''(X) = k*(k+1)*C * X^{-(k+2)}. Used in frictionless LVR formula."""
        return self.k * (self.k + 1.0) * self.C * X ** (-(self.k + 2.0))
