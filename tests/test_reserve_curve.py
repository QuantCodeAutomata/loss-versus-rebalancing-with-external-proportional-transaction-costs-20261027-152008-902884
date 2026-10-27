"""
Unit tests for src/reserve_curve.py.

Tests verify that the geometric-mean reserve curve correctly implements:
  - f(x) = C * x^{-k}  (paper Section 2.1.1)
  - S = k*Y/X = -f'(X)  (paper Section 2.1.1)
  - A = S/(1-tau), B = (1-tau)*S  (paper Section 2.1.1)
  - gamma = (1-tau)*(1-lambda), h = -log(gamma)  (paper eq (4))
  - No-arbitrage band: [gamma*S*, S*/gamma]  (paper eq (4))
"""

import numpy as np
import pytest

from src.reserve_curve import GeometricMeanCurve


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_curve(k=1.0, C=1.0, tau=0.003, lam=0.001):
    return GeometricMeanCurve(k=k, C=C, tau=tau, lam=lam)


# ── Reserve curve identity tests ──────────────────────────────────────────────

class TestCurveIdentities:
    """Tests for Y = f(X) = C * X^{-k} and S = k*Y/X."""

    def test_Y_from_X_k1(self):
        """k=1: Y = C/X (constant product)."""
        curve = make_curve(k=1.0, C=2.0)
        X = 1.5
        assert np.isclose(curve.Y_from_X(X), 2.0 / 1.5)

    def test_Y_from_X_k2(self):
        """k=2: Y = C/X^2."""
        curve = make_curve(k=2.0, C=3.0)
        X = 2.0
        assert np.isclose(curve.Y_from_X(X), 3.0 / 4.0)

    def test_curve_constant_preserved(self):
        """X^k * Y should equal C for any X."""
        curve = make_curve(k=1.0, C=5.0)
        X_vals = np.linspace(0.1, 10.0, 20)
        consts = curve.curve_constant(X_vals)
        np.testing.assert_allclose(consts, 5.0, rtol=1e-12)

    def test_S_equals_k_Y_over_X(self):
        """S = k*Y/X = -f'(X). Paper Section 2.1.1."""
        curve = make_curve(k=2.0, C=1.0)
        X = 1.3
        Y = curve.Y_from_X(X)
        S_direct = curve.S_from_X(X)
        S_ratio = 2.0 * Y / X
        assert np.isclose(S_direct, S_ratio, rtol=1e-12)

    def test_S_equals_negative_f_prime(self):
        """S = -f'(X). Paper: S := -f'(X)."""
        curve = make_curve(k=1.5, C=0.8)
        X = 0.9
        S_from_X = curve.S_from_X(X)
        neg_fprime = -curve.f_prime(X)
        assert np.isclose(S_from_X, neg_fprime, rtol=1e-12)

    def test_X_from_S_roundtrip(self):
        """X_from_S is the inverse of S_from_X."""
        curve = make_curve(k=1.0, C=1.0)
        X = 0.75
        S = curve.S_from_X(X)
        X_back = curve.X_from_S(S)
        assert np.isclose(X_back, X, rtol=1e-12)

    def test_f_double_prime_positive(self):
        """f''(X) > 0: f is strictly convex. Paper Section 2.1.1."""
        curve = make_curve(k=1.0, C=1.0)
        X_vals = np.linspace(0.1, 5.0, 50)
        assert np.all(curve.f_double_prime(X_vals) > 0)


# ── Execution price tests ──────────────────────────────────────────────────────

class TestExecutionPrices:
    """Tests for A, B and external bid/ask. Paper Section 2.1.1 / eq (1)."""

    def test_ask_price(self):
        """A = S/(1-tau). Pool ask: cost to buy from pool."""
        curve = make_curve(tau=0.003)
        S = 1.5
        A = curve.ask_price(S)
        assert np.isclose(A, S / (1 - 0.003), rtol=1e-12)

    def test_bid_price(self):
        """B = (1-tau)*S. Pool bid: received when selling to pool."""
        curve = make_curve(tau=0.003)
        S = 1.5
        B = curve.bid_price(S)
        assert np.isclose(B, (1 - 0.003) * S, rtol=1e-12)

    def test_bid_le_S_le_ask(self):
        """B <= S <= A for any S > 0, tau in (0,1). Paper: fee-inclusive spread."""
        curve = make_curve(tau=0.01)
        S = 2.0
        assert curve.bid_price(S) <= S <= curve.ask_price(S)

    def test_external_bid(self):
        """S^b = (1-lambda)*S*. Paper eq (1)."""
        curve = make_curve(lam=0.002)
        S_star = 1.0
        Sb = curve.external_bid(S_star)
        assert np.isclose(Sb, 0.998, rtol=1e-12)

    def test_external_ask(self):
        """S^a = S*/(1-lambda). Paper eq (1)."""
        curve = make_curve(lam=0.002)
        S_star = 1.0
        Sa = curve.external_ask(S_star)
        assert np.isclose(Sa, 1.0 / 0.998, rtol=1e-12)

    def test_external_spread_positive(self):
        """S^a > S^b for lambda > 0."""
        curve = make_curve(lam=0.005)
        S_star = 2.0
        assert curve.external_ask(S_star) > curve.external_bid(S_star)


# ── No-arbitrage band tests ────────────────────────────────────────────────────

class TestNoArbitrageBand:
    """Tests for gamma and no-arbitrage band. Paper eq (4)."""

    def test_gamma_formula(self):
        """gamma = (1-tau)*(1-lambda). Paper eq (4)."""
        curve = make_curve(tau=0.003, lam=0.001)
        expected = (1 - 0.003) * (1 - 0.001)
        assert np.isclose(curve.gamma, expected, rtol=1e-12)

    def test_h_positive(self):
        """h = -log(gamma) > 0 when gamma < 1."""
        curve = make_curve(tau=0.003, lam=0.001)
        assert curve.h > 0.0

    def test_band_lower_equals_gamma_S(self):
        """Lower band = gamma * S*. Paper eq (4)."""
        curve = make_curve(tau=0.003, lam=0.001)
        S_star = 1.5
        lb, ub = curve.no_arb_band(S_star)
        assert np.isclose(lb, curve.gamma * S_star, rtol=1e-12)

    def test_band_upper_equals_S_over_gamma(self):
        """Upper band = S* / gamma. Paper eq (4)."""
        curve = make_curve(tau=0.003, lam=0.001)
        S_star = 1.5
        lb, ub = curve.no_arb_band(S_star)
        assert np.isclose(ub, S_star / curve.gamma, rtol=1e-12)

    def test_band_width_increases_with_tau(self):
        """Larger tau → smaller gamma → wider band."""
        c1 = make_curve(tau=0.001, lam=0.001)
        c2 = make_curve(tau=0.01, lam=0.001)
        S_star = 1.0
        lb1, ub1 = c1.no_arb_band(S_star)
        lb2, ub2 = c2.no_arb_band(S_star)
        assert ub2 - lb2 > ub1 - lb1

    def test_boundary_condition_at_lower_band(self):
        """At S = gamma*S*, pool ask A = external bid S^b (paper eq (3))."""
        curve = make_curve(tau=0.003, lam=0.001)
        S_star = 1.0
        S = curve.gamma * S_star  # lower boundary
        A = curve.ask_price(S)
        Sb = curve.external_bid(S_star)
        assert np.isclose(A, Sb, rtol=1e-10)

    def test_boundary_condition_at_upper_band(self):
        """At S = S*/gamma, pool bid B = external ask S^a (paper eq (3))."""
        curve = make_curve(tau=0.003, lam=0.001)
        S_star = 1.0
        S = S_star / curve.gamma  # upper boundary
        B = curve.bid_price(S)
        Sa = curve.external_ask(S_star)
        assert np.isclose(B, Sa, rtol=1e-10)

    def test_tau_plus_lambda_zero_raises(self):
        """tau=0, lambda=0 should raise: Theorem 3.2 requires tau+lambda>0."""
        with pytest.raises(AssertionError):
            GeometricMeanCurve(k=1.0, C=1.0, tau=0.0, lam=0.0)


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_vectorized_inputs(self):
        """Functions handle numpy arrays correctly."""
        curve = make_curve()
        X = np.array([0.5, 1.0, 2.0, 4.0])
        Y = curve.Y_from_X(X)
        S = curve.S_from_X(X)
        assert Y.shape == X.shape
        assert S.shape == X.shape
        assert np.all(Y > 0)
        assert np.all(S > 0)

    def test_high_k(self):
        """k=5 curve: still maintains X^k * Y = C."""
        curve = make_curve(k=5.0, C=1.0)
        X = 1.2
        Y = curve.Y_from_X(X)
        assert np.isclose(X**5 * Y, 1.0, rtol=1e-12)

    def test_extreme_tau(self):
        """tau close to 1 still produces valid (narrow) band width."""
        curve = make_curve(tau=0.999, lam=0.0001)
        S_star = 1.0
        lb, ub = curve.no_arb_band(S_star)
        assert ub > lb > 0
