"""
Unit tests for src/amm_simulation.py.

Verifies that the separate-fee AMM simulation correctly:
  - Maintains reserve curve identity Y = C * X^{-k}
  - Computes fee accounts correctly (paper eq (6))
  - Computes total inventory Q = X + Xf
  - Computes cash flows C_LP and C_R (paper eq (19), (22))
  - Produces LVR^tc = V_R - V_LP ≈ 0 (paper Theorem 3.2)
"""

import numpy as np
import pytest

from src.amm_simulation import (
    SeparateFeeSimResult,
    generate_deterministic_path,
    generate_gbm_price,
    simulate_separate_fee_amm,
)
from src.reserve_curve import GeometricMeanCurve


# ── Helper factories ───────────────────────────────────────────────────────────

def make_curve(k=1.0, C=1.0, tau=0.003, lam=0.001):
    return GeometricMeanCurve(k=k, C=C, tau=tau, lam=lam)


def make_flat_S_star(N=100, S0=1.0):
    """Flat external price path — no trading should occur."""
    return np.ones(N + 1) * S0


def make_gbm(N=1000, S0=1.0, sigma=0.3, seed=42):
    return generate_gbm_price(S0=S0, mu=0.0, sigma=sigma, T=1.0, N=N, seed=seed)


def run_default(N=1000, sigma=0.3, tau=0.003, lam=0.001, k=1.0, C=1.0, X0=1.0, seed=42):
    curve = make_curve(k=k, C=C, tau=tau, lam=lam)
    S_star = make_gbm(N=N, sigma=sigma, seed=seed)
    return simulate_separate_fee_amm(S_star=S_star, curve=curve, X0=X0)


# ── Output shape tests ─────────────────────────────────────────────────────────

class TestOutputShapes:
    def test_all_arrays_correct_length(self):
        N = 200
        r = run_default(N=N)
        for attr in ["S_star", "X", "Y", "S", "Xf", "Yf", "Q",
                     "dX_up", "dX_down", "dQ_up", "dQ_down",
                     "C_LP", "C_R", "V_LP", "V_R", "LVR_tc", "Z", "dL", "dU"]:
            arr = getattr(r, attr)
            assert len(arr) == N + 1, f"Array '{attr}' has length {len(arr)}, expected {N+1}"


# ── Reserve curve identity ─────────────────────────────────────────────────────

class TestReserveCurveIdentity:
    def test_X_k_times_Y_equals_C(self):
        """X^k * Y = C for all active reserves (paper Section 2.1.1)."""
        r = run_default()
        consts = r.X ** r.curve.k * r.Y
        np.testing.assert_allclose(consts, r.curve.C, rtol=1e-10,
                                   err_msg="X^k * Y must equal C throughout")

    def test_S_equals_k_Y_over_X(self):
        """S = k * Y / X throughout (paper Section 2.1.1)."""
        r = run_default()
        S_check = r.curve.k * r.Y / r.X
        np.testing.assert_allclose(r.S, S_check, rtol=1e-10)


# ── Initial conditions ────────────────────────────────────────────────────────

class TestInitialConditions:
    def test_fee_accounts_zero_at_t0(self):
        """Xf[0] = 0, Yf[0] = 0 (no fees collected at inception)."""
        r = run_default()
        assert r.Xf[0] == 0.0
        assert r.Yf[0] == 0.0

    def test_Q_at_t0(self):
        """Q[0] = X[0] + Xf[0] = X0.

        Use X0=1.0 (equilibrium for k=1, C=1, S*_0=1.0) so initial internal price
        S0 = k*C*X0^{-(k+1)} = 1.0 lies within the no-arbitrage band.
        """
        X0 = 1.0
        r = run_default(X0=X0)
        assert np.isclose(r.Q[0], X0)

    def test_C_LP_C_R_equal_at_t0(self):
        """C_LP[0] = C_R[0] = Y[0] (paper: same initial cash)."""
        r = run_default()
        assert np.isclose(r.C_LP[0], r.C_R[0], rtol=1e-12)


# ── Fee account dynamics ───────────────────────────────────────────────────────

class TestFeeAccounts:
    def test_Xf_nondecreasing(self):
        """Xf is non-decreasing (only receives, never spends). Paper eq (6)."""
        r = run_default()
        assert np.all(np.diff(r.Xf) >= -1e-15)

    def test_Yf_nondecreasing(self):
        """Yf is non-decreasing (only receives, never spends). Paper eq (6)."""
        r = run_default()
        assert np.all(np.diff(r.Yf) >= -1e-15)

    def test_Q_equals_X_plus_Xf(self):
        """Q = X + Xf at every time step."""
        r = run_default()
        np.testing.assert_allclose(r.Q, r.X + r.Xf, rtol=1e-12)


# ── Inventory increment decomposition ─────────────────────────────────────────

class TestInventoryDecomposition:
    def test_dX_up_down_nonneg(self):
        """dX_up and dX_down are non-negative."""
        r = run_default()
        assert np.all(r.dX_up >= -1e-15)
        assert np.all(r.dX_down >= -1e-15)

    def test_dQ_up_equals_dX_up_over_one_minus_tau(self):
        """dQ_up = dX_up / (1 - tau). Paper eq (8)."""
        r = run_default(tau=0.003)
        tau = 0.003
        np.testing.assert_allclose(r.dQ_up, r.dX_up / (1 - tau), rtol=1e-12)

    def test_dQ_down_equals_dX_down(self):
        """dQ_down = dX_down. Paper eq (8)."""
        r = run_default()
        np.testing.assert_allclose(r.dQ_down, r.dX_down, rtol=1e-12)


# ── Flat price path (no trading) ───────────────────────────────────────────────

class TestFlatPrice:
    def test_no_trading_flat_price(self):
        """Flat S* → no boundary hits → no trading → Q constant."""
        curve = make_curve()
        S_star = make_flat_S_star(N=100)
        r = simulate_separate_fee_amm(S_star=S_star, curve=curve, X0=1.0)
        np.testing.assert_allclose(r.Q, r.Q[0], rtol=1e-12,
                                   err_msg="Q must be constant when S* is flat")

    def test_no_fee_accumulation_flat(self):
        """Flat S* → no fee accumulation."""
        curve = make_curve()
        S_star = make_flat_S_star(N=100)
        r = simulate_separate_fee_amm(S_star=S_star, curve=curve, X0=1.0)
        np.testing.assert_allclose(r.Xf, 0.0, atol=1e-15)
        np.testing.assert_allclose(r.Yf, 0.0, atol=1e-15)

    def test_LVR_zero_flat_price(self):
        """Flat S* → LVR = 0 trivially."""
        curve = make_curve()
        S_star = make_flat_S_star(N=100)
        r = simulate_separate_fee_amm(S_star=S_star, curve=curve, X0=1.0)
        np.testing.assert_allclose(r.LVR_tc, 0.0, atol=1e-12)


# ── Zero LVR (Theorem 3.2) ─────────────────────────────────────────────────────

class TestZeroLVR:
    """Central tests for Theorem 3.2: LVR^tc = 0 under continuous prices."""

    def test_LVR_zero_gbm_base_case(self):
        """Theorem 3.2: LVR^tc ≈ 0 for GBM path (tau=0.003, lambda=0.001)."""
        r = run_default(N=10000, tau=0.003, lam=0.001, sigma=0.4, seed=0)
        np.testing.assert_allclose(r.LVR_tc, 0.0, atol=1e-8,
                                   err_msg="Theorem 3.2: LVR^tc must be zero")

    def test_LVR_zero_tau_only(self):
        """LVR^tc = 0 when tau > 0, lambda = 0."""
        r = run_default(N=5000, tau=0.005, lam=0.0, sigma=0.3, seed=1)
        np.testing.assert_allclose(r.LVR_tc, 0.0, atol=1e-8)

    def test_LVR_zero_lambda_only(self):
        """LVR^tc = 0 when tau = 0, lambda > 0 (external spread alone suffices)."""
        r = run_default(N=5000, tau=0.0, lam=0.005, sigma=0.3, seed=2)
        np.testing.assert_allclose(r.LVR_tc, 0.0, atol=1e-8)

    def test_LVR_zero_monotone_up(self):
        """LVR^tc = 0 on monotone increasing price path."""
        N = 1000
        S_star = generate_deterministic_path(S0=1.0, path_type="monotone_up", T=1.0, N=N)
        curve = make_curve(tau=0.003, lam=0.001)
        r = simulate_separate_fee_amm(S_star=S_star, curve=curve, X0=1.0)
        np.testing.assert_allclose(r.LVR_tc, 0.0, atol=1e-12)

    def test_LVR_zero_monotone_down(self):
        """LVR^tc = 0 on monotone decreasing price path."""
        N = 1000
        S_star = generate_deterministic_path(S0=1.0, path_type="monotone_down", T=1.0, N=N)
        curve = make_curve(tau=0.003, lam=0.001)
        r = simulate_separate_fee_amm(S_star=S_star, curve=curve, X0=1.0)
        np.testing.assert_allclose(r.LVR_tc, 0.0, atol=1e-12)

    def test_LVR_zero_oscillating(self):
        """LVR^tc = 0 on oscillating price path (multiple boundary hits)."""
        N = 2000
        S_star = generate_deterministic_path(S0=1.0, path_type="oscillating", T=1.0, N=N)
        curve = make_curve(tau=0.003, lam=0.001)
        r = simulate_separate_fee_amm(S_star=S_star, curve=curve, X0=1.0)
        np.testing.assert_allclose(r.LVR_tc, 0.0, atol=1e-12)

    def test_LVR_zero_large_fees(self):
        """LVR^tc = 0 even with larger fees (tau=0.01, lambda=0.01)."""
        r = run_default(N=5000, tau=0.01, lam=0.01, sigma=0.3, seed=3)
        np.testing.assert_allclose(r.LVR_tc, 0.0, atol=1e-8)

    def test_LVR_equals_C_R_minus_C_LP(self):
        """V_R - V_LP = C_R - C_LP (Q cancels exactly). Paper Definition 3.1.

        The difference V_R - V_LP = (C_R + S_b*Q) - (C_LP + S_b*Q) = C_R - C_LP.
        Tolerance is atol=1e-14 to accommodate floating-point cancellation at ~machine epsilon.
        """
        r = run_default(N=2000, seed=10)
        np.testing.assert_allclose(r.LVR_tc, r.C_R - r.C_LP, atol=1e-14)


# ── Wealth non-negativity ──────────────────────────────────────────────────────

class TestWealthNonNeg:
    def test_Q_positive(self):
        """Q = X + Xf > 0 always (pool cannot go negative)."""
        r = run_default(N=5000, sigma=0.5, seed=99)
        assert np.all(r.Q > 0), "Q must remain positive throughout"

    def test_V_LP_positive(self):
        """V_LP > 0: pool liquidation wealth should remain positive."""
        r = run_default(N=5000, sigma=0.5, seed=99)
        assert np.all(r.V_LP > 0)


# ── Price path generators ─────────────────────────────────────────────────────

class TestPriceGenerators:
    def test_gbm_length(self):
        N = 500
        S = generate_gbm_price(S0=1.0, mu=0.0, sigma=0.2, T=1.0, N=N, seed=0)
        assert len(S) == N + 1

    def test_gbm_positive(self):
        S = generate_gbm_price(S0=2.0, mu=0.0, sigma=0.3, T=1.0, N=1000, seed=0)
        assert np.all(S > 0)

    def test_gbm_initial_price(self):
        S = generate_gbm_price(S0=3.0, mu=0.0, sigma=0.2, T=1.0, N=100, seed=0)
        assert np.isclose(S[0], 3.0)

    def test_gbm_reproducible(self):
        S1 = generate_gbm_price(S0=1.0, mu=0.0, sigma=0.2, T=1.0, N=100, seed=42)
        S2 = generate_gbm_price(S0=1.0, mu=0.0, sigma=0.2, T=1.0, N=100, seed=42)
        np.testing.assert_array_equal(S1, S2)

    def test_deterministic_monotone_up(self):
        N = 100
        S = generate_deterministic_path(S0=1.0, path_type="monotone_up", T=1.0, N=N)
        assert np.all(np.diff(S) >= 0)

    def test_deterministic_monotone_down(self):
        N = 100
        S = generate_deterministic_path(S0=1.0, path_type="monotone_down", T=1.0, N=N)
        assert np.all(np.diff(S) <= 0)

    def test_deterministic_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown path_type"):
            generate_deterministic_path(S0=1.0, path_type="zigzag", T=1.0, N=100)
