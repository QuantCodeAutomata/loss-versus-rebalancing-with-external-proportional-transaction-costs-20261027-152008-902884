"""
Unit tests for src/skorokhod.py.

Verifies the Skorokhod reflection correctly maintains Z_t in [-h, h]
and produces the correct boundary regulator increments.

Paper: "Loss-versus-rebalancing with external proportional transaction costs"
Proposition 2.3 (pages 8-9): Reflected dynamics.
"""

import numpy as np
import pytest

from src.skorokhod import (
    at_lower_boundary,
    at_upper_boundary,
    run_skorokhod_path,
    skorokhod_reflect_scalar,
)


class TestSkorokhodScalar:
    """Tests for the scalar Skorokhod reflection step."""

    def test_interior_no_reflection(self):
        """No reflection needed when z_free is already in (-h, h)."""
        z, dL, dU = skorokhod_reflect_scalar(z_free=0.0, h=1.0)
        assert np.isclose(z, 0.0)
        assert dL == 0.0
        assert dU == 0.0

    def test_lower_boundary_hit(self):
        """z_free < -h: lower boundary regulator dL pushes z up to -h.

        dL = max(0, -h - z_free) = -(-0.5) + (-0.8 correction) = -0.5 - (-0.8) = 0.3
        """
        h = 0.5
        z_free = -0.8  # below -h = -0.5
        z, dL, dU = skorokhod_reflect_scalar(z_free, h)
        assert np.isclose(z, -h)
        assert np.isclose(dL, -h - z_free)   # = -0.5 - (-0.8) = 0.3
        assert np.isclose(dL, 0.3)
        assert dU == 0.0

    def test_upper_boundary_hit(self):
        """z_free > h: upper boundary regulator dU pushes z down to h."""
        h = 0.5
        z_free = 0.9  # above h
        z, dL, dU = skorokhod_reflect_scalar(z_free, h)
        assert np.isclose(z, h)
        assert dL == 0.0
        assert np.isclose(dU, z_free - h)

    def test_result_always_in_band(self):
        """Reflected z always in [-h, h] for arbitrary z_free."""
        h = 1.0
        for z_free in [-5.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0]:
            z, dL, dU = skorokhod_reflect_scalar(z_free, h)
            assert -h - 1e-12 <= z <= h + 1e-12, f"z={z} outside band for z_free={z_free}"

    def test_non_negative_regulators(self):
        """dL and dU are always non-negative."""
        h = 0.3
        for z_free in np.linspace(-2.0, 2.0, 20):
            z, dL, dU = skorokhod_reflect_scalar(z_free, h)
            assert dL >= 0.0
            assert dU >= 0.0

    def test_at_boundary_no_regulator(self):
        """z_free exactly at -h: no regulator needed."""
        h = 0.5
        z, dL, dU = skorokhod_reflect_scalar(-h, h)
        assert np.isclose(z, -h)
        assert dL == 0.0
        assert dU == 0.0

    def test_h_positive_assertion(self):
        """h <= 0 should raise AssertionError."""
        with pytest.raises(AssertionError):
            skorokhod_reflect_scalar(0.0, h=-0.1)


class TestSkorokhodPath:
    """Tests for the full Skorokhod path simulation."""

    def _make_gbm(self, N=1000, S0=1.0, sigma=0.3, seed=0):
        rng = np.random.default_rng(seed)
        dt = 1.0 / N
        logS = np.zeros(N + 1)
        logS[0] = np.log(S0)
        logS[1:] = logS[0] + np.cumsum(-0.5 * sigma**2 * dt + sigma * np.sqrt(dt) * rng.standard_normal(N))
        return np.exp(logS)

    def test_Z_stays_in_band(self):
        """Z_t = log(S_t/S*_t) must stay in [-h, h] at all times."""
        S_star = self._make_gbm(N=500, sigma=0.5, seed=1)
        h = 0.1
        S, Z, dL, dU = run_skorokhod_path(S_star, S0=S_star[0], h=h)
        assert np.all(Z >= -h - 1e-10), f"Z below -h: min={Z.min():.6g}"
        assert np.all(Z <= h + 1e-10), f"Z above h: max={Z.max():.6g}"

    def test_S_in_no_arb_band(self):
        """S_t in [exp(-h)*S*, exp(h)*S*] = [gamma*S*, S*/gamma] for all t."""
        S_star = self._make_gbm(N=500, sigma=0.3, seed=2)
        h = 0.05
        S, Z, dL, dU = run_skorokhod_path(S_star, S0=S_star[0], h=h)
        lb = np.exp(-h) * S_star
        ub = np.exp(h) * S_star
        assert np.all(S >= lb * (1 - 1e-9))
        assert np.all(S <= ub * (1 + 1e-9))

    def test_regulators_nonneg_and_nondecreasing(self):
        """dL, dU >= 0; cumL, cumU are non-decreasing."""
        S_star = self._make_gbm(N=300, sigma=0.5, seed=3)
        h = 0.08
        S, Z, dL, dU = run_skorokhod_path(S_star, S0=S_star[0], h=h)
        assert np.all(dL >= 0)
        assert np.all(dU >= 0)
        assert np.all(np.cumsum(dL) >= 0)
        assert np.all(np.cumsum(dU) >= 0)

    def test_dL_dU_mutually_singular(self):
        """dL and dU cannot both be positive at the same time step."""
        S_star = self._make_gbm(N=1000, sigma=0.6, seed=4)
        h = 0.05
        S, Z, dL, dU = run_skorokhod_path(S_star, S0=S_star[0], h=h)
        simultaneous = (dL > 0) & (dU > 0)
        assert not np.any(simultaneous), "dL and dU must be mutually singular"

    def test_dL_only_at_lower_boundary(self):
        """dL > 0 only when Z reaches -h (lower boundary)."""
        S_star = self._make_gbm(N=500, sigma=0.4, seed=5)
        h = 0.05
        S, Z, dL, dU = run_skorokhod_path(S_star, S0=S_star[0], h=h)
        lower_active = dL > 0
        if np.any(lower_active):
            assert np.all(np.abs(Z[lower_active] + h) < 1e-10)

    def test_dU_only_at_upper_boundary(self):
        """dU > 0 only when Z reaches +h (upper boundary)."""
        S_star = self._make_gbm(N=500, sigma=0.4, seed=6)
        h = 0.05
        S, Z, dL, dU = run_skorokhod_path(S_star, S0=S_star[0], h=h)
        upper_active = dU > 0
        if np.any(upper_active):
            assert np.all(np.abs(Z[upper_active] - h) < 1e-10)

    def test_initial_S_matches(self):
        """S[0] must equal S0."""
        S_star = self._make_gbm(N=100, sigma=0.2, seed=7)
        h = 0.1
        S0 = S_star[0]
        S, Z, dL, dU = run_skorokhod_path(S_star, S0=S0, h=h)
        assert np.isclose(S[0], S0, rtol=1e-12)

    def test_monotone_up_price(self):
        """Monotone increasing S* drives Z downward, hitting the lower boundary.

        Free evolution: z_free = Z[t] + log(S*[t] / S*[t+1]).
        When S* increases, S*[t]/S*[t+1] < 1, so log(...) < 0 → z_free decreases.
        The lower boundary (-h) is hit first; dL > 0 (pool sells risky asset to
        arbitrageurs who push S upward to track the rising S*).
        """
        N = 500
        t = np.linspace(0, 1, N + 1)
        S_star = np.exp(2.0 * t)  # strongly increasing
        h = 0.2
        S, Z, dL, dU = run_skorokhod_path(S_star, S0=S_star[0], h=h)
        # Lower boundary should be hit repeatedly when S* rises
        assert np.any(dL > 0), "Monotone-up path must hit lower boundary (dL > 0)"

    def test_flat_price_no_trading(self):
        """Constant S* → no trading (Z stays at initial value, no boundary hits)."""
        N = 100
        S_star = np.ones(N + 1) * 1.0
        h = 0.1
        S0 = 1.0
        S, Z, dL, dU = run_skorokhod_path(S_star, S0=S0, h=h)
        assert np.all(dL == 0)
        assert np.all(dU == 0)
        np.testing.assert_allclose(S, S0, rtol=1e-12)

    def test_outside_band_raises(self):
        """S0 outside the no-arbitrage band should raise ValueError."""
        N = 10
        S_star = np.ones(N + 1)
        h = 0.01  # very narrow band
        S0_outside = 5.0  # way outside
        with pytest.raises(ValueError, match="outside the no-arbitrage band"):
            run_skorokhod_path(S_star, S0=S0_outside, h=h)


class TestBoundaryHelpers:
    """Tests for at_lower_boundary and at_upper_boundary helpers."""

    def test_at_lower_boundary(self):
        dL = np.array([0.0, 0.1, 0.0, 0.05, 0.0])
        expected = np.array([False, True, False, True, False])
        np.testing.assert_array_equal(at_lower_boundary(dL), expected)

    def test_at_upper_boundary(self):
        dU = np.array([0.0, 0.0, 0.2, 0.0, 0.01])
        expected = np.array([False, False, True, False, True])
        np.testing.assert_array_equal(at_upper_boundary(dU), expected)
