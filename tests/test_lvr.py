"""
Unit tests for src/lvr.py.

Verifies LVR metric computation and reporting functions.
Paper: "Loss-versus-rebalancing with external proportional transaction costs"
Definition 3.1 (page 11): LVR^tc = V^R - V^LP
Theorem 3.2 (page 11): LVR^tc = 0 under continuous prices.
"""

import numpy as np
import pytest

from src.amm_simulation import (
    generate_gbm_price,
    generate_deterministic_path,
    simulate_separate_fee_amm,
)
from src.lvr import LVRMetrics, compute_lvr_metrics, format_metrics_table
from src.reserve_curve import GeometricMeanCurve


def make_result(N=2000, tau=0.003, lam=0.001, sigma=0.3, seed=7, X0=1.0):
    curve = GeometricMeanCurve(k=1.0, C=1.0, tau=tau, lam=lam)
    S_star = generate_gbm_price(S0=1.0, mu=0.0, sigma=sigma, T=1.0, N=N, seed=seed)
    return simulate_separate_fee_amm(S_star=S_star, curve=curve, X0=X0)


class TestComputeLVRMetrics:
    def test_returns_LVRMetrics_instance(self):
        result = make_result()
        m = compute_lvr_metrics(result, "test")
        assert isinstance(m, LVRMetrics)

    def test_scenario_name_preserved(self):
        result = make_result()
        m = compute_lvr_metrics(result, "my_scenario")
        assert m.scenario_name == "my_scenario"

    def test_max_abs_lvr_is_zero_theorem32(self):
        """Theorem 3.2: max|LVR^tc| ~ 0 within numerical tolerance."""
        result = make_result(N=5000, seed=10)
        m = compute_lvr_metrics(result, "theorem32", tolerance=1e-8)
        assert m.lvr_is_zero, f"max|LVR|={m.max_abs_lvr:.2e} should be ~0"

    def test_max_abs_lvr_nonneg(self):
        result = make_result()
        m = compute_lvr_metrics(result, "test")
        assert m.max_abs_lvr >= 0.0

    def test_n_steps_matches(self):
        N = 3000
        result = make_result(N=N)
        m = compute_lvr_metrics(result, "test")
        assert m.n_steps == N

    def test_final_lvr_matches_result(self):
        result = make_result()
        m = compute_lvr_metrics(result, "test")
        assert np.isclose(m.final_lvr, result.LVR_tc[-1])

    def test_V_LP_terminal_matches(self):
        result = make_result()
        m = compute_lvr_metrics(result, "test")
        assert np.isclose(m.V_LP_terminal, result.V_LP[-1])

    def test_V_R_terminal_matches(self):
        result = make_result()
        m = compute_lvr_metrics(result, "test")
        assert np.isclose(m.V_R_terminal, result.V_R[-1])

    def test_lvr_is_zero_flag_false_below_tolerance(self):
        """lvr_is_zero=False when tolerance is unreasonably tight."""
        result = make_result(N=100, seed=5)
        m = compute_lvr_metrics(result, "test", tolerance=1e-300)
        # LVR is never exactly 0.0 in float arithmetic
        # lvr_is_zero should be False for impossible tolerance
        assert m.lvr_is_zero is False

    def test_boundary_hits_nonneg(self):
        result = make_result(N=2000, sigma=0.5, seed=3)
        m = compute_lvr_metrics(result, "test")
        assert m.n_lower_hits >= 0
        assert m.n_upper_hits >= 0

    def test_total_Q_turnover_nonneg(self):
        result = make_result()
        m = compute_lvr_metrics(result, "test")
        assert m.total_Q_turnover >= 0.0

    def test_boundary_cond_flat_price(self):
        """Flat price: no trading, boundary violation = 0."""
        curve = GeometricMeanCurve(k=1.0, C=1.0, tau=0.003, lam=0.001)
        S_star = np.ones(101)
        result = simulate_separate_fee_amm(S_star=S_star, curve=curve, X0=1.0)
        m = compute_lvr_metrics(result, "flat")
        assert m.boundary_cond_max_violation == 0.0
        assert m.boundary_cond_verified


class TestFormatMetricsTable:
    def test_returns_string(self):
        result = make_result()
        m = compute_lvr_metrics(result, "test")
        table = format_metrics_table([m])
        assert isinstance(table, str)

    def test_header_in_table(self):
        result = make_result()
        m = compute_lvr_metrics(result, "test")
        table = format_metrics_table([m])
        assert "Scenario" in table
        assert "tau" in table
        assert "LVR" in table

    def test_scenario_name_in_table(self):
        result = make_result()
        m = compute_lvr_metrics(result, "my_unique_scenario")
        table = format_metrics_table([m])
        assert "my_unique_scenario" in table

    def test_empty_list(self):
        table = format_metrics_table([])
        # Should return just header rows, not crash
        assert "Scenario" in table

    def test_multiple_metrics(self):
        r1 = make_result(seed=1)
        r2 = make_result(seed=2, tau=0.005)
        m1 = compute_lvr_metrics(r1, "scenario_1")
        m2 = compute_lvr_metrics(r2, "scenario_2")
        table = format_metrics_table([m1, m2])
        assert "scenario_1" in table
        assert "scenario_2" in table
