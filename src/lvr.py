"""
LVR (Loss-Versus-Rebalancing) computation and verification metrics.

Paper: "Loss-versus-rebalancing with external proportional transaction costs"
Definition 3.1 (page 11): LVR^tc = V^R - V^LP = C^R - C^LP
Theorem 3.2 (page 11): Zero LVR under continuous prices and pure arbitrage.

The experiment verifies that LVR^tc = 0 for the separate-fee AMM under
continuous external prices and pure arbitrage when tau + lambda > 0.

# Custom — Context7 found no library equivalent (paper Theorem 3.2 / Definition 3.1)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.amm_simulation import SeparateFeeSimResult


@dataclass
class LVRMetrics:
    """Summary metrics for a single LVR simulation run."""

    scenario_name: str
    tau: float
    lam: float
    k: float

    # Zero-LVR verification (Theorem 3.2)
    max_abs_lvr: float
    final_lvr: float
    mean_abs_lvr: float
    max_relative_lvr: float       # max |LVR| / |V_LP| (relative to pool value)
    lvr_is_zero: bool             # True if |LVR| < tolerance throughout

    # Simulation statistics
    n_steps: int
    n_lower_hits: int             # times pool sold risky (lower boundary)
    n_upper_hits: int             # times pool bought risky (upper boundary)
    total_Q_turnover: float       # total variation of Q

    # Terminal values
    V_LP_terminal: float
    V_R_terminal: float
    Q_terminal: float

    # Boundary verification (paper eq 5)
    boundary_cond_max_violation: float  # max |(A - Sb)*dX_down| / S at lower bd
    boundary_cond_verified: bool

    # Numerical precision
    numerical_precision: float    # machine epsilon scale


def compute_lvr_metrics(
    result: SeparateFeeSimResult,
    scenario_name: str,
    tolerance: float = 1e-8,
) -> LVRMetrics:
    """
    Compute LVR verification metrics from a simulation result.

    The primary verification is Theorem 3.2: LVR^tc = V^R - V^LP = 0
    for all t under continuous prices and pure arbitrage.

    Parameters
    ----------
    result : SeparateFeeSimResult
        Output from simulate_separate_fee_amm().
    scenario_name : str
        Human-readable scenario label for reporting.
    tolerance : float
        Absolute tolerance for declaring LVR "zero".

    Returns
    -------
    LVRMetrics
        Summary metrics for this simulation run.
    """
    lvr = result.LVR_tc

    # Relative LVR: |LVR| / |V_LP| (avoids division by zero)
    safe_V_LP = np.where(np.abs(result.V_LP) > 1e-15, np.abs(result.V_LP), 1e-15)
    rel_lvr = np.abs(lvr) / safe_V_LP

    # Boundary condition verification (paper Proposition 2.3):
    # At the lower Skorokhod boundary (dL > 0): S = γS* exactly, so
    # A = S/(1-tau) = (1-lam)*S* = Sb.  Use dL > 0 to select only true lower
    # boundary timesteps (not interior timesteps where S also happens to increase).
    tau = result.curve.tau
    S = result.S
    S_star = result.S_star
    A = S / (1.0 - tau)
    Sb = result.curve.external_bid(S_star)

    lower_mask = result.dL > 0   # Skorokhod lower-regulator active ↔ S = γS*
    if np.any(lower_mask):
        bc_lower = np.abs(A[lower_mask] - Sb[lower_mask]) / S[lower_mask]
        bc_violation = float(np.max(bc_lower))
    else:
        bc_violation = 0.0

    # Total variation of Q
    Q = result.Q
    total_tv = float(np.sum(np.abs(np.diff(Q))))

    return LVRMetrics(
        scenario_name=scenario_name,
        tau=result.curve.tau,
        lam=result.curve.lam,
        k=result.curve.k,
        max_abs_lvr=float(np.max(np.abs(lvr))),
        final_lvr=float(lvr[-1]),
        mean_abs_lvr=float(np.mean(np.abs(lvr))),
        max_relative_lvr=float(np.max(rel_lvr)),
        lvr_is_zero=bool(np.max(np.abs(lvr)) < tolerance),
        n_steps=len(lvr) - 1,
        n_lower_hits=result.num_lower_hits,
        n_upper_hits=result.num_upper_hits,
        total_Q_turnover=total_tv,
        V_LP_terminal=float(result.V_LP[-1]),
        V_R_terminal=float(result.V_R[-1]),
        Q_terminal=float(result.Q[-1]),
        boundary_cond_max_violation=bc_violation,
        boundary_cond_verified=bc_violation < 1e-10,
        numerical_precision=float(np.finfo(np.float64).eps),
    )


def format_metrics_table(metrics_list: list[LVRMetrics]) -> str:
    """Format LVR metrics as a Markdown table for RESULTS.md."""
    header = (
        "| Scenario | tau | lambda | max|LVR| | final LVR | "
        "max rel. LVR | LVR=0? | Lower hits | Upper hits | Boundary OK? |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for m in metrics_list:
        lvr_zero = "✓" if m.lvr_is_zero else "✗"
        bc_ok = "✓" if m.boundary_cond_verified else "✗"
        rows.append(
            f"| {m.scenario_name} | {m.tau:.4f} | {m.lam:.4f} | "
            f"{m.max_abs_lvr:.2e} | {m.final_lvr:.2e} | "
            f"{m.max_relative_lvr:.2e} | {lvr_zero} | "
            f"{m.n_lower_hits} | {m.n_upper_hits} | {bc_ok} |"
        )
    return header + "\n".join(rows)
