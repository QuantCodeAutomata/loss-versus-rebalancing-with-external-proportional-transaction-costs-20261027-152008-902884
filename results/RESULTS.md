# Experiment 1 Results: Zero LVR for Separate Fee Mechanisms

## Overview
**Paper**: Loss-versus-rebalancing with external proportional transaction costs
**Theorem 3.2** (pages 11-12): Under continuous external prices and pure arbitrage,
LVR^tc = V^R - V^LP = 0 for both fee mechanisms when tau + lambda > 0.

This experiment numerically verifies Theorem 3.2 for the **separate-fee mechanism**
(Uniswap v3 type) across multiple price paths and parameter configurations.

## Methodology
- Reserve curve: f(x) = C * x^{-k} (geometric mean, paper Section 2.2.4)
- Skorokhod reflection: reflects Z_t = log(S_t / S*_t) in [-h, h], h = -log(gamma)
- Internal price: S = k*C*X^{-(k+1)}; X recovered from S at each step
- Fee accounts: dXf = (tau/(1-tau))*dX_up; dYf = (tau/(1-tau))*S*dX_down (eq 6)
- Total inventory: Q = X + Xf; dQ_up = dX_up/(1-tau); dQ_down = dX_down (eq 8)
- LP cash: dC_LP = A*dQ_down - B*dQ_up (eq 19)
- Benchmark cash: dC_R = Sb*dQ_down - Sa*dQ_up (eq 22)
- LVR^tc = V^R - V^LP = C_R - C_LP (Definition 3.1)

## Key Parameters
- k = 1.0 (constant-product curve)
- C = 1.0 (initial curve level)
- X0 = 1.0 (initial active risky reserve)
- N = 50000 time steps, T = 1.0 years
- Simulation: GBM with sigma=0.5, seed=42; also deterministic paths

## Results Summary

**8/8 scenarios verified LVR^tc = 0** (within numerical tolerance 1e-8)

| Scenario | tau | lambda | max|LVR| | final LVR | max rel. LVR | LVR=0? | Lower hits | Upper hits | Boundary OK? |
|---|---|---|---|---|---|---|---|---|---|
| base_case | 0.0030 | 0.0010 | 3.55e-15 | -3.55e-15 | 1.91e-15 | ✓ | 7419 | 7570 | ✓ |
| zero_pool_fee_positive_spread | 0.0000 | 0.0050 | 1.02e-14 | -1.02e-14 | 5.62e-15 | ✓ | 6209 | 6333 | ✓ |
| positive_pool_fee_zero_spread | 0.0050 | 0.0000 | 6.22e-15 | -6.22e-15 | 3.33e-15 | ✓ | 6209 | 6333 | ✓ |
| both_positive_large | 0.0100 | 0.0050 | 2.04e-14 | -2.04e-14 | 1.11e-14 | ✓ | 2421 | 2550 | ✓ |
| monotone_up | 0.0030 | 0.0010 | 8.88e-16 | 0.00e+00 | 4.44e-16 | ✓ | 49600 | 0 | ✓ |
| monotone_down | 0.0030 | 0.0010 | 0.00e+00 | 0.00e+00 | 0.00e+00 | ✓ | 0 | 49600 | ✓ |
| oscillating | 0.0030 | 0.0010 | 8.88e-16 | 0.00e+00 | 4.44e-16 | ✓ | 22899 | 22528 | ✓ |
| tau=0_lambda_positive | 0.0000 | 0.0010 | 2.66e-15 | -2.22e-15 | 1.33e-15 | ✓ | 16939 | 17018 | ✓ |

## Verification Details

### Mathematical identity at boundaries (paper eq (5) and Theorem 3.2 proof):
At **lower boundary** (S = gamma * S*, pool sells):
- dC_LP = A * dQ_down = (S/(1-tau)) * dX_down = ((1-lambda)*S*) * dX_down
- dC_R  = Sb * dQ_down = ((1-lambda)*S*) * dX_down
- **dC_LP = dC_R** exactly ✓

At **upper boundary** (S = S*/gamma, pool buys):
- dC_LP = -B * dQ_up = -(1-tau)*S * dX_up/(1-tau) = -S * dX_up = -(S*/(gamma)) * dX_up
- dC_R  = -Sa * dQ_up = -(S*/(1-lambda)) * dX_up/(1-tau) = -(S*/((1-tau)(1-lambda))) * dX_up = -(S*/gamma) * dX_up
- **dC_LP = dC_R** exactly ✓

Since C_LP_0 = C_R_0 and increments are equal at every trading time,
LVR^tc = C_R - C_LP = 0 for all t.

## Artifacts
- `exp1_base_case_detail.png`: 4-panel detail plot (prices, wealth, LVR, inventory)
- `exp1_multi_scenario_lvr.png`: LVR for all parameter scenarios
- `exp1_wealth_trajectories.png`: V_LP vs V_R wealth trajectories
- `exp1_deterministic_paths.png`: Deterministic path LVR verification
- `exp1_metrics.json`: Machine-readable metrics for all scenarios

## Deviations from Paper
1. **Discretization**: The paper proves the result in continuous time. We use
   N=50000 discrete steps to approximate continuous dynamics. Residual LVR is
   pure floating-point arithmetic error (O(machine epsilon)), confirming the
   theoretical identity holds exactly in discrete implementation too.
2. **Price process**: Paper requires only "continuous external prices". We use
   GBM (satisfies continuity in the limit) and deterministic paths.
3. **Explicit Skorokhod**: For the separate-fee geometric-mean curve, we directly
   invert S_t from the reflected Z_t, giving exact (not approximate) reserve paths.
