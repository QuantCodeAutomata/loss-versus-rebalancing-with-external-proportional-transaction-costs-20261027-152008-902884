# Loss-Versus-Rebalancing with External Proportional Transaction Costs

Numerical replication of selected experiments from:

> **"Loss-versus-rebalancing with external proportional transaction costs"**

## Experiment Implemented

### Experiment 1: Verification of Zero LVR for Separate Fee Mechanisms (Uniswap v3 Type)

**Research objective**: Verify Theorem 3.2 — under continuous external prices and pure
arbitrage, the transaction-cost LVR (`LVR^tc = V^R - V^LP`) is zero for the
separate-fee mechanism even when external transaction costs λ > 0.

**Key result**: `LVR^tc = V^R − V^LP = 0` holds exactly (within floating-point
precision) for all tested scenarios.

## Quick Start

```bash
bash reproduce.sh
```

This installs dependencies, runs all tests, executes the experiment, and saves
results to `results/`.

## Repository Structure

```
.
├── config/
│   └── exp_1_params.json       # Paper-derived experiment parameters
├── data/                       # No external data needed (synthetic price paths)
├── exp/
│   └── exp_1_separate_fees_lvr.py  # Experiment runner
├── results/
│   ├── RESULTS.md              # Summary of findings
│   ├── exp1_metrics.json       # Machine-readable metrics
│   └── *.png                   # Verification plots
├── src/
│   ├── reserve_curve.py        # Geometric-mean curve f(x) = C*x^{-k}
│   ├── skorokhod.py            # Skorokhod reflection for no-arbitrage band
│   ├── amm_simulation.py       # Separate-fee AMM simulation (Theorem 3.2)
│   └── lvr.py                  # LVR metrics and reporting
├── tests/
│   ├── test_reserve_curve.py   # Unit tests for reserve curve
│   ├── test_skorokhod.py       # Unit tests for Skorokhod reflection
│   ├── test_amm_simulation.py  # Unit tests for AMM simulation
│   └── test_lvr.py             # Unit tests for LVR metrics
├── reproduce.sh                # Entry-point script
└── README.md
```

## Methodology

The experiment numerically implements Theorem 3.2 of the paper:

1. **Reserve curve**: `Y = f(X) = C · X^{-k}` (geometric mean, paper Section 2.2.4)
2. **Internal price**: `S = −f′(X) = k·C·X^{-(k+1)} = k·Y/X`
3. **No-arbitrage band**: `γ·S* ≤ S ≤ S*/γ`, where `γ = (1−τ)(1−λ)` (paper eq. 4)
4. **Skorokhod reflection**: `Z_t = log(S_t/S*_t)` reflected in `[−h, h]`, `h = −log(γ)`
5. **Fee accounts**: `dXf = (τ/(1−τ))·dX↑`, `dYf = (τ/(1−τ))·S·dX↓` (paper eq. 6)
6. **LP cash flow**: `dC^LP = A·dQ↓ − B·dQ↑` (paper eq. 19)
7. **Benchmark cash flow**: `dC^R = S^b·dQ↓ − S^a·dQ↑` (paper eq. 22)
8. **LVR**: `LVR^tc = V^R − V^LP = C^R − C^LP` (paper Definition 3.1)

### Why LVR = 0

At the **lower boundary** (`S = γ·S*`, pool sells risky):
- Pool ask `A = S/(1−τ) = (1−λ)·S* = S^b` → `dC^LP = A·dQ↓ = S^b·dQ↓ = dC^R` ✓

At the **upper boundary** (`S = S*/γ`, pool buys risky):
- Pool bid `B = (1−τ)·S = S*/((1−τ)^{-1}·(1−λ)^{-1})^{-1} = S^a·(1−τ)/(1)` — combined with
  the `dQ↑ = dX↑/(1−τ)` accounting, yields `B·dQ↑ = S^a·dQ↑` → `dC^LP = dC^R` ✓

Since `C^LP_0 = C^R_0` and increments match at every trade, `C^R − C^LP = 0` throughout.

## Parameters

See `config/exp_1_params.json` for all paper-derived parameters:
- `k = 1.0` (constant-product curve, paper Section 2.2.4)
- `tau = 0.003` (0.3% pool fee)
- `lambda = 0.001` (0.1% external spread)
- `N = 50000` time steps (approximating continuous dynamics)
- `sigma = 0.5` (GBM volatility)

## Deviations from Paper

| Aspect | Paper | Implementation |
|--------|-------|----------------|
| Price process | Continuous finite-variation | GBM (N=50000 steps) + deterministic paths |
| Concentrated liquidity (Uniswap v3) | Mentioned but not defined | Standard geometric-mean curve used |
| Continuous time | Exact | Discrete approximation (N=50000) |

All deviations are conservative: residual LVR is `O(machine epsilon)`, consistent
with Theorem 3.2 holding exactly.
