#!/usr/bin/env bash
# reproduce.sh — Entry point for replicating Experiment 1.
#
# Paper: "Loss-versus-rebalancing with external proportional transaction costs"
# Theorem 3.2: LVR^tc = 0 under continuous prices and pure arbitrage.
#
# Usage: bash reproduce.sh
#
# Prerequisites: Python >= 3.10, pip
# No API keys required (synthetic price paths only).

set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "========================================================"
echo " Loss-versus-Rebalancing — Experiment Reproduction"
echo "========================================================"

# ── 1. Install dependencies ────────────────────────────────
echo ""
echo "[1/3] Installing dependencies..."
pip install --quiet --upgrade \
    numpy>=1.26 \
    pandas>=2.2 \
    scipy>=1.14 \
    matplotlib>=3.8 \
    pytest>=7.0

echo "      Done."

# ── 2. Run tests ───────────────────────────────────────────
echo ""
echo "[2/3] Running tests..."
python -m pytest tests/ -v --tb=short 2>&1
TEST_EXIT=$?

if [ "$TEST_EXIT" -ne 0 ]; then
    echo ""
    echo "ERROR: Tests failed (exit code $TEST_EXIT). Aborting experiment."
    exit "$TEST_EXIT"
fi

echo ""
echo "      All tests passed."

# ── 3. Run experiment ──────────────────────────────────────
echo ""
echo "[3/3] Running Experiment 1 (Zero LVR verification)..."
python exp/exp_1_separate_fees_lvr.py
EXP_EXIT=$?

if [ "$EXP_EXIT" -ne 0 ]; then
    echo ""
    echo "ERROR: Experiment returned non-zero exit code ($EXP_EXIT)."
    exit "$EXP_EXIT"
fi

# ── 4. Report artifacts ────────────────────────────────────
echo ""
echo "========================================================"
echo " Artifacts saved to results/"
echo "========================================================"
ls -1 results/
echo ""
echo "Summary: results/RESULTS.md"
echo "Metrics: results/exp1_metrics.json"
echo ""
echo "========================================================"
echo " Reproduction complete."
echo "========================================================"
