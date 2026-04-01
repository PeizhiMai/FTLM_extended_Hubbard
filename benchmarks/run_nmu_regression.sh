#!/usr/bin/env bash
# Canonical n(μ) regression gate for 3×2 (FTLM non-mom + k + ED). See docs/BENCHMARK_NMU_CONTRACT.md.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

if [[ ! -d "${REPO}/build" ]]; then
  echo "run_nmu_regression: configure build first: cmake -S . -B build" >&2
  exit 1
fi

cmake --build "${REPO}/build" -j8

exec bash "${REPO}/benchmarks/run_3x2_three_way_monitoring.sh"
