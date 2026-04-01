#!/usr/bin/env bash
# 3×2 cluster: all four C++ solvers (nonmom, k-gram, k-orbit, ed) with /usr/bin/time -l.
# Writes curve TSVs, full monitoring log, comparison PNG, and four_solver_performance.tsv (runtime + peak RSS).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT_LOG="${REPO}/benchmarks/data/n_vs_mu_3x2_beta20_four_solver_monitoring.txt"
BK="${REPO}/build/ftlm_bench_ftlm_nmu_rect_k"
DATA="${REPO}/benchmarks/data"

COMMON=(
  --Lx=3 --Ly=2 --beta=20 --mu-min=-5 --mu-max=25 --n-mu=221
  --ftlm-random=12 --lanczos-steps=72 --seed=7 --no-monitor=0
)

if [[ ! -x "$BK" ]]; then
  echo "Build first: cmake --build build (need ftlm_bench_ftlm_nmu_rect_k)" >&2
  exit 1
fi

mkdir -p "$DATA"

{
  echo "=== 3×2 four-solver benchmark: β=20, μ∈[-5,25], n_μ=221, Lanczos=72, random=12, seed=7 ==="
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo ""

  echo "--- 1) solver=nonmom (particle sectors, no momentum) ---"
  /usr/bin/time -l "$BK" "${COMMON[@]}" --solver=nonmom \
    --out="${DATA}/n_vs_mu_3x2_beta20_solver_nonmom.tsv" 2>&1

  echo ""
  echo "--- 2) solver=k-gram (HubbardMomentumBlock / Gram) ---"
  /usr/bin/time -l "$BK" "${COMMON[@]}" --solver=k-gram --no-log-k-dims=1 \
    --out="${DATA}/n_vs_mu_3x2_beta20_solver_k_gram.tsv" 2>&1

  echo ""
  echo "--- 3) solver=k-orbit (orbit-direct k-block) ---"
  /usr/bin/time -l "$BK" "${COMMON[@]}" --solver=k-orbit --no-log-k-dims=1 \
    --out="${DATA}/n_vs_mu_3x2_beta20_solver_k_orbit.tsv" 2>&1

  echo ""
  echo "--- 4) solver=ed (dense exact per momentum block) ---"
  /usr/bin/time -l "$BK" "${COMMON[@]}" --solver=ed --no-log-k-dims=1 \
    --out="${DATA}/n_vs_mu_3x2_beta20_solver_ed.tsv" 2>&1

  echo ""
  echo "Done. TSVs under ${DATA}/n_vs_mu_3x2_beta20_solver_*.tsv"
} | tee "$OUT_LOG"

echo ""
python3 "${REPO}/benchmarks/plot_n_vs_mu_3x2_four_solver.py" --monitoring-log="$OUT_LOG"
echo "Full log: $OUT_LOG"
echo "Performance TSV (runtime + peak RSS in GB per solver): ${DATA}/n_vs_mu_3x2_beta20_four_solver_performance.tsv"
