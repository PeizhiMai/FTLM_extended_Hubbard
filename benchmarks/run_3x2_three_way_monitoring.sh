#!/usr/bin/env bash
# 3x2 cluster: FTLM non-mom + FTLM k-blocks + NumPy ED timing, with peak RSS / wall time.
# Always regenerates benchmarks/data/n_vs_mu_3x2_beta20_ftlm_nomom_vs_k_vs_ed.png at the end.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${REPO}/benchmarks/data/n_vs_mu_3x2_beta20_three_way_monitoring.txt"
B="${REPO}/build/ftlm_bench_ftlm_nmu_rect"
BK="${REPO}/build/ftlm_bench_ftlm_nmu_rect_k"

if [[ ! -x "$B" ]] || [[ ! -x "$BK" ]]; then
  echo "Build benchmarks first: cmake --build build (need ftlm_bench_ftlm_nmu_rect, ftlm_bench_ftlm_nmu_rect_k)" >&2
  exit 1
fi

{
  echo "=== 3x2 cluster: β=20, μ∈[-5,25], n_μ=221, t=1 tp=-0.35 U=5.75 V=0.9 ==="
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo ""
  echo "--- 1) FTLM non-momentum (ftlm_bench_ftlm_nmu_rect) — C++ monitor + time -l ---"
  /usr/bin/time -l "$B" --Lx=3 --Ly=2 --beta=20 --mu-min=-5 --mu-max=25 --n-mu=221 \
    --ftlm-random=12 --lanczos-steps=72 --seed=7 --no-monitor=0 \
    --out="${REPO}/benchmarks/data/n_vs_mu_3x2_beta20_ftlm_nomom.tsv" 2>&1
  echo ""
  echo "--- 2) FTLM momentum k-blocks (ftlm_bench_ftlm_nmu_rect_k) — C++ monitor + time -l ---"
  # Match non-momentum Lanczos depth (k-bench default is 2*a_nonmom/N_sites = 24 for 3x2).
  /usr/bin/time -l "$BK" --Lx=3 --Ly=2 --beta=20 --mu-min=-5 --mu-max=25 --n-mu=221 \
    --ftlm-random=12 --lanczos-steps=72 --seed=7 --no-log-k-dims=1 --no-monitor=0 \
    --out="${REPO}/benchmarks/data/n_vs_mu_3x2_beta20_ftlm_k.tsv" 2>&1
  echo ""
  echo "--- 3) ED (NumPy hubbard_rect_ed) — /usr/bin/time -l ---"
  /usr/bin/time -l python3 -c "
import sys
from pathlib import Path
import numpy as np
REPO = Path('${REPO}')
sys.path.insert(0, str(REPO / 'benchmarks'))
from plot_n_vs_mu import load_tsv
from hubbard_rect_ed import all_eigenlevels_rect, grand_canonical_n
tsv = REPO / 'benchmarks/data/n_vs_mu_3x2_beta20_ftlm_nomom.tsv'
mu, _, meta = load_tsv(tsv)
beta = meta['beta']
p = {k: meta[k] for k in ('t','tp','U','V') if k in meta}
n_sites = 6
levels = all_eigenlevels_rect(3, 2, p, 0.0, 0.0)
for m in mu:
    grand_canonical_n(beta, float(m), levels, n_sites)
print('ED: n_mu points =', len(mu), '  eigenlevels =', len(levels))
" 2>&1
  echo ""
  echo "--- Summary (wall clock & peak RSS) — see benchmarks/memory_k_block.txt for scaling notes ---"
} | tee "$OUT"

echo ""
echo "--- Updating three-way plot PNG ---"
python3 "${REPO}/benchmarks/plot_n_vs_mu_3x2_three_way.py"
echo "Done. Log: ${OUT}"
