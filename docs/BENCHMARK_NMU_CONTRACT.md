# n(μ) benchmark contract

This document locks the **regression contract** for grand-canonical density **n vs μ** at fixed **β** so CI and manual runs stay comparable.

## Canonical regression driver

| Role | Path |
|------|------|
| **Primary gate** | `benchmarks/run_nmu_regression.sh` → builds (if needed) and runs `benchmarks/run_3x2_three_way_monitoring.sh` |
| **Underlying three-way** | `benchmarks/run_3x2_three_way_monitoring.sh` |

The three-way script runs, in order:

1. **`build/ftlm_bench_ftlm_nmu_rect`** — particle sectors, no momentum (`--Lx=3 --Ly=2 …`)
2. **`build/ftlm_bench_ftlm_nmu_rect_k`** — momentum k-blocks (same μ grid and FTLM params)
3. **NumPy ED** via `benchmarks/hubbard_rect_ed.py` / `plot_n_vs_mu.py` helpers

Outputs (regenerated each run):

- `benchmarks/data/n_vs_mu_3x2_beta20_ftlm_nomom.tsv`
- `benchmarks/data/n_vs_mu_3x2_beta20_ftlm_k.tsv`
- `benchmarks/data/n_vs_mu_3x2_beta20_three_way_monitoring.txt` (full log)
- `benchmarks/data/n_vs_mu_3x2_beta20_ftlm_nomom_vs_k_vs_ed.png`

**Default physics / grid** (encoded in `run_3x2_three_way_monitoring.sh`): **β=20**, **μ∈[-5,25]**, **n_μ=221**, **t=1, tp=-0.35, U=5.75, V=0.9**, **ftlm-random=12**, **lanczos-steps=72**, **seed=7**.

## Acceptance (after substantial FTLM / momentum changes)

From the monitoring log / plot script summary:

- **max|nonmom − ED|** — FTLM non-momentum vs exact diagonalization
- **max|k − ED|** — FTLM k-blocks vs ED
- **max|nonmom − k|** — cross-check between solvers

Typical good order of magnitude on 3×2: **~10⁻²** (see latest run footer in the monitoring txt). **Investigate** if errors jump by an order of magnitude or RSS explodes unexpectedly.

Also record **wall_time_s** and **peak_rss_mib** / **peak_rss_bytes** from the `METRIC` lines and `/usr/bin/time -l` blocks.

## Build prerequisite

```bash
cmake -S . -B build
cmake --build build -j8
```

Benchmarks require: `ftlm_bench_ftlm_nmu_rect`, `ftlm_bench_ftlm_nmu_rect_k`, Python 3 + NumPy + `benchmarks/plot_n_vs_mu_3x2_three_way.py`.

## Ladder (larger systems)

| Stage | System | Role |
|-------|--------|------|
| Smoke | 2×2 | Fast sanity (`ftlm_test_*`, small benches) |
| **Gate** | **3×2** | **Canonical regression** (`run_nmu_regression.sh`) |
| Stress | 4×2, 4×3 | Memory / time scaling |
| Target | 4×4 (16 sites) | Single-node ~128 GB goal; tune `ftlm-random`, `lanczos-steps`, solver last |

## Related docs

- Fortran Ritz / Z assembly: `docs/FTLM_FORTRAN_DATA_STRUCTURE_N_VS_MU.md`
- Quantum Basis Lanczos / symmetry reference: `docs/QUANTUM_BASIS_LANCZOS_SYMMETRY_REFERENCE.md`
- Project rules: `.cursor/rules/ftlm-memory-first.mdc`
