# FTLM n(μ) implementation roadmap

Phased plan for **grand-canonical** \(\bar n(\mu)\) via sector traces + FTLM, aligned with `docs/BENCHMARK_NMU_CONTRACT.md` and `.cursor/rules/ftlm-memory-first.mdc`.

## Phase 1 — One-sector / one-block FTLM (locked)

- **Core:** `ftlm_log_partition_complex` + `lanczos_tridiagonal`; optional `LanczosFullBasisBuffer` only for tests/small dim.
- **Momentum blocks:** `HubbardMomentumBlock::apply` in Gram–whitened coordinates; public helpers:
  - `ftlm_log_partition_hubbard_momentum_block` — builds a fresh block (simple call sites).
  - `ftlm_log_partition_hubbard_momentum_block_ref` — uses an **existing** block (benchmarks reuse one block per k-cell).
- **Regression:** `benchmarks/run_nmu_regression.sh` → `run_3x2_three_way_monitoring.sh`; compare max|nonmom−ED|, max|k−ED|, max|nonmom−k|.

## Phase 2 — Grand-canonical assembly

- **Per particle sector:** accumulate \(\ln Z_s\) over momentum shells (`logsumexp` in k-bench).
- **Combine:** `ftlm_grandcanonical_density_mu_grid` on the flat sector index layout (see `bench_ftlm_nmu_rect*.cpp`).

## Phase 3 — Benchmarks and plots

- **Outputs:** TSV + `three_way_monitoring.txt` + PNG from `benchmarks/plot_n_vs_mu_3x2_three_way.py`.
- **Parameters:** document `beta`, `n_random`, `lanczos_steps`, `seed`, `--solver` in benchmark CLIs.

## Phase 4 — Scale toward 4×4

- **Memory:** peak should scale like \(O(d_{\max})\) for a few \(d\)-vectors per sector, not dense \(H\) or full Krylov storage on the default path.
- **Tuning:** increase `lanczos_steps` / `n_random` when error vs ED is too large; profile RSS vs `d_K`.

## Related

- Regression contract: `docs/BENCHMARK_NMU_CONTRACT.md`
- Fortran data layout (reference): `docs/FTLM_FORTRAN_DATA_STRUCTURE_N_VS_MU.md`
- Quantum Basis comparison: `docs/QUANTUM_BASIS_LANCZOS_SYMMETRY_REFERENCE.md`
