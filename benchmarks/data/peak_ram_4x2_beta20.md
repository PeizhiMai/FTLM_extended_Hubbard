# Peak RAM: 4×2, β = 20 (same workload as `n_vs_mu_4x2_beta20_ftlm_ed_compare.png`)

Measured with macOS `/usr/bin/time -l` (BSD `time`).

## FTLM — `ftlm_bench_ftlm_nmu_rect`

Command (representative):

```text
./ftlm_bench_ftlm_nmu_rect --Lx=4 --Ly=2 --beta=20 --mu-min=-8 --mu-max=14 --n-mu=221 \
  --ftlm-random=16 --lanczos-steps=96 --seed=7 --out=.../n_vs_mu_4x2_beta20_ftlm.tsv
```

| Metric | Value |
|--------|--------|
| Wall time | ~190 s |
| **Maximum resident set size** | 10 780 672 B (**~10.3 MiB**) |
| **Peak memory footprint** (macOS) | 11 715 976 B (**~11.2 MiB**) |

Raw `time -l` tail: `n_vs_mu_4x2_beta20_ftlm.stderr.log` (lines after sector list).

## ED — `plot_n_vs_mu.py` (NumPy exact diagonalization + matplotlib)

Command:

```text
.venv/bin/python benchmarks/plot_n_vs_mu.py --lx 4 --ly 2 --beta 20 \
  --cpp-tsv benchmarks/data/n_vs_mu_4x2_beta20_ftlm.tsv \
  --out benchmarks/data/n_vs_mu_4x2_beta20_ftlm_ed_compare.png --no-show
```

| Metric | Value |
|--------|--------|
| Wall time | ~161 s |
| **Maximum resident set size** | 2 170 343 424 B (**~2.02 GiB**) |
| **Peak memory footprint** (macOS) | 2 196 148 456 B (**~2.05 GiB**) |

Raw output: `mem_ed_time.txt`.

### Notes

- **FTLM** stays sparse / vector-sized; peak RSS scales with the largest sector dimension and a small Lanczos workspace (no dense `H`).
- **ED** builds and diagonalizes dense sector matrices and holds **65 536** eigenlevels for the full 4⁸ Hilbert space — hence **~2 GiB** peak on this run.
- On Apple Silicon, “peak memory footprint” can differ slightly from classic RSS; both are reported above.
