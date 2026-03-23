# FTLM — extended Hubbard on square/rectangular lattices

Finite-temperature Lanczos (FTLM) for the **2D extended Hubbard model** (hopping \(t, t'\), onsite \(U\), nearest-neighbor \(V\)) on **square/rectangular** lattices, with a **C++17** core and optional **Python** bindings.

This repo is a **memory-first** successor to experiments in [PeizhiMai/FTLM](https://github.com/PeizhiMai/FTLM) and is informed by the workflow of [jurekokalj/ftlm_hub_cond](https://github.com/jurekokalj/ftlm_hub_cond) (Fortran reference for FTLM ideas, different geometry).

## Goal

Run **4×4** (16 sites) FTLM on a single cluster CPU node on the order of **128 GB RAM**, without storing dense Hamiltonians or full spectrum caches.

## Why 4×4 is hard (memory intuition)

- Working in fixed \((N_\uparrow, N_\downarrow)\) sectors, the largest block is near half-filling, e.g. \((8,8)\) on 16 sites: \(\binom{16}{8}^2 \approx 1.66\times 10^8\) basis states.
- One complex state vector at double precision is \(\sim 16\) bytes \(\times\) block dimension → **order a few GB per vector** in the largest sector.
- FTLM + Lanczos should keep only a **small, fixed window** of vectors (current + previous + Krylov workspace), store **sparse** \(H\) (or on-the-fly connections), and avoid **per-sample** allocations that scale with block dimension times sample count.

Design rule: **peak RAM \(\approx\) (few) × (largest sector dim) × (bytes/element) + sparse structure + small Lanczos tridiagonal buffers**, not **samples × Lanczos steps × vector copies**.

## Layout

```
ftlm_extended_hubbard/
├── CMakeLists.txt
├── requirements.txt
├── include/ftlm/
│   ├── fock_basis.hpp
│   ├── hubbard_hamiltonian.hpp
│   ├── hubbard_params.hpp
│   ├── lanczos.hpp
│   ├── lattice.hpp
│   └── version.hpp
├── src/
├── python/
│   ├── bindings.cpp
│   └── ftlm/          # thin Python package (stubs)
├── benchmarks/
│   └── bench_2x2.cpp
└── tests/
    ├── smoke.cpp
    └── test_hubbard_pipeline.cpp   # Fock + apply_H + Lanczos vs dense
```

## Build

```bash
cd /path/to/ftlm_extended_hubbard
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cmake -S . -B build -DFTLM_BUILD_PYTHON=ON
cmake --build build -j
ctest --test-dir build
```

`./build/ftlm_cli` prints version, lattice size, and an example sector dimension.

### 2×2 benchmark (sanity + throughput)

Builds with `-DFTLM_BUILD_BENCHMARKS=ON` (default). After `cmake --build build`:

```bash
./build/ftlm_bench_2x2
./build/ftlm_bench_2x2 --matvec-iters=200000 --lanczos-steps=36 --lanczos-seed=7
```

Reports: dense \(E_{\min},E_{\max}\) (Jacobi on the full sector Hamiltonian, many sweeps), `max|H_{\mathrm{dense}}v - \mathrm{apply}_H v|`, sustained `apply_H` throughput, and Lanczos Ritz extrema vs the dense reference. For meaningful timings use a **Release** build (`-DCMAKE_BUILD_TYPE=Release`).

**Plot (dense vs Ritz spectrum):**

```bash
./build/ftlm_bench_2x2 --write-plot-data=bench_out --matvec-iters=0
pip install -r benchmarks/requirements-plot.txt   # matplotlib + numpy
python3 benchmarks/plot_2x2_spectrum.py bench_out -o bench_2x2_spectrum.png
```

This writes `dense_evals.txt`, `lanczos_ritz.txt`, and `meta.txt` under `bench_out`, then saves a two-panel figure (sorted eigenvalues + extrema lines).

Python import path (without install): point at the CMake output tree so `ftlm/` contains both `__init__.py` and `_ftlm_core`.

```bash
export PYTHONPATH=$PWD/build/python:$PYTHONPATH
```

## References

- Jaklič & Prelovšek, *Advances in Physics* **49**, 1 (2000) — FTLM methodology.
- Your scaffold: [PeizhiMai/FTLM](https://github.com/PeizhiMai/FTLM).
- Optical-conductivity Hubbard example (triangular lattice, Fortran): [ftlm_hub_cond](https://github.com/jurekokalj/ftlm_hub_cond).

## Implementation status

Done (in order):

1. **`FockBasis`** — bit-packed \((N_\uparrow,N_\downarrow)\) sectors, `index_of` for matvec (`include/ftlm/fock_basis.hpp`).
2. **`apply_extended_hubbard`** — fused \(H|v\rangle\) from directed hoppings + \(U\), \(V\); **no dense \(H\)** (`hubbard_hamiltonian.hpp`). Geometry: `RectLattice`, `build_hubbard_geometry` (NN + four diagonals for \(t'\)), periodic wraps with \(\phi_x,\phi_y\) Peierls factors; skips degenerate on-site loops when `Lx==1` or `Ly==1`.
3. **`lanczos_extrema`** — Hermitian Lanczos + Jacobi on the small tridiagonal (`lanczos.hpp`). `tests/test_hubbard_pipeline.cpp` checks the lowest eigenvalue against a **dense** reference on a 2×1 cluster and Hermiticity under flux.

Next:

4. **FTLM estimator** for \(\ln Z\), density, etc., reusing the same `apply_h` and a **fixed** Lanczos workspace across random samples.
5. **Translation / momentum** blocks to shrink the Hilbert space before hitting 4×4 half-filling.
6. **pybind11**: expose `FockBasis::dim`, `apply_*` or `lanczos_extrema`, then `run_ftlm(...)`.
