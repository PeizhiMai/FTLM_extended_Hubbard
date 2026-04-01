# Reference: Lanczos and saved states in [jurekokalj/ftlm_hub_cond](https://github.com/jurekokalj/ftlm_hub_cond)

This note records how the Fortran reference implements Hermitian Lanczos **with explicit storage of Krylov basis vectors**, and how that relates to this repository.

## Target difference (no port required)

| Aspect | `ftlm_hub_cond` | This repo (`ftlm_extended_hubbard`) |
|--------|-----------------|-------------------------------------|
| Lattice | Anisotropic triangular (`lx1,ly1,lx2,ly2` romb); `ct1=0` → square | Rectangular extended Hubbard (`t, t', U, V`) |
| Language | Fortran 77-style + OpenMP | C++ (heavy numerics), optional pybind11 |
| Observable | Optical / DC conductivity (needs current matrix elements) | Thermodynamics: **n vs μ** first |
| Raw output | Unformatted files → `cond_spect_omp.f` post-processes | TSV / benchmarks |

The **Lanczos mathematics** (Hermitian three-term recurrence, tridiagonal `T`, optional full basis) is the same; only the Hamiltonian application and downstream FTLM moments differ.

## What the Fortran code stores

From `fltm_hub_cond/hubTri2Dcond_omp.f` (see also `notes.txt`):

1. **Full Krylov basis**  
   `complex*16 phia(0:npa, 0:lstmax+1)` — each **column** is one Lanczos vector after the recurrence (sector dimension `np`, parents `npa`). This is **O(np × L)** complex memory for `L` steps, required because they later apply the **current operator** to each column (`opercur` in a loop over `k`) to build matrix elements between first and second Lanczos bases.

2. **Tridiagonal coefficients**  
   `a(0:lstmax-1)` (diagonal α), `b(-2:lstmax-1)` (β chain with extra slots for normalization bookkeeping). After the run, **Ritz vectors** in the Lanczos subspace are obtained via `tql2` into `vka(lstmax,lstmax)` / `vkb` for the second chain.

3. **Two Lanczos procedures per random sample**  
   - First: random `phia(:,1)`, then `call lanczos(...)` → saved `phia`, `a`, `ba`, …  
   - Second: start vector `phib(:,1) ∝ J |phia⟩` (current on the initial vector), then another `lanczos` for `phib`, `b`, `bb`, …  
   That structure is specific to **conductivity**; for **n(μ)** you typically need moments of `f(H)` and number operators, not the current between two Krylov chains.

4. **One step: `lanstep`**  
   - Applies sparse `H` to `phi(:,j1)` into a scratch `psi`.  
   - Computes α, orthogonalizes, forms the next normalized direction in `phi(:,j2)` using the three-term combination with `phi(:,j0)`, `phi(:,j1)` (same recurrence as standard Lanczos, but **columns are retained** instead of overwriting after two steps).

## What this repo does today

- `ftlm::lanczos_tridiagonal` in `src/lanczos.cpp` implements the same Hermitian recurrence but keeps only **`q`, `q_prev`, `w`** (`LanczosComplexWorkspace`) and outputs **α, β** only — **O(dim)** memory, **no** full basis. That matches the project rule: fixed small Lanczos workspace, no `O(L × dim)` storage unless explicitly justified.

- That is sufficient for **extrema / small tridiagonal eigensolves** (`lanczos_extrema`). A full **FTLM** line for thermodynamics will follow the Jaklič–Prelovšek recipe (random samples, trace estimators, continued fractions or Chebyshev moments) and may or may not require **optional** full-basis storage for specific observables — to be decided when implementing estimators.

## Literature (as cited in their README)

- J. Jaklič and P. Prelovšek, *Advances in Physics* **49**, 1–49 (2000) — FTLM review, notation.  
- P. Prelovšek and J. Bonča, *Strongly Correlated Systems* Springer SSSS Vol. 176 (2013), pp. 1–30.  
- J. Kokalj and R. McKenzie, *Phys. Rev. Lett.* **110**, 206402 (2013).

For a **field-by-field map** of Fortran arrays (CSR hops, `dnf`, Ritz weights) to **Z / ⟨N⟩** and what can be skipped for n(μ), see [`FTLM_FORTRAN_DATA_STRUCTURE_N_VS_MU.md`](./FTLM_FORTRAN_DATA_STRUCTURE_N_VS_MU.md).

## Takeaway for C++ FTLM (n vs μ)

- **Reuse** the existing **sparse `H|v⟩`** and sector structure; **do not** copy triangular geometry or Fortran I/O.  
- When an observable needs **⟨n⟩** or similar, prefer formulations that use **α, β** and continued fractions / moments so the memory stays **O(dim)** for a few vectors, unless profiling shows that storing a **truncated Krylov basis** (Fortran-style columns) is unavoidable for a given observable implementation.
