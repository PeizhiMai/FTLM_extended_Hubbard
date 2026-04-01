# Fortran `ftlm_hub_cond` data structures → **n vs μ** (thermodynamics)

This document maps [jurekokalj/ftlm_hub_cond](https://github.com/jurekokalj/ftlm_hub_cond) to our rectangular extended-Hubbard FTLM. **Optical conductivity** uses extra structures; **density vs chemical potential** uses a **strict subset**.

## 1. Hilbert-space layout (main program `hubTri2Dcond_omp.f`)

| Symbol | Role |
|--------|------|
| `nn0` | Number of lattice sites (cluster). |
| `nf`, `nu`, `nd` | Total electrons and up/down counts; loops fix a **U(1)×U(1)** sector. |
| `nca`, `npa` | Total / parent state counts (compile-time upper bounds). |
| `np` | Actual **parent** count after `parham` for this `(nu,nd)`. |
| `ipar`, `ideg` | Parent orbit / degeneracy under translations. |
| `dnf(i)` | **Momentum-sector normalization** for parent `i`: destructive interference can zero a row (`dnf < eps1` → hop list cleared). |
| `iham(i)` | CSR-style **row start** for hops from parent `i` (`iham(np+1)` ends the array). |
| `hamhopp(ih)`, `hamhopm(ih)`, `hamhopdir(ih)` | Target parent index, complex hop prefactor (includes phases), bond direction for TBC phases `ctc`. |
| `hamdia(i)` | Integer **double-occupancy count** (U term uses `cu * hamdia`). |

**Apply H:** `lanstep` does onsite + sum over `ih = iham(i)..iham(i+1)-1` (sparse matvec). Same pattern as our matrix-free `apply_extended_hubbard` / momentum block, different geometry.

## 2. What one **symmetry sample** produces (before files)

For each `(ik, idk, ismp)` after the **first** Lanczos:

- `nex` = number of **Ritz** levels kept from the Lanczos tridiagonal (≤ Lanczos steps).
- `en(j)` = Ritz eigenvalues in the Krylov subspace.
- `vka(1, j)` = **first Lanczos-basis component** of Ritz eigenvector `j` (used as a weight; with random start, this encodes overlap structure).
- `ba(-1)` = norm of the **initial** random vector (used in some optical ratios).

**Second Lanczos** (`phib`, `nexb`, `crvkavib8`, …) exists for **conductivity** only.

## 3. What is **written to disk** (relevant lines ~871–895)

Per record:

1. Header: `nn0, nf, nu`, then `ik, kx, ky, dkx, dky, dkxd, dkyd, ismp`, then `npw, nsmp, ndkl`.
2. **Thermo core:** `nex`, then `do j=1,nex: en(j), vka(1,j)`, then `ba(-1)`.
3. **Conductivity / τ extras:** `nexb`, `bb(-1)`, second-chain Ritz data, `crvkavib8`, `crvkavib10`, `crvkavib9`, …

For **n(μ)** you only need the **thermo core** + metadata (`npw`, `nf`, `nsmp`, phases). You can skip writing/reading the second Lanczos block entirely if you never compute σ(ω).

## 4. How **Z** and **⟨N⟩** are assembled (`cond_spect_omp.f`, ~386–403)

For each stored sector record, for each `μ` and `T`:

```text
wgh[j] = npw * (vka1(j)^2) / nsmp * exp( -( en1(j) - μ*nf - emin ) / T )
```

- `zt(μ,T,phase)  += Σ_j wgh[j]`  (partition-function–like weight for this sector/sample).
- `nfa(μ,T,phase) += Σ_j nf * wgh[j]`  (electron-number weighted; gives **density** after dividing by Z and `nn0` elsewhere in the file).

Spin symmetry: if `nu ≠ nf-nu`, weights are multiplied by **2** (only one of the pair of sectors computed).

**emin(μ, phase)** is a reference energy (from `hmmumin`) so exponentials stay in range — same idea as shifting H by a constant in trace estimators.

So **n vs μ** needs:

- Ritz energies and **squared overlaps** in the Lanczos/Ritz basis (`vka(1,j)^2`),
- sector quantum numbers `nf`, `npw`, random-sample count `nsmp`,
- **not** full `phia(:,k)` columns and **not** current/tau matrices.

## 5. Mapping to **this** C++ codebase

| Fortran concept | Our repo |
|-----------------|----------|
| Sector `(nu, nd)` or momentum block | `FockBasis(n_sites, n_up, n_dn)` and/or `HubbardMomentumBlock` with fixed `K`. |
| Sparse H | `apply_extended_hubbard` or block `apply` (complex Hermitian). |
| Lanczos α, β | `lanczos_tridiagonal` → `alpha`, `beta`. |
| Ritz values + `|⟨e_j|e_0⟩|` style weights | `ftlm_tridiagonal_ritz_from_lanczos_coeffs` → `eigenvalues`, `w0_squared` (`V_{0,j}^2`). |
| Inner sum \(\ln\sum_j w_j e^{-\beta E_j}\) | `ftlm_log_trace_exp_beta_ritz` or `ftlm_log_tridiagonal_partition_exp` (same inner sum as `ftlm_log_partition_*` per random start). |
| Grand-canonical \( \bar n(\mu)\) from sector log-Tr | `ftlm_grandcanonical_density_mu_grid` (same combination as `bench_ftlm_nmu_rect`). |
| Stochastic / multi-sample | Fortran: explicit `nsmp` loops and `1/nsmp`. We average over `FtlmParams::n_random` in `ftlm_log_partition_*`. |

**Easier quantity:** For **⟨N⟩(μ)** in the **grand-canonical** ensemble, you still sum sector contributions with **Boltzmann weights**; the Fortran code does this **explicitly per Ritz level** in the Lanczos subspace. Our current `ftlm_log_partition_complex` estimates **log Tr e^{-βH}** via Gaussian quadrature on the tridiagonal — a **different but equivalent FTLM flavor** (stochastic trace + continued fraction / quadrature). To match **n(μ)** curves, implement either:

- **A)** Same as Fortran: per sector, Ritz diagonalization of Lanczos `T`, accumulate `wgh` and `nf*wgh` for each μ, T; or  
- **B)** Estimate `Tr[N e^{-β(H-μN)}] / Tr[e^{-β(H-μN)}]` via FTLM moments / separate observable traces using the same `apply_H` and Lanczos machinery without storing `phia`.

Both avoid **O(L×dim)** basis storage if you only keep **α, β** and a **small L×L** eigendecomposition.

## 6. What you can **drop** relative to the full Fortran pipeline

- Second Lanczos chain and **all** `crvkavib*` matrices for σ(ω) and τ sum rules.
- Writing **unformatted** huge tapes if everything is computed in one pass in C++.

Keep the **sparse sector Hamiltonian** pattern (`iham` / hops) as the mental model for **memory scaling**: **O(nnz) + O(dim)** vectors, not **O(dim²)**.
