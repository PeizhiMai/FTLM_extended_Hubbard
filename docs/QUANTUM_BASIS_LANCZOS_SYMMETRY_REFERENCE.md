# Reference: Lanczos and symmetry in [wztzjhn/quantum_basis](https://github.com/wztzjhn/quantum_basis)

This note records how **Quantum Basis** implements Hermitian Lanczos (including optional full Krylov storage and checkpointing) and how it **indexes the Hilbert space under translational symmetry** (representatives, Lin table). It is **standalone**: it does not depend on the Fortran `ftlm_hub_cond` reference note in this repository.

Primary upstream files (as of their public `master`): `src/lanczos.cc`, `src/ckpt.cc`, `src/qbasis.h`, `src/basis.cc` / `src/lattice.cc`.

## Role of Quantum Basis vs this project

| Aspect | Quantum Basis | This repo (`ftlm_extended_hubbard`) |
|--------|----------------|-------------------------------------|
| Typical use | Exact diagonalization: ground state or few extremal eigenvalues in a **fixed symmetry block** | **Finite-temperature** FTLM-style traces: \(\mathrm{Tr}[\cdots e^{-\beta H}]\) via stochastic sampling + Lanczos moments / quadrature |
| Symmetry | Build a **reduced** basis (reps + Lin table); operators act in that **sector dimension** | Same **block** idea (e.g. particle number + momentum) to shrink the space per sample; implementation is separate code paths |
| Lanczos depth | Often many steps; may keep **all** Lanczos vectors (IRAM) or checkpoint long runs | Typically **short** recurrence per random start; **O(dim)** workspace for a few vectors unless an observable explicitly needs more |

The **three-term recurrence** and tridiagonal \(T\) are the same mathematics; the **objective** (single-sector ED vs thermal trace estimation) drives memory and I/O choices.

## Lanczos in Quantum Basis (`lanczos.cc`)

1. **Matrix-free Hamiltonian**  
   The recurrence uses `mat.MultMv2(v_in, v_out)` — the same pattern as a sparse or structured `H|v\rangle` application.

2. **Tridiagonal / Hessenberg packing**  
   Diagonal \(\alpha\) and off-diagonal \(\beta\) are stored in a packed `hessenberg[]` array (with indexing tied to `maxit`), not necessarily separate `std::vector<double>` buffers.

3. **Two vector layouts (`purpose`)**  
   - **`purpose == "iram"`**: `v` holds **\(m+1\) consecutive blocks** of length `dim` — the **full Krylov basis** for implicit restart / ARPACK-style use. Memory scales as **\(O(L \cdot \dim)\)**.  
   - **Otherwise**: **two-vector (or three-vector) cycling** — only a couple of length-`dim` slices are active; \(\alpha\), \(\beta\) are still accumulated. Memory stays **\(O(\dim)\)** for the Lanczos directions.

4. **Ritz monitoring**  
   They periodically diagonalize the growing small matrix (`hess_eigen`) to track Ritz values and convergence.

5. **Checkpointing (`ckpt.cc`, `enable_ckpt`)**  
   Optional **disk resume** for long runs: e.g. per-step vector files `lanczosVk.dat`, Hessenberg sidecar files, and logic to **rewind one step** if a partial write occurred. This serves **robustness of zero-temperature / long ED jobs**, not the FTLM thermodynamic algorithm itself.

6. **Future hooks (comments in source)**  
   The code mentions possible **DGKS reorthogonalization** and selective reorthogonalization for difficult spectra (`purpose` branches).

## Symmetry and “saved states” (`qbasis.h`, basis construction)

Quantum Basis separates:

- **Full basis** — no translation reduction.  
- **Translation-symmetric representative basis** — states are indexed by **representatives** of translation orbits, not by every translated copy.

Mechanisms named in the public headers include:

- `enumerate_basis` — generate states compatible with chosen quantum numbers / species.  
- `sort_basis_Lin_order`, **`fill_Lin_table`** — **Lin table** bookkeeping (ordering and translation action).  
- `classify_trans_full2rep`, `classify_trans_rep2group` — map full configurations to **representatives** and translation-group structure.  
- `norm_trans_repr` — normalization factors for symmetry-adapted states.

Their README notes a **lattice-shape restriction** when using translation symmetry (at least one of certain dimensions even — implementation detail of their generalized Lin table).

**Interpretation:** “Saving states under symmetry” there means: **fix the sector once** (reps + table), then Lanczos vectors are ordinary vectors of length **`dim` of that sector** (or a full Krylov stack of such vectors if `iram`).

## What this repo should borrow vs avoid

**Useful ideas to align with:**

- Same **two-level** picture: **symmetry reduces `dim`**, Lanczos runs **inside** that block.  
- **Two-vector vs full-basis** tradeoff is explicit in their `purpose` flag — mirror that design when we add optional full Krylov for specific observables.  
- **Checkpointing** is orthogonal to physics; only add if we need resumable long jobs.

**Finite-temperature FTLM specifics (not in Quantum Basis’s main story):**

- Many **random starts**, \(\beta\) in estimators, and **trace** formulas — not a single ground-state Lanczos chain.  
- Prefer keeping **`O(dim)`** Lanczos workspace and **α, β** (as in `ftlm::lanczos_tridiagonal`) unless a documented observable path requires **\(O(L \cdot \dim)\)** storage.

## Citation

If you use Quantum Basis in research, their repository recommends citing the Zenodo entry linked from their README ([quantum_basis on GitHub](https://github.com/wztzjhn/quantum_basis)).
