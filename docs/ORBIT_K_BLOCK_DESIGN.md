# Orbit \(K\)-block basis (matrix-free pipeline)

Design source of truth for the translation-orbit representative basis and the matrix-free \(H_K\) apply (Steps 1–2). No Gram matrix, no `zheev`, no dense \(P_K\) or \(H_{\mathrm{red}}\).

**Benchmark default:** `ftlm_bench_ftlm_nmu_rect_k` with **no** `--solver` flag **prefers** this pipeline (one direction per surviving translation orbit, explicit \(A_\alpha(K)\) normalization — the closest analogue here to Fortran parent/orbit sparse indexing). The Gram + Bloch-\(\Phi\) path (`HubbardMomentumBlock`) remains available via **`--solver=k-gram`** or when orbit-direct is inapplicable for a sector.

## Phase conventions

- `translation_bloch_phase(K, dx, dy)` in `momentum_sector.hpp` implements **\(\chi_K^*(R)\)** (one-body Bloch factor \(\exp(-\mathrm{i}\,\mathbf{k}\cdot\mathbf{R})\) with the project’s discrete \(\mathbf{k}\cdot\mathbf{R}\) convention).
- Bloch columns use \(\chi_K^*(R)=\texttt{translation\_bloch\_phase(K, dx, dy)}\) in the orbit sum (same as `fill_phi_orbit_bloch_from_seed`).
- The **reduced hop apply** must use the **same** factor \(\chi_K^*(R)\) on the translation \(R\) with \(T_R\alpha=s'\). Using `conj` again (`conj(translation_bloch_phase)` = \(\chi_K(R)\)) was incorrect and skewed the imaginary part of \(H_K\) vs \(\Phi^\dagger H\Phi\).

## Stabilizer character and surviving orbits

For a canonical orbit representative \(\alpha\), translation subgroup \(G = \mathbb{Z}_{L_x}\times\mathbb{Z}_{L_y}\), stabilizer
\(\mathrm{Stab}(\alpha) = \{S\in G : T_S\alpha=\alpha\}\), and fermionic translation sign \(\eta(\alpha,S)\in\{\pm1\}\) from `fermionic_translation_sign_rect` (so \(U(S)|\alpha\rangle = \eta(\alpha,S)|\alpha\rangle\) for \(S\in\mathrm{Stab}(\alpha)\)):

\[
A_\alpha(K) = \sum_{S\in\mathrm{Stab}(\alpha)} \eta(\alpha,S)\,\chi_K^*(S).
\]

**Survival:** the projected orbit state is nonzero iff \(A_\alpha(K)\neq 0\) (in floating point: \(|\,A_\alpha(K)\,|\) above a small tolerance).

**Bit-only mask vs fermionic sum:** the legacy `compatible_momentum_mask` enforces \(\chi_K(S)=1\) on bit-stabilizers only. This design uses the **full fermionic character** \(A_\alpha(K)\) as the only gate for including an orbit in the reduced basis.

**Normalization (exact arithmetic):** for surviving orbits, \(|\,A_\alpha(K)\,| = |\mathrm{Stab}(\alpha)|\) when the character sum is nondestructive (see code comments). In floating point, metadata stores \(|A_\alpha|\) from the explicit sum for apply weights.

## Displacement convention (single convention for all code)

We fix:

\[
T_R\,\alpha = s',
\]

where \(T_R\) is the **forward** lattice translation by \(R=(dx,dy)\) in the same sense as `translate_raw_state_rect(\alpha, lx, ly, dx, dy)` and \(s'\) is the **bit-translated** ket. The fermionic sign is

\[
U(R)|\alpha\rangle = \eta(\alpha,R)\,|s'\rangle,
\quad \eta = \texttt{fermionic\_translation\_sign\_rect}(\alpha,lx,ly,dx,dy).
\]

We do **not** use \(T_{-R}\alpha=s'\) in this codebase path.

## Matrix-free \(H_K\) apply (Step 2; recorded here for consistency)

Let \(|\alpha;K\rangle\) be the normalized projected orbit state built from representative \(\alpha\) (same physics as the translation projector, with the fermionic stabilizer sum baked into normalization). For a Hamiltonian hop from representative \(\beta\) to a raw ket \(|s'\rangle\) with full-space amplitude \(h_{s'\beta}\) (Jordan–Wigner sign already included in \(H\)), let \(\alpha\) be the canonical representative of \(s'\), and let \(R\) be the **unique** translation with \(T_R\alpha=s'\) used in the convention above. Then the contribution to the reduced component \(y_\alpha\) from \(x_\beta\) is

\[
y_\alpha \mathrel{+}= h_{s'\beta}\,\eta(\alpha,R)\,\chi_K^*(R)\,\sqrt{\frac{|\,A_\alpha(K)\,|}{|\,A_\beta(K)\,|}}\,x_\beta.
\]

Here \(\chi_K^*(R)=\texttt{translation\_bloch\_phase(K, dx, dy)}\) for \(R=(dx,dy)\).

Step 1 implements only metadata: \(A_\alpha\), surviving reps, and `map_raw_to_rep` under this convention.

## Magnitude of \(A_\alpha(K)\) (exact arithmetic)

For a **surviving** orbit (nonzero projection), in exact arithmetic one has
\[
|\,A_\alpha(K)\,| = |\mathrm{Stab}(\alpha)|
\]
whenever the stabilizer character sum is nondestructive (each term has unit modulus and the sum does not cancel). Implementation stores \(|A_\alpha|\) from the explicit complex sum; for apply weights, \(\sqrt{|\,A_\alpha\,|/|\,A_\beta\,|}\) can use these values (Step 2).

