#pragma once

// Dimensional analysis: Gram / dense projector / production block
// ================================================================
// **Full sector:** `d_full = dim` of `FockBasis(n_sites, n_up, n_down)`.
//
// **Dense momentum projector** `P_K` (after `detail::hermitian_symmetrize_inplace` on the group sum):
// Hermitian projector on `C^{d_full}` with image `Im(P_K)` (momentum–`K` subspace in this particle sector).
// `rank(P_K) = Tr(P_K)` in exact arithmetic (eigenvalues are 0 or 1); numerically use eigenvalue count with threshold
// or rounded trace.
//
// **Raw orbit–Bloch seeds** (`momentum_phi_seeds`): one column per distinct ket in every translation orbit compatible
// with `K` (size `k_in`). These columns **span** `Im(P_K)` (same subspace as dense `P_K`); see `k_basis.hpp`.
// So **linear** dependencies among seeds are possible: `k_in ≥ rank(P_K)` always; equality iff seeds form a basis.
//
// **Gram matrix** `G = Φ_raw^† Φ_raw` is `k_in × k_in`. **Whitening** (`MomentumPhiGramBasis::build`, same as
// `build_momentum_phi_orbit_orthonormal`): diagonalize `G`, keep eigenpairs with `λ_j > tol_ev`,
// `tol_ev = max(1e-14 · λ_max(G), 1e-20)`. Output dimension **`k_out = dk_Gram`** counts surviving modes.
// Thus **`dk_Gram ≤ rank(G) ≤ k_in`**. In exact arithmetic `rank(G) = dim span(Φ_raw) = rank(P_K)`; numerically
// **`dk_Gram` can be `< rank(G)`** if some positive eigenvalues fall **below** `tol_ev` (near-nullspace of `G`).
// **Stabilizers** enter **only** through orbit sizes / normalization in `fill_phi_orbit_bloch_from_seed`, not as an
// extra dimension cut beyond `P_K`.
//
// **Production block** `HubbardMomentumBlock`: `dk = gram.k_out` (same `dk_Gram`). Operator is `Φ_whiten^† H Φ_whiten`
// in **`dk`-dimensional** orthonormal coordinates.
//
// **Gap vs full `Im(P_K)`:** If `dk_Gram < rank(P_K)`, the Gram basis spans a **proper subspace** of `Im(P_K)` (dropped
// near-null or redundant directions). The matrix-free **`P_K H P_K`** prototype in **full** `d_full` space is a
// **different-size** representation than production unless `dk_Gram = d_full` (rare).
//
// **Gram-free reduced target (prototype direction):** an orthonormal basis of **`Im(P_K)`** from the **spectral**
// decomposition of dense `P_K` (eigenvectors for eigenvalue ≈ 1) has dimension **`r = rank(P_K)`** and avoids the
// seed Gram / `zheev` on `G`, but still needs dense **`P_K`** for tiny `d_full` (O(d²) storage). This matches
// **`rank(P_K)`**, not necessarily **`dk_Gram`** when Gram drops modes.

#include <complex>
#include <cstddef>

namespace ftlm {
namespace symmetry {
namespace tiny_pk {

/// Real part of `tr(A)` for Hermitian `A` stored column-major `n×n`.
inline double trace_hermitian_colmajor(int n, const std::complex<double>* a_colmajor) {
  double t = 0.0;
  for (int i = 0; i < n; ++i) {
    t += a_colmajor[static_cast<std::size_t>(i + i * n)].real();
  }
  return t;
}

}  // namespace tiny_pk
}  // namespace symmetry
}  // namespace ftlm
