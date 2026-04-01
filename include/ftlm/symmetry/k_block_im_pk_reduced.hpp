#pragma once

// Tiny-sector prototype: momentum k-block Hamiltonian in an orthonormal basis of Im(P_K) built from the dense
// Hermitian translation projector **without** `MomentumPhiGramBasis` / raw-seed Gram / `zheev` on `G = Φ_raw† Φ_raw`.
//
// Vector space and basis U
// ------------------------
// - Coefficients live in **full** `FockBasis` index space of size `d_full = fb.dim()` (same as dense `P_K`).
// - `P_K` is the symmetrized dense projector from `detail::build_translation_projector_dense` +
//   `detail::hermitian_symmetrize_inplace` (same as elsewhere in the tiny-dense tooling).
// - `Im(P_K)` is the range of `P_K`. For a Hermitian projector, eigenvalues are in `[0,1]`; we take eigenvectors whose
//   eigenvalue is **> 0.5** (numerical 1-eigenspace). That set is orthonormal and spans `Im(P_K)` when eigenvalues are
//   tight around 0 and 1.
// - **Dimension:** `r = dim(Im(P_K)) = rank(P_K) = Tr(P_K)` in exact arithmetic; we set `r` to the count of
//   eigenvalues `> 0.5` after `zheev` on `P_K`.
// - **Columns of U:** column-major `U[p + i * d_full]` is the `i`-th basis vector in full space (same layout as LAPACK
//   eigenvectors). Columns are taken in **increasing column index `j`** from the diagonalized `P_K` matrix for which
//   `λ_j > 0.5` (ascending eigenvalue order from LAPACK `zheev`). **Sign / phase** are whatever LAPACK returns; the
//   reduced matrix `U† H U` is therefore unique only up to unitary conjugation on the block (same spectrum as any other
//   orthonormal basis of the same subspace).
//
// Relation to production `dk_Gram`
// ---------------------------------
// Production uses whitened raw orbit–Bloch seeds; `dk_Gram = k_out` from `MomentumPhiGramBasis::build`. When
// `dk_Gram = rank(P_K)` (typical in well-conditioned sectors), both span the **same** subspace `Im(P_K)`; then
// `U† H U` and `Φ_whiten† H Φ_whiten` are **unitarily equivalent** (same eigenvalues), but **not** equal entrywise
// unless the bases differ only by permutation × diagonal phases.
//
// Intended use: validation / debug only; requires dense `P_K` (`O(d_full²)`) and `zheev` on `d_full`.

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "ftlm/fock_basis.hpp"
#include "ftlm/hubbard_params.hpp"
#include "ftlm/lattice.hpp"
#include "ftlm/symmetry/momentum_sector.hpp"
#include "ftlm/symmetry/raw_state.hpp"

namespace ftlm {
namespace symmetry {

/// Reduced `r × r` Hermitian Hamiltonian `H_red = U† H U` with `U` an orthonormal basis of `Im(P_K)` from dense `P_K`.
/// **Not** the production Gram basis; compare via spectra or explicit `U† Φ` rotation when needed.
struct TinyImPkReducedHamiltonian {
  int d_full = 0;
  /// Block dimension (= `rank(P_K)` with the `> 0.5` threshold).
  int r = 0;
  /// Column-major `d_full × r`: orthonormal columns spanning `Im(P_K)`.
  std::vector<std::complex<double>> u_colmajor{};
  /// Column-major `r × r`: `H_red = U† H U` with `H` the full-sector Hubbard Hamiltonian (same as `build_dense_h_sector`).
  std::vector<std::complex<double>> h_red_colmajor{};

  void apply(const std::complex<double>* x, std::complex<double>* y) const {
    if (r <= 0) {
      return;
    }
    for (int i = 0; i < r; ++i) {
      std::complex<double> s{0.0, 0.0};
      for (int j = 0; j < r; ++j) {
        s += h_red_colmajor[static_cast<std::size_t>(i + j * r)] * x[static_cast<std::size_t>(j)];
      }
      y[static_cast<std::size_t>(i)] = s;
    }
  }
};

/// Build `TinyImPkReducedHamiltonian` from dense `P_K` and dense sector `H`. **No** `MomentumPhiGramBasis::build`.
/// Returns `false` if `d_full == 0`, `d_full > max_d_full`, or `r == 0` (empty momentum subspace).
/// Throws `std::invalid_argument` if `max_d_full` is exceeded (same spirit as `build_tiny_pk_h_pk_dense`).
bool build_tiny_im_pk_reduced_hamiltonian(const HubbardParams& p, const FockBasis& fb, int lx, int ly,
                                          MomentumSector K, const std::vector<SpinfulHopping>& hoppings,
                                          const std::vector<NearestPair>& nn_pairs, int max_d_full,
                                          TinyImPkReducedHamiltonian* out);

}  // namespace symmetry
}  // namespace ftlm
