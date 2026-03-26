#pragma once

// Tiny-system prototype: full-sector dense translation projector P_K and Hamiltonian H, then M = P_K H P_K.
// Does NOT use MomentumPhiGramBasis / zheev / Gram construction for the operator — only translation_projector_dense
// + apply_extended_hubbard.
//
// Vector space
// ------------
// - **Full** particle sector: one complex coefficient per Fock basis ket in the usual `FockBasis` ordering
//   (same index space as `HubbardMomentumBlock::d_full()` for the same n_up, n_down).
//
// Momentum projector P_K (dense)
// ------------------------------
// - Same formula as `detail::build_translation_projector_dense`:
//   \(P_K = \frac{1}{|G|}\sum_{R} \overline{\chi_K(R)}\, U(R)\) with \(G\) the translation group on the torus
//   (\(|G| = L_x L_y\)), \(U(R)\) the fermionic signed permutation of kets, and \(\chi_K(R)\) the Bloch phase
//   from `translation_bloch_phase` (must match extended Hubbard / momentum code conventions).
// - After the group sum, `detail::hermitian_symmetrize_inplace` cleans Hermitian roundoff.
//
// Matrix-free P_K (same symmetrized operator, no \(d_{\mathrm{full}}\times d_{\mathrm{full}}\) storage)
// ------------------------------------------------------------------------------------------
// - `apply_translation_U_forward_vector` / `apply_translation_U_transpose_vector`: one signed translation \(U(R)\)
//   or \(U(R)^T\) on a sector vector (same rules as `build_dense_translation_U`).
// - `apply_symmetrized_P_K_vector_matrix_free`: \(P_K x=\frac12(P_{\mathrm{raw}}x+P_{\mathrm{raw}}^\dagger x)\) with
//   \(P_{\mathrm{raw}}=\frac{1}{|G|}\sum_R \overline{\chi_K(R)}\,U(R)\), matching the Hermitianized dense matrix.
//
// Hamiltonian H (dense)
// ---------------------
// - Same `apply_extended_hubbard` as production (same `HubbardParams`, `hoppings`, `nn_pairs`).
// - Columns \(j\) of \(H\) are \(H e_j\) via one apply per column; then Hermitian symmetrization.
//
// Block operator in production vs P_K H P_K (same Φ basis)
// --------------------------------------------------------
// - Production `HubbardMomentumBlock::apply` implements **y = Φ† H (Φ x)** in orthonormal Gram–whitened coordinates.
// - Dense full-space **M = P_K H P_K** sandwiched in that basis: **Φ† M Φ = Φ† P_K H P_K Φ**.
// - For columns \(\phi_j \in \mathrm{Im}(P_K)\): \(P_K\phi_j=\phi_j\), so \(\Phi^\dagger P_K H P_K \Phi=\Phi^\dagger P_K H\Phi\).
// - Also \(\langle\phi_i|(I-P_K)=0\) (bra kills the complement of \(\mathrm{Im}(P_K)\)), hence
//   \(\phi_i^\dagger H\phi_j=\phi_i^\dagger P_K H\phi_j\). Therefore **Φ† H Φ = Φ† P_K H P_K Φ** on that subspace.
// - So the **dense** \(P_K H P_K\) matrix in the **same** orthonormal \(\Phi\) basis matches production’s block Hamiltonian
//   **when** \(\Phi\) spans a subspace of \(\mathrm{Im}(P_K)\) (as in the Gram–whitened orbit–Bloch construction).
//
// Use **only** for very small `d_full` (default cap 64): memory \(O(d_{\mathrm{full}}^2)\).

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "ftlm/fock_basis.hpp"
#include "ftlm/hubbard_hamiltonian.hpp"
#include "ftlm/hubbard_params.hpp"
#include "ftlm/lattice.hpp"
#include "ftlm/symmetry/momentum_sector.hpp"
#include "ftlm/symmetry/raw_state.hpp"
#include "ftlm/symmetry/translation_projector_dense.hpp"

namespace ftlm {
namespace symmetry {
namespace tiny_pk {

/// Column-major \(n\times n\) times vector: \(y = A x\) (overwrites `y`).
inline void matvec_colmajor_vector(int n, const std::complex<double>* A_colmajor, const std::complex<double>* x,
                                   std::complex<double>* y) {
  for (int i = 0; i < n; ++i) {
    std::complex<double> s{0.0, 0.0};
    for (int j = 0; j < n; ++j) {
      s += A_colmajor[static_cast<std::size_t>(i + j * n)] * x[static_cast<std::size_t>(j)];
    }
    y[static_cast<std::size_t>(i)] = s;
  }
}

/// One translation move \(U(R)\ket{n}\) in the Fock basis: \((U x)_{n'} += \eta\, x_n\) with \(n'=\) index of \(R\ket{n}\).
/// Same indexing as `detail::build_dense_translation_U` (fermionic sign, invalid images skipped).
inline void apply_translation_U_forward_vector(const FockBasis& fb, int lx, int ly, int ex, int ey,
                                               const std::complex<double>* x, std::complex<double>* out) {
  const int d = fb.dim();
  std::fill(out, out + d, std::complex<double>{0.0, 0.0});
  for (int n = 0; n < d; ++n) {
    const RawState s = detail::raw_from_fock_index(fb, n);
    const RawState st = translate_raw_state_rect(s, lx, ly, ex, ey);
    const int np = fb.index_of(static_cast<std::uint64_t>(st.up), static_cast<std::uint64_t>(st.dn));
    if (np < 0) {
      continue;
    }
    const double eta = fermionic_translation_sign_rect(s, lx, ly, ex, ey);
    out[np] += std::complex<double>{eta, 0.0} * x[static_cast<std::size_t>(n)];
  }
}

/// Adjoint for real signed permutation: \(U^\dagger = U^T\). Satisfies \((U^T x)_n = \eta\, x_{n'}\) with \(n'=\) index of
/// \(R\ket{n}\) (same \(n'\) as forward row index for column \(n\)).
inline void apply_translation_U_transpose_vector(const FockBasis& fb, int lx, int ly, int ex, int ey,
                                                 const std::complex<double>* x, std::complex<double>* out) {
  const int d = fb.dim();
  std::fill(out, out + d, std::complex<double>{0.0, 0.0});
  for (int n = 0; n < d; ++n) {
    const RawState s = detail::raw_from_fock_index(fb, n);
    const RawState st = translate_raw_state_rect(s, lx, ly, ex, ey);
    const int np = fb.index_of(static_cast<std::uint64_t>(st.up), static_cast<std::uint64_t>(st.dn));
    if (np < 0) {
      continue;
    }
    const double eta = fermionic_translation_sign_rect(s, lx, ly, ex, ey);
    out[static_cast<std::size_t>(n)] += std::complex<double>{eta, 0.0} * x[static_cast<std::size_t>(np)];
  }
}

/// Matrix-free apply of the **same** Hermitian projector as the dense path: build \(P_{\mathrm{raw}}=\frac{1}{|G|}\sum_R
/// \overline{\chi_K(R)}\,U(R)\), then \(P_K=\frac12(P_{\mathrm{raw}}+P_{\mathrm{raw}}^\dagger)\) entrywise matches
/// `detail::hermitian_symmetrize_inplace` on the dense matrix. Implemented as
/// \(P_K x=\frac12(P_{\mathrm{raw}}x+P_{\mathrm{raw}}^\dagger x)\).
///
/// **Stabilizers / orbits:** encoded in `translate_raw_state_rect` / `index_of` / fermionic sign — same as dense \(U(R)\).
/// **Bloch weight:** \(\overline{\chi_K(R)}/|G|\) with \(|G|=L_x L_y\), same as `build_translation_projector_dense`.
///
/// `work` must hold at least `fb.dim()` elements (reused per group element internally).
inline void apply_symmetrized_P_K_vector_matrix_free(const FockBasis& fb, int lx, int ly, MomentumSector K,
                                                     const std::complex<double>* x, std::complex<double>* y_out,
                                                     std::vector<std::complex<double>>* work) {
  const int d = fb.dim();
  work->resize(static_cast<std::size_t>(d));
  const double invg = 1.0 / static_cast<double>(lx * ly);
  std::vector<std::complex<double>> acc1(static_cast<std::size_t>(d), std::complex<double>{0.0, 0.0});
  std::vector<std::complex<double>> acc2(static_cast<std::size_t>(d), std::complex<double>{0.0, 0.0});
  for (int ey = 0; ey < ly; ++ey) {
    for (int ex = 0; ex < lx; ++ex) {
      const std::complex<double> w = std::conj(translation_bloch_phase(K, ex, ey)) * invg;
      apply_translation_U_forward_vector(fb, lx, ly, ex, ey, x, work->data());
      for (int i = 0; i < d; ++i) {
        acc1[static_cast<std::size_t>(i)] += w * (*work)[static_cast<std::size_t>(i)];
      }
      apply_translation_U_transpose_vector(fb, lx, ly, ex, ey, x, work->data());
      for (int i = 0; i < d; ++i) {
        acc2[static_cast<std::size_t>(i)] += std::conj(w) * (*work)[static_cast<std::size_t>(i)];
      }
    }
  }
  for (int i = 0; i < d; ++i) {
    y_out[static_cast<std::size_t>(i)] =
        0.5 * (acc1[static_cast<std::size_t>(i)] + acc2[static_cast<std::size_t>(i)]);
  }
}

inline void hermitian_symmetrize_flat(int n, std::vector<std::complex<double>>* M) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      const std::complex<double> h =
          0.5 * ((*M)[static_cast<std::size_t>(i + j * n)] +
                 std::conj((*M)[static_cast<std::size_t>(j + i * n)]));
      (*M)[static_cast<std::size_t>(i + j * n)] = h;
      (*M)[static_cast<std::size_t>(j + i * n)] = std::conj(h);
    }
  }
}

/// Column-major \(C = A B\) for \(n\times n\) complex matrices.
inline void matmul_colmajor(int n, const std::complex<double>* A, const std::complex<double>* B,
                            std::complex<double>* C) {
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      std::complex<double> s{0.0, 0.0};
      for (int k = 0; k < n; ++k) {
        s += A[static_cast<std::size_t>(i + k * n)] * B[static_cast<std::size_t>(k + j * n)];
      }
      C[static_cast<std::size_t>(i + j * n)] = s;
    }
  }
}

inline void vv_to_colmajor(const std::vector<std::vector<std::complex<double>>>& A,
                           std::vector<std::complex<double>>* flat) {
  const int n = static_cast<int>(A.size());
  flat->assign(static_cast<std::size_t>(n * n), std::complex<double>{0.0, 0.0});
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      (*flat)[static_cast<std::size_t>(i + j * n)] = A[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
    }
  }
}

/// Dense reference: symmetrized \(P_K\) matvec (same as `TinyPkHpkDense::p_colmajor` first factor).
inline void apply_symmetrized_P_K_vector_dense(const FockBasis& fb, int lx, int ly, MomentumSector K,
                                               const std::complex<double>* x, std::complex<double>* y_out,
                                               std::vector<std::complex<double>>* p_flat_buf) {
  const int d = fb.dim();
  std::vector<std::vector<std::complex<double>>> p_vv;
  detail::build_translation_projector_dense(fb, lx, ly, K, &p_vv);
  detail::hermitian_symmetrize_inplace(&p_vv);
  vv_to_colmajor(p_vv, p_flat_buf);
  matvec_colmajor_vector(d, p_flat_buf->data(), x, y_out);
}

/// Full-sector \(y = P_K H P_K x\) with symmetrized \(P_K\) (matrix-free \(P_K\), dense \(H\) via `apply_extended_hubbard`).
inline void apply_symmetrized_PK_H_PK_vector_matrix_free(const HubbardParams& p, const FockBasis& fb, int lx, int ly,
                                                         MomentumSector K, const std::vector<SpinfulHopping>& hoppings,
                                                         const std::vector<NearestPair>& nn_pairs,
                                                         const std::complex<double>* x, std::complex<double>* y_out,
                                                         std::vector<std::complex<double>>* w1,
                                                         std::vector<std::complex<double>>* w2) {
  const int d = fb.dim();
  w1->resize(static_cast<std::size_t>(d));
  w2->resize(static_cast<std::size_t>(d));
  apply_symmetrized_P_K_vector_matrix_free(fb, lx, ly, K, x, w1->data(), w2);
  std::fill(w2->begin(), w2->end(), std::complex<double>{0.0, 0.0});
  apply_extended_hubbard(p, fb, hoppings, nn_pairs, w1->data(), w2->data());
  apply_symmetrized_P_K_vector_matrix_free(fb, lx, ly, K, w2->data(), y_out, w1);
}

/// Dense sector Hamiltonian \(H\) with columns \(H e_j\) from `apply_extended_hubbard`.
inline void build_dense_h_sector(const HubbardParams& p, const FockBasis& fb,
                                 const std::vector<SpinfulHopping>& hoppings,
                                 const std::vector<NearestPair>& nn_pairs, std::vector<std::complex<double>>* h_out) {
  const int n = fb.dim();
  h_out->assign(static_cast<std::size_t>(n * n), std::complex<double>{0.0, 0.0});
  std::vector<std::complex<double>> ej(static_cast<std::size_t>(n), std::complex<double>{0.0, 0.0});
  std::vector<std::complex<double>> col(static_cast<std::size_t>(n), std::complex<double>{0.0, 0.0});
  for (int j = 0; j < n; ++j) {
    std::fill(ej.begin(), ej.end(), std::complex<double>{0.0, 0.0});
    ej[static_cast<std::size_t>(j)] = std::complex<double>{1.0, 0.0};
    apply_extended_hubbard(p, fb, hoppings, nn_pairs, ej.data(), col.data());
    for (int i = 0; i < n; ++i) {
      (*h_out)[static_cast<std::size_t>(i + j * n)] = col[static_cast<std::size_t>(i)];
    }
  }
  hermitian_symmetrize_flat(n, h_out);
}

/// Holds column-major \(P_K\), \(H\), and \(M = P_K H P_K\) on the full sector.
struct TinyPkHpkDense {
  int d_full = 0;
  std::vector<std::complex<double>> p_colmajor{};
  std::vector<std::complex<double>> h_colmajor{};
  /// Full-space momentum-projected Hamiltonian \(P_K H P_K\).
  std::vector<std::complex<double>> m_colmajor{};

  /// `y += M x` with \(M = P_K H P_K\) (column-major).
  void apply_full(const std::complex<double>* x, std::complex<double>* y) const {
    const int n = d_full;
    for (int i = 0; i < n; ++i) {
      std::complex<double> s{0.0, 0.0};
      for (int j = 0; j < n; ++j) {
        s += m_colmajor[static_cast<std::size_t>(i + j * n)] * x[static_cast<std::size_t>(j)];
      }
      y[static_cast<std::size_t>(i)] += s;
    }
  }
};

/// Build independent dense \(P_K\), \(H\), and \(M = P_K H P_K\). Caps `fb.dim()` by `max_d_full` (default 64).
inline TinyPkHpkDense build_tiny_pk_h_pk_dense(const HubbardParams& p, const FockBasis& fb, int lx, int ly,
                                               MomentumSector K, const std::vector<SpinfulHopping>& hoppings,
                                               const std::vector<NearestPair>& nn_pairs, int max_d_full = 64) {
  const int d = fb.dim();
  if (d > max_d_full) {
    throw std::invalid_argument("build_tiny_pk_h_pk_dense: sector dimension exceeds max_d_full (prototype only)");
  }
  TinyPkHpkDense out;
  out.d_full = d;
  std::vector<std::vector<std::complex<double>>> p_vv;
  detail::build_translation_projector_dense(fb, lx, ly, K, &p_vv);
  detail::hermitian_symmetrize_inplace(&p_vv);
  vv_to_colmajor(p_vv, &out.p_colmajor);
  build_dense_h_sector(p, fb, hoppings, nn_pairs, &out.h_colmajor);
  out.m_colmajor.assign(static_cast<std::size_t>(d * d), std::complex<double>{0.0, 0.0});
  std::vector<std::complex<double>> tmp(static_cast<std::size_t>(d * d));
  matmul_colmajor(d, out.p_colmajor.data(), out.h_colmajor.data(), tmp.data());
  matmul_colmajor(d, tmp.data(), out.p_colmajor.data(), out.m_colmajor.data());
  hermitian_symmetrize_flat(d, &out.m_colmajor);
  return out;
}

/// \(\Phi^\dagger M \Phi\) with \(\Phi\) column-major `d_full × dk`, \(M\) column-major `d_full × d_full`;
/// result `out` is `dk × dk` column-major.
inline void phi_dagger_M_phi(int d_full, int dk, const std::complex<double>* phi_colmajor,
                             const std::complex<double>* m_colmajor, std::complex<double>* out_colmajor) {
  std::vector<std::complex<double>> mphi(static_cast<std::size_t>(d_full * dk), std::complex<double>{0.0, 0.0});
  for (int j = 0; j < dk; ++j) {
    for (int i = 0; i < d_full; ++i) {
      std::complex<double> s{0.0, 0.0};
      for (int k = 0; k < d_full; ++k) {
        s += m_colmajor[static_cast<std::size_t>(i + k * d_full)] *
             phi_colmajor[static_cast<std::size_t>(k + j * d_full)];
      }
      mphi[static_cast<std::size_t>(i + j * d_full)] = s;
    }
  }
  for (int j = 0; j < dk; ++j) {
    for (int i = 0; i < dk; ++i) {
      std::complex<double> s{0.0, 0.0};
      for (int p = 0; p < d_full; ++p) {
        s += std::conj(phi_colmajor[static_cast<std::size_t>(p + i * d_full)]) *
             mphi[static_cast<std::size_t>(p + j * d_full)];
      }
      out_colmajor[static_cast<std::size_t>(i + j * dk)] = s;
    }
  }
}

/// \(\Phi^\dagger H \Phi\) (production-style sandwich without extra \(P_K\) on \(H\)).
inline void phi_dagger_H_phi(int d_full, int dk, const std::complex<double>* phi_colmajor,
                             const std::complex<double>* h_colmajor, std::complex<double>* out_colmajor) {
  phi_dagger_M_phi(d_full, dk, phi_colmajor, h_colmajor, out_colmajor);
}

}  // namespace tiny_pk
}  // namespace symmetry
}  // namespace ftlm
