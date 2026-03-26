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
#include "ftlm/symmetry/translation_projector_dense.hpp"

namespace ftlm {
namespace symmetry {
namespace tiny_pk {

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
