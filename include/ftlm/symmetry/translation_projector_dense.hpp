#pragma once

// Optional dense translation projector P_k = (1/|G|) ∑_R χ_k(R)^† U(R) for diagnostics / benchmarks only.
// Production momentum blocks use orbit_bloch_phi.hpp (no d_full×d_full allocation).

#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

#include "ftlm/fock_basis.hpp"
#include "ftlm/symmetry/fermionic_translation.hpp"
#include "ftlm/symmetry/momentum_sector.hpp"
#include "ftlm/symmetry/raw_state.hpp"

namespace ftlm {
namespace symmetry {
namespace detail {

inline RawState raw_from_fock_index(const FockBasis& fb, int idx) {
  return RawState{static_cast<std::uint16_t>(fb.up_mask(idx) & 0xFFFFu),
                  static_cast<std::uint16_t>(fb.down_mask(idx) & 0xFFFFu)};
}

inline void build_dense_translation_U(const FockBasis& fb, int lx, int ly, int ex, int ey,
                                      std::vector<std::vector<std::complex<double>>>* U) {
  const int d = fb.dim();
  U->assign(static_cast<std::size_t>(d), std::vector<std::complex<double>>(static_cast<std::size_t>(d), {0.0, 0.0}));
  for (int n = 0; n < d; ++n) {
    const RawState s = raw_from_fock_index(fb, n);
    const RawState st = translate_raw_state_rect(s, lx, ly, ex, ey);
    const int np = fb.index_of(static_cast<std::uint64_t>(st.up), static_cast<std::uint64_t>(st.dn));
    if (np < 0) {
      continue;
    }
    const double eta = fermionic_translation_sign_rect(s, lx, ly, ex, ey);
    (*U)[static_cast<std::size_t>(np)][static_cast<std::size_t>(n)] = {eta, 0.0};
  }
}

/// Hermitian part of a nearly Hermitian matrix (numerical noise in group sums).
inline void hermitian_symmetrize_inplace(std::vector<std::vector<std::complex<double>>>* M) {
  const int n = static_cast<int>(M->size());
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      const std::complex<double> h =
          0.5 * ((*M)[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] +
                 std::conj((*M)[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)]));
      (*M)[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = h;
      (*M)[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] = std::conj(h);
    }
  }
}

/// P_k = (1/|G|) ∑_R conj(χ_k(R)) U(R) with fermionic signed permutations U(R).
inline void build_translation_projector_dense(const FockBasis& fb, int lx, int ly, MomentumSector K,
                                              std::vector<std::vector<std::complex<double>>>* P) {
  const int d = fb.dim();
  const double invg = 1.0 / static_cast<double>(lx * ly);
  P->assign(static_cast<std::size_t>(d), std::vector<std::complex<double>>(static_cast<std::size_t>(d), {0.0, 0.0}));
  std::vector<std::vector<std::complex<double>>> U;
  for (int ey = 0; ey < ly; ++ey) {
    for (int ex = 0; ex < lx; ++ex) {
      build_dense_translation_U(fb, lx, ly, ex, ey, &U);
      const std::complex<double> w = std::conj(translation_bloch_phase(K, ex, ey)) * invg;
      for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
          (*P)[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] +=
              w * U[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
        }
      }
    }
  }
}

}  // namespace detail
}  // namespace symmetry
}  // namespace ftlm
