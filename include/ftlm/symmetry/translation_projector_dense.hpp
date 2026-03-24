#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "ftlm/fock_basis.hpp"
#include "ftlm/symmetry/fermionic_translation.hpp"
#include "ftlm/symmetry/k_basis.hpp"
#include "ftlm/symmetry/momentum_sector.hpp"
#include "ftlm/symmetry/orbit.hpp"
#include "ftlm/symmetry/packed_raw_state.hpp"
#include "ftlm/symmetry/raw_state.hpp"

namespace ftlm {
namespace symmetry {
namespace detail {

inline std::size_t stabilizer_size_of_rep(const MomentumSectorMap& map, RawState r) {
  const std::uint32_t pk = pack_raw_state(r);
  for (const auto& o : map.orbits) {
    if (pack_raw_state(o.representative) == pk) {
      return o.stabilizer.size();
    }
  }
  throw std::out_of_range("translation_projector_dense: representative not in orbit_map");
}

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

void eigh_hermitian_small(const std::vector<std::vector<std::complex<double>>>& H, std::vector<double>* evals,
                          std::vector<std::vector<std::complex<double>>>* evec_columns, double tol_pair);

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

/// Column-major Phi[d * dk]: orthonormal basis of range(P_k) from the spectral decomposition of Hermitian P_k.
///
/// **Rank** is the number of eigenvalues above a relative tolerance (not a fixed 0.5 cutoff and not `KBasis::dim()`).
/// Stabilizer-only orbit counting can disagree with the fermionic translation algebra (e.g. Γ projector may vanish
/// while an orbit mask still marks reps as “Γ-compatible”).
inline void build_phi_from_translation_projector_dense(const FockBasis& fb, int lx, int ly, MomentumSector K,
                                                       std::vector<std::complex<double>>* phi, std::size_t* dk_out) {
  const int d = fb.dim();
  std::vector<std::vector<std::complex<double>>> P;
  build_translation_projector_dense(fb, lx, ly, K, &P);
  hermitian_symmetrize_inplace(&P);

  std::vector<double> evals;
  std::vector<std::vector<std::complex<double>>> evecs;
  eigh_hermitian_small(P, &evals, &evecs, 1e-8);

  double eval_max = 0.0;
  for (int j = 0; j < d; ++j) {
    eval_max = std::max(eval_max, evals[static_cast<std::size_t>(j)]);
  }
  const double tol = std::max(1e-10, 1e-12 * std::max(1.0, eval_max));

  std::vector<std::pair<double, int>> scored;
  scored.reserve(static_cast<std::size_t>(d));
  for (int j = 0; j < d; ++j) {
    const double ev = evals[static_cast<std::size_t>(j)];
    if (ev > tol) {
      scored.push_back({-ev, j});
    }
  }
  std::sort(scored.begin(), scored.end());
  const std::size_t dk = scored.size();
  if (dk_out != nullptr) {
    *dk_out = dk;
  }

  phi->assign(static_cast<std::size_t>(d) * dk, std::complex<double>(0.0, 0.0));
  for (std::size_t jc = 0; jc < dk; ++jc) {
    const int col_src = scored[static_cast<std::size_t>(jc)].second;
    for (int p = 0; p < d; ++p) {
      (*phi)[p + jc * static_cast<std::size_t>(d)] = evecs[static_cast<std::size_t>(p)][static_cast<std::size_t>(col_src)];
    }
  }
}

inline void fill_phi_orbit_bloch_bitonly(const MomentumSectorMap& orbit_map, MomentumSector K, const KBasis& basis, int lx,
                                       int ly, const FockBasis& fb, int jcol, std::complex<double>* col) {
  const int d_full = fb.dim();
  std::fill(col, col + d_full, std::complex<double>(0.0, 0.0));
  const RawState r = basis.representatives[static_cast<std::size_t>(jcol)];
  const std::size_t stab = stabilizer_size_of_rep(orbit_map, r);
  const double c = momentum_orbit_normalization_factor_rect(stab, lx, ly);
  for (int ey = 0; ey < ly; ++ey) {
    for (int ex = 0; ex < lx; ++ex) {
      const RawState st = translate_raw_state_rect(r, lx, ly, ex, ey);
      const int idx = fb.index_of(static_cast<std::uint64_t>(st.up), static_cast<std::uint64_t>(st.dn));
      if (idx >= 0) {
        col[idx] += c * std::conj(translation_bloch_phase(K, ex, ey));
      }
    }
  }
}

}  // namespace detail
}  // namespace symmetry
}  // namespace ftlm
