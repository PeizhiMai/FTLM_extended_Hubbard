#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

#include "ftlm/fock_basis.hpp"
#include "ftlm/symmetry/fermionic_translation.hpp"  // fermionic_translation_sign_rect — must match dense P_K = (1/|G|)∑ χ* U(R)
#include "ftlm/symmetry/k_basis.hpp"  // KBasis + momentum_phi_seeds
#include "ftlm/symmetry/momentum_sector.hpp"
#include "ftlm/symmetry/orbit.hpp"
#include "ftlm/symmetry/packed_raw_state.hpp"
#include "ftlm/symmetry/raw_state.hpp"

namespace ftlm {
namespace symmetry {
namespace detail {

/// Stabilizer size for the translation orbit containing `r` (any member), matching `OrbitRecord` for its canonical rep.
inline std::size_t stabilizer_size_of_rep(const MomentumSectorMap& map, RawState r) {
  const RawState canon = canonical_raw_state_rect(r, map.lx, map.ly);
  const std::uint32_t pk = pack_raw_state(canon);
  for (const auto& o : map.orbits) {
    if (pack_raw_state(o.representative) == pk) {
      return o.stabilizer.size();
    }
  }
  throw std::out_of_range("orbit_bloch_phi: state not in orbit_map");
}

/// One column of Φ: Bloch sum starting from ket `seed` (any state in its translation orbit), matching dense
/// \(P_K=(1/|G|)\sum_R \chi_K^*(R)\,U(R)\) with fermionic \(U(R)\) (`fermionic_translation_sign_rect`).
inline void fill_phi_orbit_bloch_from_seed(const MomentumSectorMap& orbit_map, MomentumSector K, int lx, int ly,
                                           const FockBasis& fb, RawState seed, std::complex<double>* col) {
  const int d_full = fb.dim();
  std::fill(col, col + d_full, std::complex<double>(0.0, 0.0));
  const std::size_t stab = stabilizer_size_of_rep(orbit_map, seed);
  const double c = momentum_orbit_normalization_factor_rect(stab, lx, ly);
  for (int ey = 0; ey < ly; ++ey) {
    for (int ex = 0; ex < lx; ++ex) {
      const RawState st = translate_raw_state_rect(seed, lx, ly, ex, ey);
      const int idx = fb.index_of(static_cast<std::uint64_t>(st.up), static_cast<std::uint64_t>(st.dn));
      if (idx >= 0) {
        const double eta = fermionic_translation_sign_rect(seed, lx, ly, ex, ey);
        col[idx] += c * eta * std::conj(translation_bloch_phase(K, ex, ey));
      }
    }
  }
}

/// Legacy helper: one column per canonical `KBasis` representative only (does not span `P_K` when `|G|/|S|>1`).
inline void fill_phi_orbit_bloch_bitonly(const MomentumSectorMap& orbit_map, MomentumSector K, const KBasis& basis, int lx,
                                         int ly, const FockBasis& fb, int jcol, std::complex<double>* col) {
  fill_phi_orbit_bloch_from_seed(orbit_map, K, lx, ly, fb, basis.representatives[static_cast<std::size_t>(jcol)], col);
}

}  // namespace detail

/// Column-major Φ: modified Gram–Schmidt with dependent columns dropped (linear combinations of earlier columns).
/// Returns the number of orthonormal columns kept (≤ `dk_in`). Output is compact column-major `phi[p + j*d_full]`, `j < dk_out`.
inline std::size_t gram_schmidt_orthonormalize_phi_columns(int d_full, std::size_t dk_in,
                                                             std::vector<std::complex<double>>* phi_column_major) {
  if (d_full <= 0 || dk_in == 0) {
    phi_column_major->clear();
    return 0;
  }
  double max_norm = 0.0;
  for (std::size_t j = 0; j < dk_in; ++j) {
    double s = 0.0;
    const std::complex<double>* col = phi_column_major->data() + j * static_cast<std::size_t>(d_full);
    for (int p = 0; p < d_full; ++p) {
      s += std::norm(col[static_cast<std::size_t>(p)]);
    }
    max_norm = std::max(max_norm, std::sqrt(s));
  }
  const double tol =
      1e-12 * std::max(1.0, max_norm) * std::sqrt(static_cast<double>(std::max(1, d_full)));

  std::vector<std::complex<double>> out;
  out.reserve(static_cast<std::size_t>(d_full) * dk_in);
  std::vector<std::complex<double>> col(static_cast<std::size_t>(d_full));

  for (std::size_t j = 0; j < dk_in; ++j) {
    for (int p = 0; p < d_full; ++p) {
      col[static_cast<std::size_t>(p)] =
          (*phi_column_major)[static_cast<std::size_t>(p) + j * static_cast<std::size_t>(d_full)];
    }
    const std::size_t out_cols = out.size() / static_cast<std::size_t>(d_full);
    for (std::size_t k = 0; k < out_cols; ++k) {
      const std::complex<double>* colk_ptr = out.data() + k * static_cast<std::size_t>(d_full);
      std::complex<double> ip{0.0, 0.0};
      for (int p = 0; p < d_full; ++p) {
        ip += std::conj(colk_ptr[static_cast<std::size_t>(p)]) * col[static_cast<std::size_t>(p)];
      }
      for (int p = 0; p < d_full; ++p) {
        col[static_cast<std::size_t>(p)] -= ip * colk_ptr[static_cast<std::size_t>(p)];
      }
    }
    double nrm2 = 0.0;
    for (int p = 0; p < d_full; ++p) {
      nrm2 += std::norm(col[static_cast<std::size_t>(p)]);
    }
    const double nrm = std::sqrt(nrm2);
    if (nrm < tol) {
      continue;
    }
    const double inv = 1.0 / nrm;
    for (int p = 0; p < d_full; ++p) {
      out.push_back(col[static_cast<std::size_t>(p)] * inv);
    }
  }
  phi_column_major->swap(out);
  return phi_column_major->size() / static_cast<std::size_t>(d_full);
}

/// Build Φ from `momentum_phi_seeds` (one Bloch column per orbit member), then Gram whitening on `k×k`.
///
/// **Reference / tests only**: materializes dense Φ (`d_full × d_k`). Production momentum blocks use
/// `MomentumPhiGramBasis` instead (matrix-free lift/project).
void build_momentum_phi_orbit_orthonormal(const MomentumSectorMap& orbit_map, MomentumSector K, int lx, int ly,
                                          const FockBasis& fb, std::vector<std::complex<double>>* phi_column_major,
                                          std::size_t* dk_out);

/// LAPACK `zheev` workspace reused across Gram builds (replaces unbounded `thread_local` in `zheev_full_hermitian_inplace`).
struct ZheevHermitianScratch {
  std::vector<std::complex<double>> work{};
  std::vector<double> rwork{};
  void shrink_to_fit() {
    work.shrink_to_fit();
    rwork.shrink_to_fit();
  }
};

/// Reusable temporaries for `MomentumPhiGramBasis::project_block_from_full` / `lift_full_from_block` (avoids
/// unbounded `thread_local` growth when iterating many momentum sectors).
struct MomentumPhiGramApplyScratch {
  std::vector<std::complex<double>> a{};
  std::vector<std::complex<double>> w{};
  std::vector<std::complex<double>> col{};
  void ensure(std::size_t k_in, int d_full) {
    if (k_in == 0 || d_full <= 0) {
      return;
    }
    a.resize(k_in);
    w.resize(k_in);
    col.resize(static_cast<std::size_t>(d_full));
  }
  void shrink_to_fit() {
    a.shrink_to_fit();
    w.shrink_to_fit();
    col.shrink_to_fit();
  }
};

/// Matrix-free data for the **same** orthonormal Bloch basis as `build_momentum_phi_orbit_orthonormal`:
/// raw Bloch columns from `seeds`, Gram matrix \(G=\Phi_{\mathrm{raw}}^\dagger \Phi_{\mathrm{raw}}\), Hermitian
/// diagonalization \(G = V \Lambda V^\dagger\), then \(\Phi = \Phi_{\mathrm{raw}} V \Lambda^{-1/2}\) with small
/// eigenvalues dropped. **No** dense `d_full × d_k` storage — \(O(k_{\mathrm{in}} k_{\mathrm{out}})\) for the kept slice of \(V\) plus seeds.
struct MomentumPhiGramBasis {
  int d_full = 0;
  int lx = 0;
  int ly = 0;
  std::vector<RawState> seeds{};
  /// Columns of \(V\) for **kept** eigenpairs only: column-major `k_in × k_out` (whitening \(\Phi=\Phi_{\mathrm{raw}} V\Lambda^{-1/2}\)).
  std::vector<std::complex<double>> V{};
  /// Eigenvalues \(\lambda_j\) aligned with columns of `V` (length `k_out`).
  std::vector<double> evals{};
  std::size_t k_in = 0;
  std::size_t k_out = 0;

  /// Build Gram data from orbit Bloch seeds. Returns false if the block is empty.
  /// If `zheev_scratch` is non-null, LAPACK workspace is stored there; otherwise local buffers are used (tests / one-off).
  static bool build(const MomentumSectorMap& orbit_map, MomentumSector K, int lx, int ly, const FockBasis& fb,
                    MomentumPhiGramBasis* out, ZheevHermitianScratch* zheev_scratch = nullptr);

  /// `y_block = Φ† x_full` (block length `k_out`).
  void project_block_from_full(const MomentumSectorMap& orbit_map, MomentumSector K, const FockBasis& fb,
                               const std::complex<double>* x_full, std::complex<double>* y_block,
                               MomentumPhiGramApplyScratch* scratch = nullptr) const;

  /// `x_full = Φ y_block` (full length `d_full`).
  void lift_full_from_block(const MomentumSectorMap& orbit_map, MomentumSector K, const FockBasis& fb,
                            const std::complex<double>* y_block, std::complex<double>* x_full,
                            MomentumPhiGramApplyScratch* scratch = nullptr) const;

  /// Bytes of persistent storage (V, evals, seeds, indices) — **zero** dense Φ bytes.
  std::size_t storage_bytes() const noexcept;
};

}  // namespace symmetry
}  // namespace ftlm
