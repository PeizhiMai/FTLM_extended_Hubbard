#pragma once

// Translation-orbit K-block metadata for matrix-free H_K (no Gram, no zheev, no dense P_K).
// Design formulas: docs/ORBIT_K_BLOCK_DESIGN.md

#include <complex>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ftlm/fock_basis.hpp"
#include "ftlm/hubbard_params.hpp"
#include "ftlm/lattice.hpp"
#include "ftlm/symmetry/momentum_sector.hpp"
#include "ftlm/symmetry/raw_state.hpp"

namespace ftlm {
namespace symmetry {

/// χ_K^*(R) as returned by `translation_bloch_phase(K, dx, dy)` (same factor as in Bloch column sums).
/// The reduced hop apply uses this **unconjugated** phase; do not use `conj(translation_bloch_phase)` there.
inline std::complex<double> momentum_character_chi_K(MomentumSector K, int dx, int dy) {
  return std::conj(translation_bloch_phase(K, dx, dy));
}

/// Fermionic stabilizer sum A_α(K) = Σ_{S∈Stab(α)} η(α,S) χ_K^*(S).
/// Stabilizer list must match `OrbitRecord::stabilizer` for α.
std::complex<double> fermionic_stabilizer_character_A_rect(
    const RawState& rep, MomentumSector K, int lx, int ly,
    const std::vector<std::pair<std::int8_t, std::int8_t>>& stabilizer);

/// One surviving orbit in the reduced K basis (one basis direction per orbit).
struct OrbitKRepEntry {
  RawState rep{};
  std::size_t orbit_index = 0;  // index into `MomentumSectorMap::orbits`
  std::uint16_t orbit_size = 0;
  std::uint16_t stabilizer_size = 0;
  std::complex<double> A_k{0.0, 0.0};
  double abs_A = 0.0;  // |A_k|; for surviving orbits, sqrt weights use this
};

/// Metadata for matrix-free K projection on translation-orbit representatives.
class OrbitKBlockBasis {
 public:
  MomentumSector K{};
  int lx = 0;
  int ly = 0;

  /// Build from particle sector `fb` and momentum `K` (uses `K.lx`, `K.ly`).
  /// Orbits with |A_α(K)| <= tol are excluded. Requires `fb.n_sites() <= 16` for `RawState` packing.
  bool build(const FockBasis& fb, MomentumSector K, double survive_tol = 1e-12);

  int dim() const noexcept { return static_cast<int>(entries_.size()); }

  const std::vector<OrbitKRepEntry>& entries() const noexcept { return entries_; }

  double survive_tol() const noexcept { return survive_tol_; }

  /// Index of reduced coordinate for canonical rep `rep`, or -1 if not in basis.
  int reduced_index_of_rep(RawState rep) const;

  /// Map `s` to its orbit's reduced index and translation with **T_R rep = s** (see design doc).
  /// Returns false if `s` is not in the sector universe or its orbit does not survive |A_k|>tol.
  bool map_raw_to_rep(RawState s, int* reduced_index, int* dx, int* dy, double* eta) const;

 private:
  std::vector<OrbitKRepEntry> entries_;
  /// Canonical rep pack -> row in `entries_`
  std::unordered_map<std::uint32_t, std::size_t> rep_pack_to_row_;
  double survive_tol_ = 1e-12;
};

/// Matrix-free y = H_K x in the orbit-representative basis.
///
/// Formula (T_R alpha = s' convention; matches Φ† H Φ in orbit coordinates, see
/// `benchmarks/diag_orbit_h_matrix_compare.cpp`):
///   y_alpha += h_{s'beta} * eta(alpha,R) * chi_K^*(R) * sqrt(|A_alpha|/|A_beta|) * x_beta
/// where chi_K^*(R) = `translation_bloch_phase(K,dx,dy)` and h_{s'beta} is the full-space amplitude rep(beta)→s'.
void apply_orbit_k_block(const HubbardParams& p, const FockBasis& fb, const OrbitKBlockBasis& basis,
                         const std::vector<SpinfulHopping>& hoppings,
                         const std::vector<NearestPair>& nn_pairs,
                         const std::complex<double>* x,
                         std::complex<double>* y);

}  // namespace symmetry
}  // namespace ftlm

