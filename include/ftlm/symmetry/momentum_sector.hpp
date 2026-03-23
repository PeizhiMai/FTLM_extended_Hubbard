#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "ftlm/symmetry/lattice_4x4.hpp"
#include "ftlm/symmetry/orbit.hpp"
#include "ftlm/symmetry/packed_raw_state.hpp"
#include "ftlm/symmetry/translation_group.hpp"

namespace ftlm {
namespace symmetry {

/// Bloch momentum on the discrete torus: \(k_x,k_y \in \{0,1,2,3\}\) (units of \(2\pi/L\) with \(L=4\)).
struct MomentumSector {
  int kx = 0;
  int ky = 0;

  constexpr int flat_index() const noexcept { return momentum_flat_index(kx, ky); }

  /// Orbit contributes iff χ_k is trivial on its stabilizer (same bit as `compatible_momentum_mask`).
  bool is_compatible(const OrbitRecord& r) const noexcept {
    const std::uint16_t bit = static_cast<std::uint16_t>(1u << flat_index());
    return (r.compatible_momentum_mask & bit) != 0;
  }
};

/// One-dimensional characters of Z₄×Z₄: χ_k(T(dx,dy)) = exp( i 2π (kx·dx + ky·dy) / 4 ).
/// Trivial on a translation iff (kx·dx + ky·dy) ≡ 0 (mod 4).
inline bool momentum_character_trivial(int kx, int ky, int dx, int dy) {
  const int s = kx * dx + ky * dy;
  return Lattice4x4::imod(s, 4) == 0;
}

/// Phase \(\exp(-\mathrm{i}\,\mathbf{k}\cdot\mathbf{R})\) for translation by \((dx,dy)\) lattice steps,
/// with \(\mathbf{k}\cdot\mathbf{R} \equiv 2\pi(k_x dx + k_y dy)/4\). Matches χ_k^* on the group generator.
inline std::complex<double> translation_bloch_phase(MomentumSector K, int dx, int dy) {
  const int n = Lattice4x4::imod(K.kx * dx + K.ky * dy, 4);
  const double theta = -0.5 * std::acos(-1.0) * static_cast<double>(n);  // -2π n / 4
  return {std::cos(theta), std::sin(theta)};
}

/// Amplitude prefactor for the normalized Bloch combination
/// \(\lvert k,\mathrm{orbit}\rangle = \texttt{factor} \sum_{g\in G} \chi_k(g)^\dagger T_g \lvert\text{rep}\rangle\)
/// with \(|G|=16\). When χ_k is trivial on the stabilizer, \(\lVert\lvert k,\mathrm{orbit}\rangle\rVert=1\)
/// in the orthonormal unprojected basis.
inline double momentum_orbit_normalization_factor(std::size_t stabilizer_size) noexcept {
  const double g = static_cast<double>(translation_group_order());
  return 1.0 / std::sqrt(g * static_cast<double>(stabilizer_size));
}

/// Bit mask over `momentum_flat_index(kx,ky)` for kx,ky ∈ {0,…,3}.
inline std::uint16_t compatible_momentum_mask_for_stabilizer(
    const std::vector<std::pair<std::int8_t, std::int8_t>>& stabilizer) {
  std::uint16_t mask = 0;
  for (int ky = 0; ky < 4; ++ky) {
    for (int kx = 0; kx < 4; ++kx) {
      bool ok = true;
      for (const auto& e : stabilizer) {
        if (!momentum_character_trivial(kx, ky, e.first, e.second)) {
          ok = false;
          break;
        }
      }
      if (ok) {
        mask |= static_cast<std::uint16_t>(1u << momentum_flat_index(kx, ky));
      }
    }
  }
  return mask;
}

/// Fill `compatible_momentum_mask` from `stabilizer` (does not touch other fields).
inline void attach_compatible_momenta(OrbitRecord* r) {
  r->compatible_momentum_mask = compatible_momentum_mask_for_stabilizer(r->stabilizer);
}

/// Matrix-free view: which translation orbits contribute to each momentum shell (by orbit index).
struct MomentumSectorMap {
  std::vector<OrbitRecord> orbits;
  /// For flat momentum index m ∈ [0,15], lists orbit indices whose representative can be symmetrized into that sector.
  std::array<std::vector<std::size_t>, 16> orbit_indices_by_momentum{};
};

inline std::vector<RawState> unique_sorted_raw_states(std::vector<RawState> states) {
  std::vector<std::uint32_t> packs;
  packs.reserve(states.size());
  for (RawState s : states) {
    packs.push_back(pack_raw_state(s));
  }
  std::sort(packs.begin(), packs.end());
  packs.erase(std::unique(packs.begin(), packs.end()), packs.end());
  std::vector<RawState> out;
  out.reserve(packs.size());
  for (std::uint32_t p : packs) {
    out.push_back(unpack_raw_state(p));
  }
  return out;
}

/// Partition `universe` into disjoint translation orbits; for each orbit store canonical data and momentum compatibility.
/// `universe` is deduplicated by packed `RawState` before processing.
inline MomentumSectorMap build_momentum_sector_map(std::vector<RawState> universe) {
  universe = unique_sorted_raw_states(std::move(universe));
  std::unordered_map<std::uint32_t, std::size_t> pack_to_orbit_index;
  pack_to_orbit_index.reserve(universe.size() * 2);

  MomentumSectorMap map;
  for (RawState s : universe) {
    const std::uint32_t pk = pack_raw_state(s);
    if (pack_to_orbit_index.find(pk) != pack_to_orbit_index.end()) {
      continue;
    }
    const RawState rep = canonical_raw_state(s);
    std::vector<RawState> members;
    enumerate_translation_orbit(rep, &members);
    const std::size_t oid = map.orbits.size();
    for (RawState m : members) {
      pack_to_orbit_index[pack_raw_state(m)] = oid;
    }
    OrbitRecord rec;
    rec.representative = rep;
    rec.orbit_size = static_cast<std::uint16_t>(members.size());
    rec.period_tx = primitive_period_tx(rep);
    rec.period_ty = primitive_period_ty(rep);
    rec.stabilizer = translation_stabilizer(rep);
    attach_compatible_momenta(&rec);
    map.orbits.push_back(std::move(rec));
  }

  for (auto& row : map.orbit_indices_by_momentum) {
    row.clear();
  }
  for (std::size_t oi = 0; oi < map.orbits.size(); ++oi) {
    const std::uint16_t mmask = map.orbits[oi].compatible_momentum_mask;
    for (int k = 0; k < 16; ++k) {
      if ((mmask >> k) & 1) {
        map.orbit_indices_by_momentum[static_cast<std::size_t>(k)].push_back(oi);
      }
    }
  }
  return map;
}

/// Number of set bits in the 16-bit momentum mask (should equal `orbit_size` for a valid translation orbit).
inline int popcount_momentum_mask(std::uint16_t m) {
  int c = 0;
  for (int i = 0; i < 16; ++i) {
    if ((m >> i) & 1) {
      ++c;
    }
  }
  return c;
}

}  // namespace symmetry
}  // namespace ftlm
