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
  int lx = 4;
  int ly = 4;

  constexpr int flat_index() const noexcept { return ky * lx + kx; }

  /// Orbit contributes iff χ_k is trivial on its stabilizer (same bit as `compatible_momentum_mask`).
  bool is_compatible(const OrbitRecord& r) const noexcept {
    const std::uint16_t bit = static_cast<std::uint16_t>(1u << flat_index());
    return (r.compatible_momentum_mask & bit) != 0;
  }
};

inline bool momentum_character_trivial_rect(int kx, int ky, int lx, int ly, int dx, int dy) {
  int a = lx;
  int b = ly;
  while (b != 0) {
    const int t = a % b;
    a = b;
    b = t;
  }
  const int g = (a == 0) ? 1 : a;
  const int lcm = (lx / g) * ly;
  const int sx = kx * dx * (lcm / lx);
  const int sy = ky * dy * (lcm / ly);
  return Lattice4x4::imod(sx + sy, lcm) == 0;
}

/// Backward-compatible 4x4 character triviality.
inline bool momentum_character_trivial(int kx, int ky, int dx, int dy) {
  return momentum_character_trivial_rect(kx, ky, 4, 4, dx, dy);
}

/// Phase \(\exp(-\mathrm{i}\,\mathbf{k}\cdot\mathbf{R})\) for translation by \((dx,dy)\) lattice steps,
/// with \(\mathbf{k}\cdot\mathbf{R} \equiv 2\pi(k_x dx + k_y dy)/4\). Matches χ_k^* on the group generator.
inline std::complex<double> translation_bloch_phase(MomentumSector K, int dx, int dy) {
  const double pi = std::acos(-1.0);
  const double theta = -2.0 * pi *
                       (static_cast<double>(K.kx * dx) / static_cast<double>(K.lx) +
                        static_cast<double>(K.ky * dy) / static_cast<double>(K.ly));
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

inline double momentum_orbit_normalization_factor_rect(std::size_t stabilizer_size, int lx, int ly) noexcept {
  const double g = static_cast<double>(translation_group_order(lx, ly));
  return 1.0 / std::sqrt(g * static_cast<double>(stabilizer_size));
}

/// Bit mask over `momentum_flat_index(kx,ky)` for kx,ky ∈ {0,…,3}.
inline std::uint16_t compatible_momentum_mask_for_stabilizer(
    const std::vector<std::pair<std::int8_t, std::int8_t>>& stabilizer, int lx = 4, int ly = 4) {
  std::uint32_t mask = 0;
  for (int ky = 0; ky < ly; ++ky) {
    for (int kx = 0; kx < lx; ++kx) {
      bool ok = true;
      for (const auto& e : stabilizer) {
        if (!momentum_character_trivial_rect(kx, ky, lx, ly, e.first, e.second)) {
          ok = false;
          break;
        }
      }
      if (ok) {
        mask |= static_cast<std::uint32_t>(1u << (ky * lx + kx));
      }
    }
  }
  return static_cast<std::uint16_t>(mask);
}

/// Fill `compatible_momentum_mask` from `stabilizer` (does not touch other fields).
inline void attach_compatible_momenta(OrbitRecord* r) {
  r->compatible_momentum_mask = compatible_momentum_mask_for_stabilizer(r->stabilizer, 4, 4);
}

/// Matrix-free view: which translation orbits contribute to each momentum shell (by orbit index).
struct MomentumSectorMap {
  int lx = 4;
  int ly = 4;
  std::vector<OrbitRecord> orbits;
  /// For flat momentum index m ∈ [0,lx*ly), lists orbit indices whose representative can be symmetrized into that sector.
  std::vector<std::vector<std::size_t>> orbit_indices_by_momentum{};
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

inline MomentumSectorMap build_momentum_sector_map_rect(std::vector<RawState> universe, int lx, int ly);

/// Partition `universe` into disjoint translation orbits; for each orbit store canonical data and momentum compatibility.
/// `universe` is deduplicated by packed `RawState` before processing.
inline MomentumSectorMap build_momentum_sector_map(std::vector<RawState> universe) {
  return build_momentum_sector_map_rect(std::move(universe), 4, 4);
}

inline MomentumSectorMap build_momentum_sector_map_rect(std::vector<RawState> universe, int lx, int ly) {
  universe = unique_sorted_raw_states(std::move(universe));
  std::unordered_map<std::uint32_t, std::size_t> pack_to_orbit_index;
  pack_to_orbit_index.reserve(universe.size() * 2);

  MomentumSectorMap map;
  map.lx = lx;
  map.ly = ly;
  map.orbit_indices_by_momentum.assign(static_cast<std::size_t>(lx * ly), {});
  for (RawState s : universe) {
    const std::uint32_t pk = pack_raw_state(s);
    if (pack_to_orbit_index.find(pk) != pack_to_orbit_index.end()) {
      continue;
    }
    const RawState rep = canonical_raw_state_rect(s, lx, ly);
    std::vector<RawState> members;
    enumerate_translation_orbit_rect(rep, lx, ly, &members);
    const std::size_t oid = map.orbits.size();
    for (RawState m : members) {
      pack_to_orbit_index[pack_raw_state(m)] = oid;
    }
    OrbitRecord rec;
    rec.representative = rep;
    rec.orbit_size = static_cast<std::uint16_t>(members.size());
    rec.period_tx = primitive_period_tx_rect(rep, lx, ly);
    rec.period_ty = primitive_period_ty_rect(rep, lx, ly);
    rec.stabilizer = translation_stabilizer_rect(rep, lx, ly);
    rec.compatible_momentum_mask = compatible_momentum_mask_for_stabilizer(rec.stabilizer, lx, ly);
    map.orbits.push_back(std::move(rec));
  }
  for (std::size_t oi = 0; oi < map.orbits.size(); ++oi) {
    const std::uint16_t mmask = map.orbits[oi].compatible_momentum_mask;
    for (int k = 0; k < lx * ly; ++k) {
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
