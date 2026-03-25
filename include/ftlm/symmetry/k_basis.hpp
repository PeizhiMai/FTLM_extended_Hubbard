#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "ftlm/symmetry/momentum_sector.hpp"
#include "ftlm/symmetry/orbit.hpp"
#include "ftlm/symmetry/packed_raw_state.hpp"
#include "ftlm/symmetry/raw_state.hpp"

namespace ftlm {
namespace symmetry {

/// Ordered list of **canonical orbit representatives** that admit momentum \((k_x,k_y)\).
///
/// Indices are stable after `build_k_basis` (sorted by `pack_raw_state`). Use `index_of` to map a
/// compatible representative to its sector index; incompatible / non-representative states yield `npos`.
struct KBasis {
  static constexpr std::size_t npos = std::numeric_limits<std::size_t>::max();

  MomentumSector K{};
  /// Same length as `rep_packed`; `representatives[i]` matches `rep_packed[i]`.
  std::vector<std::uint32_t> rep_packed{};
  std::vector<RawState> representatives{};

  std::size_t dim() const noexcept { return representatives.size(); }

  /// Build sector basis from a precomputed `MomentumSectorMap` (one entry per translation orbit).
  static KBasis build(const MomentumSectorMap& map, MomentumSector Ksec) {
    KBasis b;
    b.K = Ksec;
    b.K.lx = map.lx;
    b.K.ly = map.ly;
    const std::size_t flat = static_cast<std::size_t>(Ksec.flat_index());
    if (flat >= map.orbit_indices_by_momentum.size()) {
      return b;
    }
    b.rep_packed.reserve(map.orbit_indices_by_momentum[flat].size());
    for (std::size_t oi : map.orbit_indices_by_momentum[flat]) {
      b.rep_packed.push_back(pack_raw_state(map.orbits[oi].representative));
    }
    std::sort(b.rep_packed.begin(), b.rep_packed.end());
    b.rep_packed.erase(std::unique(b.rep_packed.begin(), b.rep_packed.end()), b.rep_packed.end());
    b.representatives.reserve(b.rep_packed.size());
    for (std::uint32_t p : b.rep_packed) {
      b.representatives.push_back(unpack_raw_state(p));
    }
    return b;
  }

  /// Index of a **canonical** representative in this sector, or `npos` if absent.
  std::size_t index_of(RawState rep) const noexcept {
    const std::uint32_t p = pack_raw_state(rep);
    const auto it = std::lower_bound(rep_packed.begin(), rep_packed.end(), p);
    if (it == rep_packed.end() || *it != p) {
      return npos;
    }
    return static_cast<std::size_t>(it - rep_packed.begin());
  }
};

/// One raw Bloch column per orbit **member** (not only the canonical representative). For each translation orbit
/// compatible with `K`, appends every distinct state in that orbit (size `orbit_size`). After Gram whitening,
/// this spans the same `P_K` subspace as the dense translation projector.
inline std::vector<RawState> momentum_phi_seeds(const MomentumSectorMap& map, MomentumSector K) {
  std::vector<RawState> seeds;
  const std::size_t flat = static_cast<std::size_t>(K.flat_index());
  if (flat >= map.orbit_indices_by_momentum.size()) {
    return seeds;
  }
  for (std::size_t oi : map.orbit_indices_by_momentum[flat]) {
    std::vector<RawState> members;
    enumerate_translation_orbit_rect(map.orbits[oi].representative, map.lx, map.ly, &members);
    seeds.insert(seeds.end(), members.begin(), members.end());
  }
  std::sort(seeds.begin(), seeds.end(), [](RawState a, RawState b) {
    return pack_raw_state(a) < pack_raw_state(b);
  });
  seeds.erase(std::unique(seeds.begin(), seeds.end(),
                           [](RawState a, RawState b) { return pack_raw_state(a) == pack_raw_state(b); }),
              seeds.end());
  return seeds;
}

}  // namespace symmetry
}  // namespace ftlm
