#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "ftlm/symmetry/packed_raw_state.hpp"
#include "ftlm/symmetry/raw_state.hpp"
#include "ftlm/symmetry/translation_group.hpp"

namespace ftlm {
namespace symmetry {

/// Summary of one translation orbit (Z₄×Z₄ acting jointly on up+dn). Built from the **canonical**
/// representative so `orbit_size` and periods are orbit-invariant.
struct OrbitInfo {
  RawState representative{};
  std::uint16_t orbit_size = 0;
  /// Smallest p ∈ {1,2,4} with T_x^p |ψ⟩ = |ψ⟩ (same bitmask after `translate_raw_state(..., p, 0)`).
  std::uint8_t period_tx = 1;
  /// Smallest q ∈ {1,2,4} with T_y^q |ψ⟩ = |ψ⟩.
  std::uint8_t period_ty = 1;
};

/// One translation orbit under Z₄×Z₄: canonical rep, stabilizer, size, and (for momentum sectors) mask.
struct OrbitRecord {
  RawState representative{};
  std::uint16_t orbit_size = 0;
  std::uint8_t period_tx = 1;
  std::uint8_t period_ty = 1;
  /// All group elements (dx,dy) that fix `representative`.
  std::vector<std::pair<std::int8_t, std::int8_t>> stabilizer;
  /// Bit `momentum_flat_index(kx,ky)` set iff character χ_k is trivial on `stabilizer`.
  std::uint16_t compatible_momentum_mask = 0;
};

/// Primitive period of **unit** x-translation on this configuration (divides 4 on a 4-site torus).
inline std::uint8_t primitive_period_tx(RawState s) {
  for (std::uint8_t p : {static_cast<std::uint8_t>(1), static_cast<std::uint8_t>(2),
                        static_cast<std::uint8_t>(4)}) {
    if (translate_raw_state(s, p, 0) == s) {
      return p;
    }
  }
  return 4;
}

inline std::uint8_t primitive_period_ty(RawState s) {
  for (std::uint8_t p : {static_cast<std::uint8_t>(1), static_cast<std::uint8_t>(2),
                        static_cast<std::uint8_t>(4)}) {
    if (translate_raw_state(s, 0, p) == s) {
      return p;
    }
  }
  return 4;
}

inline std::uint8_t primitive_period_tx_rect(RawState s, int lx, int ly) {
  for (int p = 1; p <= lx; ++p) {
    if (translate_raw_state_rect(s, lx, ly, p, 0) == s) {
      return static_cast<std::uint8_t>(p);
    }
  }
  return static_cast<std::uint8_t>(lx);
}

inline std::uint8_t primitive_period_ty_rect(RawState s, int lx, int ly) {
  for (int p = 1; p <= ly; ++p) {
    if (translate_raw_state_rect(s, lx, ly, 0, p) == s) {
      return static_cast<std::uint8_t>(p);
    }
  }
  return static_cast<std::uint8_t>(ly);
}

/// Distinct images of `seed` under all 16 translations, sorted ascending by `pack_raw_state`.
inline void enumerate_translation_orbit(RawState seed, std::vector<RawState>* out) {
  std::vector<std::uint32_t> packs;
  packs.reserve(translation_group_order());
  for_each_translation_z4z4([&](int dx, int dy) {
    packs.push_back(pack_raw_state(translate_raw_state(seed, dx, dy)));
  });
  std::sort(packs.begin(), packs.end());
  packs.erase(std::unique(packs.begin(), packs.end()), packs.end());
  out->clear();
  out->reserve(packs.size());
  for (std::uint32_t p : packs) {
    out->push_back(unpack_raw_state(p));
  }
}

inline void enumerate_translation_orbit_rect(RawState seed, int lx, int ly, std::vector<RawState>* out) {
  std::vector<std::uint32_t> packs;
  packs.reserve(static_cast<std::size_t>(translation_group_order(lx, ly)));
  for_each_translation_rect(lx, ly, [&](int dx, int dy) {
    packs.push_back(pack_raw_state(translate_raw_state_rect(seed, lx, ly, dx, dy)));
  });
  std::sort(packs.begin(), packs.end());
  packs.erase(std::unique(packs.begin(), packs.end()), packs.end());
  out->clear();
  out->reserve(packs.size());
  for (std::uint32_t p : packs) {
    out->push_back(unpack_raw_state(p));
  }
}

/// Lexicographic minimum on packed `(up, dn)` within the orbit.
inline RawState canonical_raw_state(RawState seed) {
  std::vector<RawState> orb;
  enumerate_translation_orbit(seed, &orb);
  return *std::min_element(orb.begin(), orb.end(), RawStateLess{});
}

inline RawState canonical_raw_state_rect(RawState seed, int lx, int ly) {
  std::vector<RawState> orb;
  enumerate_translation_orbit_rect(seed, lx, ly, &orb);
  return *std::min_element(orb.begin(), orb.end(), RawStateLess{});
}

/// Canonical rep, distinct orbit size, and primitive Tx/Ty periods (evaluated on the canonical rep).
inline OrbitInfo make_orbit_info(RawState seed) {
  const RawState rep = canonical_raw_state(seed);
  std::vector<RawState> members;
  enumerate_translation_orbit(rep, &members);
  OrbitInfo info;
  info.representative = rep;
  info.orbit_size = static_cast<std::uint16_t>(members.size());
  info.period_tx = primitive_period_tx(rep);
  info.period_ty = primitive_period_ty(rep);
  return info;
}

/// Stabilizer subgroup of translations fixing `s` (always includes (0,0)).
inline std::vector<std::pair<std::int8_t, std::int8_t>> translation_stabilizer(RawState s) {
  std::vector<std::pair<std::int8_t, std::int8_t>> stab;
  stab.reserve(16);
  for_each_translation_z4z4([&](int dx, int dy) {
    if (translate_raw_state(s, dx, dy) == s) {
      stab.push_back({static_cast<std::int8_t>(dx), static_cast<std::int8_t>(dy)});
    }
  });
  return stab;
}

inline std::vector<std::pair<std::int8_t, std::int8_t>> translation_stabilizer_rect(RawState s, int lx, int ly) {
  std::vector<std::pair<std::int8_t, std::int8_t>> stab;
  stab.reserve(static_cast<std::size_t>(translation_group_order(lx, ly)));
  for_each_translation_rect(lx, ly, [&](int dx, int dy) {
    if (translate_raw_state_rect(s, lx, ly, dx, dy) == s) {
      stab.push_back({static_cast<std::int8_t>(dx), static_cast<std::int8_t>(dy)});
    }
  });
  return stab;
}

/// Sanity: |orbit| · |stabilizer| = |Z₄×Z₄| for a translation group action on a single state.
inline bool check_orbit_stabilizer_count_identity(const OrbitRecord& r) {
  const std::uint32_t prod = static_cast<std::uint32_t>(r.orbit_size) *
                             static_cast<std::uint32_t>(r.stabilizer.size());
  return prod == static_cast<std::uint32_t>(translation_group_order());
}

}  // namespace symmetry
}  // namespace ftlm
