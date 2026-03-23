#pragma once

#include <cstdint>
#include <functional>

#include "ftlm/symmetry/raw_state.hpp"

namespace ftlm {
namespace symmetry {

/// Total order / hashing for `RawState` without allocating. Only low 16 bits per spin are used.
inline constexpr std::uint32_t pack_raw_state(RawState s) noexcept {
  return (static_cast<std::uint32_t>(s.up) << 16) | static_cast<std::uint32_t>(s.dn);
}

inline constexpr RawState unpack_raw_state(std::uint32_t p) noexcept {
  return RawState{static_cast<std::uint16_t>(p >> 16),
                  static_cast<std::uint16_t>(p & 0xFFFFu)};
}

/// Flatten momentum quantum numbers (kx, ky) with kx, ky ∈ {0,1,2,3}.
inline constexpr int momentum_flat_index(int kx, int ky) noexcept { return kx + 4 * ky; }

inline constexpr void momentum_from_flat(int flat, int* kx, int* ky) noexcept {
  *kx = flat % 4;
  *ky = flat / 4;
}

struct PackedRawStateHash {
  std::size_t operator()(RawState s) const noexcept {
    return std::hash<std::uint32_t>{}(pack_raw_state(s));
  }
};

struct RawStateLess {
  bool operator()(RawState a, RawState b) const noexcept {
    return pack_raw_state(a) < pack_raw_state(b);
  }
};

}  // namespace symmetry
}  // namespace ftlm
