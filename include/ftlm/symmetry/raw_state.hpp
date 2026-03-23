#pragma once

#include <cstdint>

#include "ftlm/symmetry/occup_bits_translate.hpp"

namespace ftlm {
namespace symmetry {

/// Unprojected many-body configuration on the 4×4 torus (one sector-agnostic raw state).
/// Bit `s` of `up` / `dn` is set iff the corresponding spin occupies site `s` (`Lattice4x4` order).
struct RawState {
  std::uint16_t up = 0;
  std::uint16_t dn = 0;
};

inline bool operator==(RawState a, RawState b) {
  return a.up == b.up && a.dn == b.dn;
}

inline bool operator!=(RawState a, RawState b) { return !(a == b); }

/// Apply the same lattice translation to both spin species (global Z₄×Z₄ subgroup).
inline RawState translate_raw_state(RawState s, int dx, int dy) {
  return RawState{translate_occupation_bits(s.up, dx, dy), translate_occupation_bits(s.dn, dx, dy)};
}

inline RawState translate_raw_state_Tx(RawState s) {
  return RawState{translate_occupation_Tx(s.up), translate_occupation_Tx(s.dn)};
}

inline RawState translate_raw_state_Ty(RawState s) {
  return RawState{translate_occupation_Ty(s.up), translate_occupation_Ty(s.dn)};
}

}  // namespace symmetry
}  // namespace ftlm
