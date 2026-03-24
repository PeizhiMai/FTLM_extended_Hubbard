#pragma once

#include <cstdint>

#include "ftlm/symmetry/lattice_4x4.hpp"

namespace ftlm {
namespace symmetry {

inline std::uint16_t translate_occupation_bits_rect(std::uint16_t bits, int lx, int ly, int dx, int dy) {
  std::uint16_t out = 0;
  for (int s = 0; s < lx * ly; ++s) {
    if ((bits >> s) & 1u) {
      const int x = s % lx;
      const int y = s / lx;
      int nx = (x + dx) % lx;
      int ny = (y + dy) % ly;
      if (nx < 0) nx += lx;
      if (ny < 0) ny += ly;
      const int t = ny * lx + nx;
      out |= static_cast<std::uint16_t>(1u << t);
    }
  }
  return out;
}

/// Translate a 16-bit occupation pattern as a **passive** relabeling of sites:
/// an electron at site `s` moves to `Lattice4x4::translate_site_index(s, dx, dy)`.
///
/// Equivalently: bit `t` of the output is set iff bit `s` of the input is set with
/// `t == translate_site_index(s, dx, dy)` (particle picture).
///
/// Only the low `Lattice4x4::N` bits are used; upper bits are cleared on output.
inline std::uint16_t translate_occupation_bits(std::uint16_t bits, int dx, int dy) {
  std::uint16_t out = 0;
  for (int s = 0; s < Lattice4x4::N; ++s) {
    if ((bits >> s) & 1u) {
      const int t = Lattice4x4::translate_site_index(s, dx, dy);
      out |= static_cast<std::uint16_t>(1u << t);
    }
  }
  return out;
}

inline std::uint16_t translate_occupation_Tx(std::uint16_t bits) {
  return translate_occupation_bits(bits, 1, 0);
}

inline std::uint16_t translate_occupation_Ty(std::uint16_t bits) {
  return translate_occupation_bits(bits, 0, 1);
}

}  // namespace symmetry
}  // namespace ftlm
