#pragma once

#include <cstdint>
#include <vector>

#include "ftlm/symmetry/raw_state.hpp"

namespace ftlm {
namespace symmetry {

/// Site index after lattice translation by `(dx,dy)` on an `lx×ly` periodic torus (row-major sites).
inline int translated_site_rect(int site, int lx, int ly, int dx, int dy) {
  const int x = site % lx;
  const int y = site / lx;
  int nx = (x + dx) % lx;
  int ny = (y + dy) % ly;
  if (nx < 0) nx += lx;
  if (ny < 0) ny += ly;
  return ny * lx + nx;
}

/// Jordan–Wigner mode index: `m = 2*site + spin` with `spin=0` (↑) and `spin=1` (↓), matching
/// `apply_extended_hubbard` / `fermions_between_modes` ordering.
///
/// Returns the sign relating the **fermionic** translation \(T_R|s\rangle\) to the **bit-translated**
/// occupation ket \(|s'\rangle\) with the same up/dn masks as `translate_raw_state_rect(s, lx, ly, dx, dy)`:
/// \(T_R|s\rangle = \eta\,|s'\rangle\) with \(\eta \in \{+1,-1\}\).
///
/// Construction: list occupied modes in increasing site order, up then down at each site; map each
/// mode through the same site translation as `translate_occupation_bits_rect`; count inversions in
/// the resulting mode sequence; \(\eta = (-1)^{\text{inversions}}\).
inline double fermionic_translation_sign_rect(RawState s, int lx, int ly, int dx, int dy) {
  std::vector<int> transformed_modes;
  transformed_modes.reserve(static_cast<std::size_t>(2 * lx * ly));
  for (int site = 0; site < lx * ly; ++site) {
    if ((s.up >> site) & 1u) {
      const int tsite = translated_site_rect(site, lx, ly, dx, dy);
      transformed_modes.push_back(2 * tsite + 0);
    }
    if ((s.dn >> site) & 1u) {
      const int tsite = translated_site_rect(site, lx, ly, dx, dy);
      transformed_modes.push_back(2 * tsite + 1);
    }
  }
  int inv = 0;
  for (std::size_t i = 0; i < transformed_modes.size(); ++i) {
    for (std::size_t j = i + 1; j < transformed_modes.size(); ++j) {
      if (transformed_modes[i] > transformed_modes[j]) {
        ++inv;
      }
    }
  }
  return (inv & 1) ? -1.0 : 1.0;
}

}  // namespace symmetry
}  // namespace ftlm
