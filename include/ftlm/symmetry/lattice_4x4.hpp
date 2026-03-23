#pragma once

#include <cstdint>

namespace ftlm {
namespace symmetry {

/// Fixed 4×4 periodic torus for the Hubbard many-body Hilbert space layout.
///
/// Site indexing (row-major, x runs fastest):
///   index = y * Lx + x,   with x ∈ {0,…,Lx−1}, y ∈ {0,…,Ly−1}.
/// Drawing with y downward (common for grids):
///   (0,0) (1,0) (2,0) (3,0)   ← y = 0  →  indices 0 1 2 3
///   (0,1) …                     …
/// So index 1 is “to the right” of index 0; index 4 is “below” index 0.
///
/// Translations are periodic: coordinates use modular arithmetic in each direction.
struct Lattice4x4 {
  static constexpr int Lx = 4;
  static constexpr int Ly = 4;
  static constexpr int N = Lx * Ly;  // 16

  static constexpr int site_from_xy(int x, int y) { return y * Lx + x; }

  static constexpr void xy_from_site(int site, int* x, int* y) {
    *x = site % Lx;
    *y = site / Lx;
  }

  /// `a mod m` in 0 … m−1 for m > 0 (works for negative `a`).
  static constexpr int imod(int a, int m) {
    int r = a % m;
    return r >= 0 ? r : r + m;
  }

  /// Periodic translate of a site index by (dx, dy) in lattice units.
  static constexpr int translate_site_index(int site, int dx, int dy) {
    int x = 0;
    int y = 0;
    xy_from_site(site, &x, &y);
    const int nx = imod(x + dx, Lx);
    const int ny = imod(y + dy, Ly);
    return site_from_xy(nx, ny);
  }

  /// Unit translation +x (same row, next column, periodic).
  static constexpr int translate_site_Tx(int site) { return translate_site_index(site, 1, 0); }

  /// Unit translation +y (next row, same column, periodic).
  static constexpr int translate_site_Ty(int site) { return translate_site_index(site, 0, 1); }
};

}  // namespace symmetry
}  // namespace ftlm
