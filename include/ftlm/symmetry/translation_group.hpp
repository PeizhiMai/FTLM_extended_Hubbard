#pragma once

namespace ftlm {
namespace symmetry {

/// Translation group order for a rectangular torus: Z_Lx × Z_Ly.
inline constexpr int translation_group_order(int lx, int ly) noexcept { return lx * ly; }

/// Backward-compatible 4x4 order.
inline constexpr int translation_group_order() noexcept { return translation_group_order(4, 4); }

/// Visit every translation (dx, dy) with dx, dy ∈ {0,1,2,3} in row-major order (dx inner).
template <class F>
void for_each_translation_z4z4(F&& f) {
  for (int dy = 0; dy < 4; ++dy) {
    for (int dx = 0; dx < 4; ++dx) {
      f(dx, dy);
    }
  }
}

template <class F>
void for_each_translation_rect(int lx, int ly, F&& f) {
  for (int dy = 0; dy < ly; ++dy) {
    for (int dx = 0; dx < lx; ++dx) {
      f(dx, dy);
    }
  }
}

}  // namespace symmetry
}  // namespace ftlm
