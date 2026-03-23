#pragma once

namespace ftlm {
namespace symmetry {

/// The spatial translation group for the 4×4 torus: Z₄ × Z₄ (16 elements).
inline constexpr int translation_group_order() noexcept { return 16; }

/// Visit every translation (dx, dy) with dx, dy ∈ {0,1,2,3} in row-major order (dx inner).
template <class F>
void for_each_translation_z4z4(F&& f) {
  for (int dy = 0; dy < 4; ++dy) {
    for (int dx = 0; dx < 4; ++dx) {
      f(dx, dy);
    }
  }
}

}  // namespace symmetry
}  // namespace ftlm
