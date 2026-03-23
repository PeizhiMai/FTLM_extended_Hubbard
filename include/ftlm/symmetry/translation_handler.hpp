#pragma once

#include <vector>

#include "ftlm/symmetry/orbit.hpp"
#include "ftlm/symmetry/raw_state.hpp"

namespace ftlm {
namespace symmetry {

/// Facade for translation symmetry on `RawState` (joint action on up+dn).
/// Keeps call sites readable while the free functions in `orbit.hpp` stay easy to test.
struct TranslationHandler {
  static RawState canonical(RawState s) { return canonical_raw_state(s); }

  static void orbit(RawState seed, std::vector<RawState>* out) {
    enumerate_translation_orbit(seed, out);
  }

  static OrbitInfo info(RawState seed) { return make_orbit_info(seed); }
};

}  // namespace symmetry
}  // namespace ftlm
