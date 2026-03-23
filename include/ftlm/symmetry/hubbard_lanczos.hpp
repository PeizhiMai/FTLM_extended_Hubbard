#pragma once

#include "ftlm/lanczos.hpp"
#include "ftlm/lanczos_engine.hpp"
#include "ftlm/symmetry/hubbard_momentum_action.hpp"
#include "ftlm/symmetry/k_basis.hpp"
#include "ftlm/symmetry/momentum_sector.hpp"

namespace ftlm {
namespace symmetry {

/// Binds `HubbardMomentumAction::apply` to a fixed \((k_x,k_y)\) block for use with `LanczosEngine` /
/// `lanczos_extrema`. Callers must keep `hub`, `map`, and `basis` alive for the binder’s lifetime.
struct HubbardKBlockApply {
  const HubbardMomentumAction* hub = nullptr;
  const MomentumSectorMap* map = nullptr;
  MomentumSector K{};
  const KBasis* basis = nullptr;

  void operator()(const std::complex<double>* x, std::complex<double>* y) const {
    hub->apply(*map, K, *basis, x, y);
  }
};

inline LanczosExtrema lanczos_extrema_hubbard_k_block(const HubbardMomentumAction& hub,
                                                      const MomentumSectorMap& map, MomentumSector K,
                                                      const KBasis& basis, int max_steps,
                                                      unsigned seed = 1) {
  const int dim = static_cast<int>(basis.dim());
  const HubbardKBlockApply binder{&hub, &map, K, &basis};
  return lanczos_extrema(dim, binder, max_steps, seed);
}

inline LanczosEngine make_lanczos_engine_hubbard_k_block(const HubbardMomentumAction& hub,
                                                           const MomentumSectorMap& map,
                                                           MomentumSector K, const KBasis& basis) {
  const int dim = static_cast<int>(basis.dim());
  return LanczosEngine(dim, HubbardKBlockApply{&hub, &map, K, &basis});
}

}  // namespace symmetry
}  // namespace ftlm
