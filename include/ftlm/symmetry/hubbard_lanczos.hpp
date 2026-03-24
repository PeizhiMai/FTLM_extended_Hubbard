#pragma once

#include "ftlm/lanczos.hpp"
#include "ftlm/lanczos_engine.hpp"
#include "ftlm/symmetry/hubbard_momentum_action.hpp"
#include "ftlm/symmetry/k_basis.hpp"
#include "ftlm/symmetry/momentum_sector.hpp"
#include "ftlm/symmetry/raw_state.hpp"

namespace ftlm {
namespace symmetry {
namespace detail {

inline int popcount16_hub(std::uint16_t x) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_popcount(static_cast<unsigned>(x));
#else
  int c = 0;
  while (x) {
    ++c;
    x &= static_cast<std::uint16_t>(x - 1);
  }
  return c;
#endif
}

}  // namespace detail

/// Binds `HubbardMomentumAction::apply` to a fixed \((k_x,k_y)\) block for use with `LanczosEngine` /
/// `lanczos_extrema`. Callers must keep `hub`, `map`, and `basis` alive for the binder’s lifetime.
struct HubbardKBlockApply {
  const HubbardMomentumAction* hub = nullptr;
  const MomentumSectorMap* map = nullptr;
  MomentumSector K{};
  int n_up = 0;
  int n_dn = 0;

  void operator()(const std::complex<double>* x, std::complex<double>* y) const {
    hub->apply(*map, K, n_up, n_dn, x, y);
  }
};

inline LanczosExtrema lanczos_extrema_hubbard_k_block(const HubbardMomentumAction& hub,
                                                      const MomentumSectorMap& map, MomentumSector K,
                                                      const KBasis& basis, int max_steps,
                                                      unsigned seed = 1) {
  const RawState r0 = basis.representatives[0];
  const int n_up = detail::popcount16_hub(r0.up);
  const int n_dn = detail::popcount16_hub(r0.dn);
  const int dim = static_cast<int>(hub.momentum_block_dim(map, K, n_up, n_dn));
  const HubbardKBlockApply binder{&hub, &map, K, n_up, n_dn};
  return lanczos_extrema(dim, binder, max_steps, seed);
}

inline LanczosEngine make_lanczos_engine_hubbard_k_block(const HubbardMomentumAction& hub,
                                                           const MomentumSectorMap& map,
                                                           MomentumSector K, const KBasis& basis) {
  const RawState r0 = basis.representatives[0];
  const int n_up = detail::popcount16_hub(r0.up);
  const int n_dn = detail::popcount16_hub(r0.dn);
  const int dim = static_cast<int>(hub.momentum_block_dim(map, K, n_up, n_dn));
  return LanczosEngine(dim, HubbardKBlockApply{&hub, &map, K, n_up, n_dn});
}

}  // namespace symmetry
}  // namespace ftlm
