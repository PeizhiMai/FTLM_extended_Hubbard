#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

#include "ftlm/fock_basis.hpp"
#include "ftlm/symmetry/k_basis.hpp"
#include "ftlm/symmetry/momentum_sector.hpp"
#include "ftlm/symmetry/raw_state.hpp"

namespace {

int popcount_bits(std::uint16_t x, int bits) {
  int c = 0;
  for (int i = 0; i < bits; ++i) {
    if ((x >> i) & 1u) {
      ++c;
    }
  }
  return c;
}

}  // namespace

int main() {
  constexpr int Lx = 2;
  constexpr int Ly = 2;
  constexpr int n_sites = Lx * Ly;
  constexpr int nk = Lx * Ly;

  for (int nu = 0; nu <= n_sites; ++nu) {
    for (int nd = 0; nd <= n_sites; ++nd) {
      ftlm::FockBasis fb(n_sites, nu, nd);
      const int dim_full = fb.dim();
      if (dim_full <= 0) {
        continue;
      }
      std::vector<ftlm::symmetry::RawState> universe;
      universe.reserve(static_cast<size_t>(dim_full));
      for (int i = 0; i < dim_full; ++i) {
        universe.push_back(
            ftlm::symmetry::RawState{static_cast<std::uint16_t>(fb.up_mask(i)),
                                     static_cast<std::uint16_t>(fb.down_mask(i))});
      }

      const auto map = ftlm::symmetry::build_momentum_sector_map_rect(std::move(universe), Lx, Ly);
      int sum_dims = 0;
      for (int ky = 0; ky < Ly; ++ky) {
        for (int kx = 0; kx < Lx; ++kx) {
          const ftlm::symmetry::MomentumSector K{kx, ky, Lx, Ly};
          const auto kb = ftlm::symmetry::KBasis::build(map, K);
          sum_dims += static_cast<int>(kb.dim());
        }
      }

      int orbit_compat_sum = 0;
      for (const auto& orb : map.orbits) {
        orbit_compat_sum += popcount_bits(orb.compatible_momentum_mask, nk);
      }

      if (sum_dims != dim_full || orbit_compat_sum != dim_full) {
        std::cerr << "Momentum partition mismatch at sector (" << nu << "," << nd
                  << "): dim_full=" << dim_full << " sum_k_dim=" << sum_dims
                  << " sum_orbit_compat=" << orbit_compat_sum << "\n";
        return 1;
      }
    }
  }

  std::cout << "test_momentum_partition_2x2 ok\n";
  return 0;
}
