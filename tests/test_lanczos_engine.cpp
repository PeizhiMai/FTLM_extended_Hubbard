#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

#include "ftlm/hubbard_params.hpp"
#include "ftlm/lanczos_engine.hpp"
#include "ftlm/symmetry/hubbard_lanczos.hpp"
#include "ftlm/symmetry/hubbard_momentum_action.hpp"
#include "ftlm/symmetry/k_basis.hpp"
#include "ftlm/symmetry/momentum_sector.hpp"
#include "ftlm/symmetry/raw_state.hpp"

namespace {

using ftlm::HubbardParams;
using ftlm::LanczosEngine;
using ftlm::symmetry::HubbardMomentumAction;
using ftlm::symmetry::KBasis;
using ftlm::symmetry::MomentumSector;
using ftlm::symmetry::MomentumSectorMap;
using ftlm::symmetry::RawState;
using ftlm::symmetry::build_momentum_sector_map;
using ftlm::symmetry::lanczos_extrema_hubbard_k_block;
using ftlm::symmetry::make_lanczos_engine_hubbard_k_block;

double tb_disp(int kx, int ky, double t) {
  const double pi = std::acos(-1.0);
  return -2.0 * t * (std::cos(0.5 * pi * static_cast<double>(kx)) +
                     std::cos(0.5 * pi * static_cast<double>(ky)));
}

void test_lanczos_single_particle_matches_dispersion() {
  HubbardParams p;
  p.Lx = p.Ly = 4;
  p.t = 1.0;
  p.U = 0.0;
  HubbardMomentumAction hub(p);
  std::vector<RawState> seeds;
  for (int s = 0; s < 16; ++s) {
    seeds.push_back(RawState{static_cast<std::uint16_t>(1u << s), 0});
  }
  const MomentumSectorMap map = build_momentum_sector_map(std::move(seeds));

  for (int ky = 0; ky < 4; ++ky) {
    for (int kx = 0; kx < 4; ++kx) {
      const MomentumSector K{kx, ky};
      const KBasis basis = KBasis::build(map, K);
      const double eps = tb_disp(kx, ky, p.t);
      const auto rz = lanczos_extrema_hubbard_k_block(hub, map, K, basis, 24, 3u + static_cast<unsigned>(kx + 4 * ky));
      assert(std::abs(rz.min_eval - eps) < 1e-8);
      assert(std::abs(rz.max_eval - eps) < 1e-8);
    }
  }
}

void test_lanczos_engine_wrapper_two_level_gamma() {
  HubbardParams p;
  p.Lx = p.Ly = 4;
  p.t = 0.0;
  p.U = 1.0;
  HubbardMomentumAction hub(p);
  std::vector<RawState> seeds;
  seeds.push_back({static_cast<std::uint16_t>(3u << 0), 0});
  seeds.push_back({1u << 0, 1u << 0});
  const MomentumSectorMap map = build_momentum_sector_map(std::move(seeds));
  const KBasis basis = KBasis::build(map, MomentumSector{0, 0});
  assert(basis.dim() == 2u);

  LanczosEngine eng = make_lanczos_engine_hubbard_k_block(hub, map, MomentumSector{0, 0}, basis);
  assert(static_cast<int>(eng.v_cur.size()) == 2);
  const auto rz = eng.run_extrema(32, 99u);
  assert(std::abs(rz.min_eval - 0.0) < 1e-8);
  assert(std::abs(rz.max_eval - 1.0) < 1e-8);
}

}  // namespace

int main() {
  test_lanczos_single_particle_matches_dispersion();
  test_lanczos_engine_wrapper_two_level_gamma();
  std::cout << "test_lanczos_engine ok\n";
  return 0;
}
