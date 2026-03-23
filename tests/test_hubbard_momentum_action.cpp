#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

#include "ftlm/hubbard_params.hpp"
#include "ftlm/symmetry/hubbard_momentum_action.hpp"
#include "ftlm/symmetry/k_basis.hpp"
#include "ftlm/symmetry/momentum_sector.hpp"
#include "ftlm/symmetry/raw_state.hpp"

namespace {

using ftlm::HubbardParams;
using ftlm::symmetry::HubbardMomentumAction;
using ftlm::symmetry::KBasis;
using ftlm::symmetry::MomentumSector;
using ftlm::symmetry::MomentumSectorMap;
using ftlm::symmetry::RawState;
using ftlm::symmetry::build_momentum_sector_map;

double tight_binding_dispersion(int kx, int ky, double t) {
  const double kx_pi = 0.5 * std::acos(-1.0) * static_cast<double>(kx);
  const double ky_pi = 0.5 * std::acos(-1.0) * static_cast<double>(ky);
  return -2.0 * t * (std::cos(kx_pi) + std::cos(ky_pi));
}

void test_single_particle_dispersion_4x4() {
  HubbardParams p;
  p.Lx = 4;
  p.Ly = 4;
  p.t = 1.0;
  p.tp = 0.0;
  p.U = 0.0;
  p.V = 0.0;
  HubbardMomentumAction hop(p);

  std::vector<RawState> seeds;
  for (int s = 0; s < 16; ++s) {
    seeds.push_back(RawState{static_cast<std::uint16_t>(1u << s), 0});
  }
  const MomentumSectorMap map = build_momentum_sector_map(std::move(seeds));
  assert(map.orbits.size() == 1u);

  for (int ky = 0; ky < 4; ++ky) {
    for (int kx = 0; kx < 4; ++kx) {
      const MomentumSector K{kx, ky};
      const KBasis basis = KBasis::build(map, K);
      assert(basis.dim() == 1u);
      std::complex<double> x[1] = {std::complex<double>(1.0, 0.0)};
      std::complex<double> y[1] = {0.0};
      hop.apply(map, K, basis, x, y);
      const double eps = tight_binding_dispersion(kx, ky, p.t);
      assert(std::abs(y[0].real() - eps) < 1e-10);
      assert(std::abs(y[0].imag()) < 1e-10);
    }
  }
}

void test_onsite_u_diagonal_in_gamma() {
  HubbardParams p;
  p.Lx = 4;
  p.Ly = 4;
  p.t = 0.0;
  p.U = 2.5;
  p.V = 0.0;
  HubbardMomentumAction hub(p);

  std::vector<RawState> seeds;
  seeds.push_back({static_cast<std::uint16_t>(3u << 0), 0});  // sites 0,1 up — not double occ
  seeds.push_back({1u << 0, 1u << 0});                        // double on site 0
  const MomentumSectorMap map = build_momentum_sector_map(std::move(seeds));
  const MomentumSector gamma{0, 0};
  const KBasis basis = KBasis::build(map, gamma);
  assert(basis.dim() == 2u);

  for (std::size_t j = 0; j < basis.dim(); ++j) {
    std::vector<std::complex<double>> x(basis.dim(), 0.0);
    std::vector<std::complex<double>> y(basis.dim(), 0.0);
    x[j] = 1.0;
    hub.apply(map, gamma, basis, x.data(), y.data());
    const std::uint16_t ud =
        static_cast<std::uint16_t>(basis.representatives[j].up & basis.representatives[j].dn);
    int docc = 0;
    for (int b = 0; b < 16; ++b) {
      if ((ud >> b) & 1) {
        ++docc;
      }
    }
    const double expect = p.U * static_cast<double>(docc);
    for (std::size_t i = 0; i < basis.dim(); ++i) {
      if (i == j) {
        assert(std::abs(y[i].real() - expect) < 1e-12);
      } else {
        assert(std::abs(y[i]) < 1e-12);
      }
    }
  }
}

}  // namespace

int main() {
  test_single_particle_dispersion_4x4();
  test_onsite_u_diagonal_in_gamma();
  std::cout << "test_hubbard_momentum_action ok\n";
  return 0;
}
