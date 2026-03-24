#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

#include "ftlm/fock_basis.hpp"
#include "ftlm/hubbard_hamiltonian.hpp"
#include "ftlm/hubbard_params.hpp"
#include "ftlm/lattice.hpp"
#include "ftlm/symmetry/hubbard_momentum_action.hpp"
#include "ftlm/symmetry/k_basis.hpp"
#include "ftlm/symmetry/momentum_sector.hpp"
#include "ftlm/symmetry/raw_state.hpp"

namespace {

double full_sector_trace(const ftlm::HubbardParams& p, const ftlm::FockBasis& fb,
                         const std::vector<ftlm::SpinfulHopping>& hops,
                         const std::vector<ftlm::NearestPair>& pairs) {
  const int dim = fb.dim();
  std::vector<std::complex<double>> x(static_cast<size_t>(dim), {0.0, 0.0});
  std::vector<std::complex<double>> y(static_cast<size_t>(dim), {0.0, 0.0});
  double tr = 0.0;
  for (int j = 0; j < dim; ++j) {
    std::fill(x.begin(), x.end(), std::complex<double>(0.0, 0.0));
    x[static_cast<size_t>(j)] = {1.0, 0.0};
    ftlm::apply_extended_hubbard(p, fb, hops, pairs, x.data(), y.data());
    tr += y[static_cast<size_t>(j)].real();
  }
  return tr;
}

double momentum_block_trace_sum(const ftlm::symmetry::HubbardMomentumAction& hub,
                                const ftlm::symmetry::MomentumSectorMap& map, int lx, int ly, int nu, int nd) {
  double tr = 0.0;
  for (int ky = 0; ky < ly; ++ky) {
    for (int kx = 0; kx < lx; ++kx) {
      const ftlm::symmetry::MomentumSector K{kx, ky, lx, ly};
      const auto kb = ftlm::symmetry::KBasis::build(map, K);
      const int dk = static_cast<int>(hub.momentum_block_dim(map, K, nu, nd));
      if (dk <= 0) {
        continue;
      }
      (void)kb;
      std::vector<std::complex<double>> x(static_cast<size_t>(dk), {0.0, 0.0});
      std::vector<std::complex<double>> y(static_cast<size_t>(dk), {0.0, 0.0});
      for (int j = 0; j < dk; ++j) {
        std::fill(x.begin(), x.end(), std::complex<double>(0.0, 0.0));
        x[static_cast<size_t>(j)] = {1.0, 0.0};
        hub.apply(map, K, nu, nd, x.data(), y.data());
        tr += y[static_cast<size_t>(j)].real();
      }
    }
  }
  return tr;
}

}  // namespace

int main() {
  ftlm::HubbardParams p;
  p.Lx = 2;
  p.Ly = 2;
  p.t = 1.0;
  p.tp = -0.35;
  p.U = 5.75;
  p.V = 0.9;
  p.phi_x = 0.0;
  p.phi_y = 0.0;

  ftlm::RectLattice lat{p.Lx, p.Ly};
  std::vector<ftlm::SpinfulHopping> hops;
  std::vector<ftlm::NearestPair> pairs;
  ftlm::build_hubbard_geometry(p, lat, &hops, &pairs);
  ftlm::symmetry::HubbardMomentumAction hub(p);

  for (int nu = 0; nu <= 4; ++nu) {
    for (int nd = 0; nd <= 4; ++nd) {
      ftlm::FockBasis fb(4, nu, nd);
      if (fb.dim() <= 0) {
        continue;
      }
      std::vector<ftlm::symmetry::RawState> universe;
      universe.reserve(static_cast<size_t>(fb.dim()));
      for (int i = 0; i < fb.dim(); ++i) {
        universe.push_back(ftlm::symmetry::RawState{static_cast<std::uint16_t>(fb.up_mask(i)),
                                                    static_cast<std::uint16_t>(fb.down_mask(i))});
      }
      const auto map = ftlm::symmetry::build_momentum_sector_map_rect(std::move(universe), 2, 2);

      const double tr_full = full_sector_trace(p, fb, hops, pairs);
      const double tr_k = momentum_block_trace_sum(hub, map, 2, 2, nu, nd);
      const double err = std::abs(tr_full - tr_k);
      if (err > 1e-8) {
        std::cerr << "trace mismatch sector (" << nu << "," << nd << "): full=" << tr_full
                  << " ksum=" << tr_k << " err=" << err << "\n";
        return 1;
      }
    }
  }

  std::cout << "test_momentum_trace_2x2 ok\n";
  return 0;
}
