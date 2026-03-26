// Prototype: KBlockHamiltonianApply adapter matches current HubbardMomentumBlock (sanity for future matrix-free path).
#include "ftlm/fock_basis.hpp"
#include "ftlm/hubbard_params.hpp"
#include "ftlm/symmetry/hubbard_momentum_action.hpp"
#include "ftlm/symmetry/k_block_matrix_free.hpp"
#include "ftlm/symmetry/momentum_sector.hpp"

#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

namespace {

bool near_zero(std::complex<double> z) { return std::abs(z) < 1e-12; }

}  // namespace

int main() {
  ftlm::HubbardParams p{};
  p.Lx = 2;
  p.Ly = 2;
  p.t = 1.0;
  p.tp = 0.0;
  p.U = 4.0;
  p.V = 0.0;
  ftlm::symmetry::HubbardMomentumAction hub(p);

  const int n_up = 1;
  const int n_dn = 1;
  ftlm::FockBasis fb(4, n_up, n_dn);
  std::vector<ftlm::symmetry::RawState> universe;
  universe.reserve(static_cast<std::size_t>(fb.dim()));
  for (int i = 0; i < fb.dim(); ++i) {
    universe.push_back(ftlm::symmetry::RawState{static_cast<std::uint16_t>(fb.up_mask(i)),
                                                static_cast<std::uint16_t>(fb.down_mask(i))});
  }
  const auto map = ftlm::symmetry::build_momentum_sector_map_rect(std::move(universe), p.Lx, p.Ly);
  const ftlm::symmetry::MomentumSector K{0, 0, p.Lx, p.Ly};
  ftlm::symmetry::MomentumBlockScratch scratch;
  ftlm::symmetry::HubbardMomentumBlock block(hub, map, K, n_up, n_dn, &scratch);
  const int d = static_cast<int>(block.dim());
  if (d <= 0) {
    std::cerr << "test_k_block_matrix_free_interface: empty block\n";
    return 2;
  }

  ftlm::symmetry::k_block_matrix_free::HubbardBlockAdapter<ftlm::symmetry::HubbardMomentumBlock> adapter(&block);
  if (adapter.dim() != d) {
    std::cerr << "adapter dim mismatch\n";
    return 2;
  }

  std::vector<std::complex<double>> x(static_cast<std::size_t>(d), std::complex<double>(0.0, 0.0));
  std::vector<std::complex<double>> y0(static_cast<std::size_t>(d));
  std::vector<std::complex<double>> y1(static_cast<std::size_t>(d));
  x[0] = {1.0, 0.0};
  block.apply(x.data(), y0.data());
  adapter.apply(x.data(), y1.data());
  for (int i = 0; i < d; ++i) {
    if (!near_zero(y0[static_cast<std::size_t>(i)] - y1[static_cast<std::size_t>(i)])) {
      std::cerr << "apply mismatch at " << i << "\n";
      return 2;
    }
  }
  std::cout << "ok k_block_matrix_free adapter matches HubbardMomentumBlock dim=" << d << "\n";
  return 0;
}
