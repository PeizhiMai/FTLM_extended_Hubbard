// `ftlm_log_partition_hubbard_momentum_block_ref` matches explicit `ftlm_log_partition_complex` on the same block.
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

#include "ftlm/ftlm_thermo.hpp"
#include "ftlm/fock_basis.hpp"
#include "ftlm/hubbard_params.hpp"
#include "ftlm/symmetry/hubbard_lanczos.hpp"
#include "ftlm/symmetry/hubbard_momentum_action.hpp"
#include "ftlm/symmetry/momentum_sector.hpp"

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
    std::cerr << "test_ftlm_hubbard_momentum_block_ref: empty block\n";
    return 2;
  }

  ftlm::FtlmParams par;
  par.n_random = 8;
  par.lanczos_steps = 24;
  par.seed = 11u;
  const double beta = 2.0;

  const double z_ref =
      ftlm::symmetry::ftlm_log_partition_hubbard_momentum_block_ref(block, beta, par);
  const double z_lam = ftlm::ftlm_log_partition_complex(
      d,
      [&block](const std::complex<double>* x, std::complex<double>* y) { block.apply(x, y); },
      beta, par);

  const double diff = std::abs(z_ref - z_lam);
  if (diff > 1e-10) {
    std::cerr << "test_ftlm_hubbard_momentum_block_ref: mismatch ref=" << z_ref << " lambda=" << z_lam
              << " |diff|=" << diff << "\n";
    return 1;
  }

  std::cout << "test_ftlm_hubbard_momentum_block_ref ok dim=" << d << "\n";
  return 0;
}
