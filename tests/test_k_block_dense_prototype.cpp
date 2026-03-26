// Dense-column prototype vs HubbardMomentumBlock::apply (same Φ† H Φ block operator in orthonormal coords).
#include "ftlm/fock_basis.hpp"
#include "ftlm/hubbard_params.hpp"
#include "ftlm/symmetry/hubbard_momentum_action.hpp"
#include "ftlm/symmetry/k_block_dense_prototype.hpp"
#include "ftlm/symmetry/momentum_sector.hpp"

#include <cmath>
#include <complex>
#include <iostream>
#include <random>
#include <vector>

namespace {

bool same_vec(const std::vector<std::complex<double>>& a, const std::vector<std::complex<double>>& b, double tol) {
  if (a.size() != b.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (std::abs(a[i] - b[i]) > tol) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  ftlm::HubbardParams p{};
  p.Lx = 2;
  p.Ly = 2;
  p.t = 1.0;
  p.tp = -0.2;
  p.U = 4.0;
  p.V = 0.5;
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
    std::cerr << "test_k_block_dense_prototype: empty block\n";
    return 2;
  }

  ftlm::symmetry::k_block_matrix_free::DenseBlockHamiltonianFromApply dense(
      d, [&](const std::complex<double>* x, std::complex<double>* y) { block.apply(x, y); }, 64);

  const double tol = 1e-10;
  std::vector<std::complex<double>> x(static_cast<std::size_t>(d));
  std::vector<std::complex<double>> y0(static_cast<std::size_t>(d));
  std::vector<std::complex<double>> y1(static_cast<std::size_t>(d));

  // 1) First basis vector
  std::fill(x.begin(), x.end(), std::complex<double>(0.0, 0.0));
  x[0] = {1.0, 0.0};
  block.apply(x.data(), y0.data());
  dense.apply(x.data(), y1.data());
  if (!same_vec(y0, y1, tol)) {
    std::cerr << "mismatch: basis e0\n";
    return 2;
  }

  // 2) Random vector
  std::mt19937 rng(42);
  std::normal_distribution<double> g(0.0, 1.0);
  for (int i = 0; i < d; ++i) {
    x[static_cast<std::size_t>(i)] = {g(rng), g(rng)};
  }
  block.apply(x.data(), y0.data());
  dense.apply(x.data(), y1.data());
  if (!same_vec(y0, y1, tol)) {
    std::cerr << "mismatch: random\n";
    return 2;
  }

  // 3) All-ones (if d small)
  for (auto& v : x) {
    v = {1.0, 0.0};
  }
  block.apply(x.data(), y0.data());
  dense.apply(x.data(), y1.data());
  if (!same_vec(y0, y1, tol)) {
    std::cerr << "mismatch: ones\n";
    return 2;
  }

  std::cout << "ok dense_prototype matches HubbardMomentumBlock dim=" << d << "\n";
  return 0;
}
