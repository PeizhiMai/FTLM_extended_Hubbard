// Matrix-free symmetrized P_K vs dense P_K; matrix-free P_K H P_K vs dense M and Φ-basis production block.
#include "ftlm/fock_basis.hpp"
#include "ftlm/hubbard_params.hpp"
#include "ftlm/symmetry/hubbard_momentum_action.hpp"
#include "ftlm/symmetry/k_block_dense_prototype.hpp"
#include "ftlm/symmetry/k_block_pk_tiny_dense.hpp"
#include "ftlm/symmetry/momentum_sector.hpp"
#include "ftlm/symmetry/orbit_bloch_phi.hpp"

#include <cmath>
#include <complex>
#include <iostream>
#include <random>
#include <vector>

namespace {

double max_abs_vec(int n, const std::complex<double>* a, const std::complex<double>* b) {
  double m = 0.0;
  for (int i = 0; i < n; ++i) {
    m = std::max(m, std::abs(a[static_cast<std::size_t>(i)] - b[static_cast<std::size_t>(i)]));
  }
  return m;
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
  const int d = fb.dim();
  const int dk = static_cast<int>(block.dim());
  if (dk <= 0) {
    std::cerr << "empty block\n";
    return 2;
  }

  const double tol = 1e-11;
  std::vector<std::complex<double>> work;
  std::vector<std::complex<double>> y_dense(static_cast<std::size_t>(d));
  std::vector<std::complex<double>> y_mf(static_cast<std::size_t>(d));
  std::vector<std::complex<double>> p_flat;

  auto run_P_case = [&](const std::complex<double>* xv) {
    ftlm::symmetry::tiny_pk::apply_symmetrized_P_K_vector_dense(fb, p.Lx, p.Ly, K, xv, y_dense.data(), &p_flat);
    ftlm::symmetry::tiny_pk::apply_symmetrized_P_K_vector_matrix_free(fb, p.Lx, p.Ly, K, xv, y_mf.data(), &work);
    return max_abs_vec(d, y_dense.data(), y_mf.data());
  };

  double max_P = 0.0;
  std::vector<std::complex<double>> x(static_cast<std::size_t>(d), std::complex<double>{0.0, 0.0});
  x[0] = {1.0, 0.0};
  max_P = std::max(max_P, run_P_case(x.data()));
  for (int i = 0; i < d; ++i) {
    x[static_cast<std::size_t>(i)] = {1.0, 0.0};
  }
  max_P = std::max(max_P, run_P_case(x.data()));
  std::mt19937 rng(11);
  std::normal_distribution<double> g(0.0, 1.0);
  for (int i = 0; i < d; ++i) {
    x[static_cast<std::size_t>(i)] = {g(rng), g(rng)};
  }
  max_P = std::max(max_P, run_P_case(x.data()));

  std::cout << "max|dense P_K x - matrix-free P_K x| = " << max_P << "\n";
  if (max_P > tol) {
    std::cerr << "P_K mismatch\n";
    return 2;
  }

  ftlm::symmetry::tiny_pk::TinyPkHpkDense dense_pk = ftlm::symmetry::tiny_pk::build_tiny_pk_h_pk_dense(
      p, fb, p.Lx, p.Ly, K, hub.hoppings, hub.nn_pairs, 64);

  std::vector<std::complex<double>> w1;
  std::vector<std::complex<double>> w2;
  std::vector<std::complex<double>> y_ph(static_cast<std::size_t>(d));

  auto run_PHP_case = [&](const std::complex<double>* xv) {
    ftlm::symmetry::tiny_pk::matvec_colmajor_vector(d, dense_pk.m_colmajor.data(), xv, y_dense.data());
    ftlm::symmetry::tiny_pk::apply_symmetrized_PK_H_PK_vector_matrix_free(p, fb, p.Lx, p.Ly, K, hub.hoppings,
                                                                          hub.nn_pairs, xv, y_mf.data(), &w1, &w2);
    return max_abs_vec(d, y_dense.data(), y_mf.data());
  };

  double max_PHP = 0.0;
  std::fill(x.begin(), x.end(), std::complex<double>{0.0, 0.0});
  x[0] = {1.0, 0.0};
  max_PHP = std::max(max_PHP, run_PHP_case(x.data()));
  for (int i = 0; i < d; ++i) {
    x[static_cast<std::size_t>(i)] = {1.0, 0.0};
  }
  max_PHP = std::max(max_PHP, run_PHP_case(x.data()));
  for (int i = 0; i < d; ++i) {
    x[static_cast<std::size_t>(i)] = {g(rng), g(rng)};
  }
  max_PHP = std::max(max_PHP, run_PHP_case(x.data()));

  std::cout << "max|dense P_K H P_K x - matrix-free P_K H P_K x| = " << max_PHP << "\n";
  if (max_PHP > tol) {
    std::cerr << "P_K H P_K mismatch\n";
    return 2;
  }

  std::vector<std::complex<double>> phi_cm;
  std::size_t dk_phi = 0;
  ftlm::symmetry::build_momentum_phi_orbit_orthonormal(map, K, p.Lx, p.Ly, fb, &phi_cm, &dk_phi);
  if (static_cast<int>(dk_phi) != dk) {
    std::cerr << "dk phi mismatch\n";
    return 2;
  }

  ftlm::symmetry::k_block_matrix_free::DenseBlockHamiltonianFromApply h_prod(
      dk, [&](const std::complex<double>* xb, std::complex<double>* yb) { block.apply(xb, yb); }, 64);

  std::vector<std::complex<double>> xb(static_cast<std::size_t>(dk));
  std::vector<std::complex<double>> yb0(static_cast<std::size_t>(dk));
  std::vector<std::complex<double>> yb1(static_cast<std::size_t>(dk));
  std::vector<std::complex<double>> a_phi(static_cast<std::size_t>(dk * dk));

  auto max_phi_diff = [&](const std::complex<double>* xf) {
    ftlm::symmetry::tiny_pk::phi_dagger_M_phi(d, dk, phi_cm.data(), dense_pk.m_colmajor.data(), a_phi.data());
    ftlm::symmetry::tiny_pk::matvec_colmajor_vector(dk, a_phi.data(), xf, yb0.data());
    h_prod.apply(xf, yb1.data());
    return max_abs_vec(dk, yb0.data(), yb1.data());
  };

  double max_block = 0.0;
  std::fill(xb.begin(), xb.end(), std::complex<double>{0.0, 0.0});
  xb[0] = {1.0, 0.0};
  max_block = std::max(max_block, max_phi_diff(xb.data()));
  for (int i = 0; i < dk; ++i) {
    xb[static_cast<std::size_t>(i)] = {1.0, 0.0};
  }
  max_block = std::max(max_block, max_phi_diff(xb.data()));
  for (int i = 0; i < dk; ++i) {
    xb[static_cast<std::size_t>(i)] = {g(rng), g(rng)};
  }
  max_block = std::max(max_block, max_phi_diff(xb.data()));

  std::cout << "max|Phi^dagger M x - HubbardMomentumBlock::apply x| = " << max_block << "\n";
  if (max_block > tol) {
    std::cerr << "Phi-basis production mismatch\n";
    return 2;
  }

  std::cout << "ok matrix-free P_K and P_K H P_K match dense; Phi-basis matches production.\n";
  return 0;
}
