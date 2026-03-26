// Independent dense P_K H P_K vs production HubbardMomentumBlock (Phi^dagger H Phi) on a tiny 2x2 sector.
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

double max_abs_diff_mat(int dk, const std::complex<double>* A, const std::complex<double>* B) {
  double m = 0.0;
  for (int j = 0; j < dk; ++j) {
    for (int i = 0; i < dk; ++i) {
      m = std::max(m, std::abs(A[static_cast<std::size_t>(i + j * dk)] - B[static_cast<std::size_t>(i + j * dk)]));
    }
  }
  return m;
}

double max_abs_diff_vec(int n, const std::complex<double>* a, const std::complex<double>* b) {
  double m = 0.0;
  for (int i = 0; i < n; ++i) {
    m = std::max(m, std::abs(a[static_cast<std::size_t>(i)] - b[static_cast<std::size_t>(i)]));
  }
  return m;
}

/// Phi^dagger P H Phi (columns of Phi in Im(P_K); P Phi = Phi so this equals Phi^dagger P H P_K Phi = Phi^dagger M Phi).
void phi_dagger_P_H_phi(int d_full, int dk, const std::complex<double>* phi_colmajor, const std::complex<double>* p_colmajor,
                        const std::complex<double>* h_colmajor, std::complex<double>* out_colmajor) {
  std::vector<std::complex<double>> hphi(static_cast<std::size_t>(d_full * dk), std::complex<double>{0.0, 0.0});
  for (int j = 0; j < dk; ++j) {
    for (int i = 0; i < d_full; ++i) {
      std::complex<double> s{0.0, 0.0};
      for (int k = 0; k < d_full; ++k) {
        s += h_colmajor[static_cast<std::size_t>(i + k * d_full)] *
             phi_colmajor[static_cast<std::size_t>(k + j * d_full)];
      }
      hphi[static_cast<std::size_t>(i + j * d_full)] = s;
    }
  }
  std::vector<std::complex<double>> phphi(static_cast<std::size_t>(d_full * dk), std::complex<double>{0.0, 0.0});
  for (int j = 0; j < dk; ++j) {
    for (int i = 0; i < d_full; ++i) {
      std::complex<double> s{0.0, 0.0};
      for (int k = 0; k < d_full; ++k) {
        s += p_colmajor[static_cast<std::size_t>(i + k * d_full)] *
             hphi[static_cast<std::size_t>(k + j * d_full)];
      }
      phphi[static_cast<std::size_t>(i + j * d_full)] = s;
    }
  }
  for (int j = 0; j < dk; ++j) {
    for (int i = 0; i < dk; ++i) {
      std::complex<double> s{0.0, 0.0};
      for (int p = 0; p < d_full; ++p) {
        s += std::conj(phi_colmajor[static_cast<std::size_t>(p + i * d_full)]) *
             phphi[static_cast<std::size_t>(p + j * d_full)];
      }
      out_colmajor[static_cast<std::size_t>(i + j * dk)] = s;
    }
  }
}

void matvec_dk(int dk, const std::complex<double>* A_colmajor, const std::complex<double>* x, std::complex<double>* y) {
  for (int i = 0; i < dk; ++i) {
    std::complex<double> s{0.0, 0.0};
    for (int j = 0; j < dk; ++j) {
      s += A_colmajor[static_cast<std::size_t>(i + j * dk)] * x[static_cast<std::size_t>(j)];
    }
    y[static_cast<std::size_t>(i)] = s;
  }
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
  const int dk = static_cast<int>(block.dim());
  const int d_full = static_cast<int>(fb.dim());
  if (dk <= 0) {
    std::cerr << "test_k_block_pk_tiny_vs_production: empty block\n";
    return 2;
  }

  ftlm::symmetry::tiny_pk::TinyPkHpkDense pk =
      ftlm::symmetry::tiny_pk::build_tiny_pk_h_pk_dense(p, fb, p.Lx, p.Ly, K, hub.hoppings, hub.nn_pairs, 64);

  if (pk.d_full != d_full) {
    std::cerr << "d_full mismatch\n";
    return 2;
  }

  std::vector<std::complex<double>> phi_cm;
  std::size_t dk_phi = 0;
  ftlm::symmetry::build_momentum_phi_orbit_orthonormal(map, K, p.Lx, p.Ly, fb, &phi_cm, &dk_phi);
  if (static_cast<int>(dk_phi) != dk) {
    std::cerr << "dk mismatch phi vs block " << dk_phi << " vs " << dk << "\n";
    return 2;
  }

  const double tol = 1e-9;
  std::vector<std::complex<double>> a_pk(static_cast<std::size_t>(dk * dk));
  ftlm::symmetry::tiny_pk::phi_dagger_M_phi(d_full, static_cast<int>(dk_phi), phi_cm.data(), pk.m_colmajor.data(),
                                            a_pk.data());

  std::vector<std::complex<double>> phi_PH(static_cast<std::size_t>(dk * dk));
  phi_dagger_P_H_phi(d_full, static_cast<int>(dk_phi), phi_cm.data(), pk.p_colmajor.data(), pk.h_colmajor.data(),
                     phi_PH.data());

  const double chk_phi_PH = max_abs_diff_mat(dk, a_pk.data(), phi_PH.data());
  if (chk_phi_PH > tol) {
    std::cerr << "Phi^dagger P H P_K Phi vs Phi^dagger P H Phi mismatch max=" << chk_phi_PH << "\n";
    return 2;
  }

  std::vector<std::complex<double>> phi_H_phi(static_cast<std::size_t>(dk * dk));
  ftlm::symmetry::tiny_pk::phi_dagger_H_phi(d_full, static_cast<int>(dk_phi), phi_cm.data(), pk.h_colmajor.data(),
                                              phi_H_phi.data());

  ftlm::symmetry::k_block_matrix_free::DenseBlockHamiltonianFromApply h_dense(
      dk, [&](const std::complex<double>* x, std::complex<double>* y) { block.apply(x, y); }, 64);

  const std::vector<std::complex<double>>& h_prod = h_dense.dense_cols();

  const double chk_prod_gram = max_abs_diff_mat(dk, h_prod.data(), phi_H_phi.data());
  if (chk_prod_gram > tol) {
    std::cerr << "production dense vs Phi^dagger H Phi max=" << chk_prod_gram << "\n";
    return 2;
  }

  const double diff_pk_vs_prod = max_abs_diff_mat(dk, a_pk.data(), h_prod.data());
  std::cout << "max|Phi^dagger P_K H P_K Phi - Phi^dagger H Phi| = " << diff_pk_vs_prod << "\n";
  std::cout << "(algebraically same as Phi^dagger H Phi when Phi columns lie in Im(P_K); see k_block_pk_tiny_dense.hpp.)\n";

  std::vector<std::complex<double>> x(static_cast<std::size_t>(dk));
  std::vector<std::complex<double>> y0(static_cast<std::size_t>(dk));
  std::vector<std::complex<double>> y1(static_cast<std::size_t>(dk));

  std::fill(x.begin(), x.end(), std::complex<double>{0.0, 0.0});
  x[0] = {1.0, 0.0};
  matvec_dk(dk, a_pk.data(), x.data(), y0.data());
  h_dense.apply(x.data(), y1.data());
  std::cout << "e0: max|(P_K H P_K block) x - H_prod x| = " << max_abs_diff_vec(dk, y0.data(), y1.data()) << "\n";

  for (int i = 0; i < dk; ++i) {
    x[static_cast<std::size_t>(i)] = {1.0, 0.0};
  }
  matvec_dk(dk, a_pk.data(), x.data(), y0.data());
  h_dense.apply(x.data(), y1.data());
  std::cout << "all-ones: max diff = " << max_abs_diff_vec(dk, y0.data(), y1.data()) << "\n";

  std::mt19937 rng(7);
  std::normal_distribution<double> g(0.0, 1.0);
  for (int i = 0; i < dk; ++i) {
    x[static_cast<std::size_t>(i)] = {g(rng), g(rng)};
  }
  matvec_dk(dk, a_pk.data(), x.data(), y0.data());
  h_dense.apply(x.data(), y1.data());
  const double vec_diff = max_abs_diff_vec(dk, y0.data(), y1.data());

  std::cout << "random x: max|(Phi^dagger P_K H P_K Phi)x - H_prod x| = " << vec_diff << "\n";
  std::cout << "ok tiny independent P_K H P_K matches production block matrix on this sector.\n";
  return 0;
}
