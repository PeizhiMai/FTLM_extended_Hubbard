#include <cmath>
#include <iostream>
#include <vector>

#include "ftlm/ftlm_thermo.hpp"

namespace {

void apply_diag2(const double* x, double* y) {
  y[0] = 2.0 * x[0];
  y[1] = 3.0 * x[1];
}

}  // namespace

int main() {
  const double beta = 1.0;
  const double exact_logZ = std::log(std::exp(-2.0 * beta) + std::exp(-3.0 * beta));

  ftlm::FtlmParams par;
  par.n_random = 48;
  par.lanczos_steps = 32;
  par.seed = 12345u;

  const double est = ftlm::ftlm_log_partition_real(2, apply_diag2, beta, par);
  const double err = std::abs(est - exact_logZ);
  std::cout << "exact logZ=" << exact_logZ << "  FTLM=" << est << "  |diff|=" << err << "\n";
  if (err > 0.15) {
    std::cerr << "test_ftlm_thermo: error too large\n";
    return 1;
  }

  // Ritz path must match ftlm_log_tridiagonal_partition_exp (2×2 diagonal H → T diagonal in one step).
  ftlm::FtlmTridiagonalQuadratureScratch scratch;
  std::vector<double> alpha = {2.0, 3.0};
  std::vector<double> beta_td = {0.0};  // diagonal T (zero subdiagonal)
  std::vector<double> evals;
  std::vector<double> w0;
  if (!ftlm::ftlm_tridiagonal_ritz_from_lanczos_coeffs(alpha, beta_td, &evals, &w0, &scratch)) {
    std::cerr << "test_ftlm_thermo: ritz decomposition failed\n";
    return 1;
  }
  double sum_w = 0.0;
  for (double w : w0) {
    sum_w += w;
  }
  if (std::abs(sum_w - 1.0) > 1e-8) {
    std::cerr << "test_ftlm_thermo: sum |V_{0k}|^2 should be 1, got " << sum_w << "\n";
    return 1;
  }
  const double logp = ftlm::ftlm_log_tridiagonal_partition_exp(alpha, beta_td, beta, &scratch);
  const double logp2 = ftlm::ftlm_log_trace_exp_beta_ritz(evals, w0, beta);
  if (std::abs(logp - logp2) > 1e-12) {
    std::cerr << "test_ftlm_thermo: partition exp mismatch " << logp << " vs " << logp2 << "\n";
    return 1;
  }

  // Grand-canonical combine: two fake sectors (dim irrelevant), N=0 and N=2, equal logZ → n̄ = 1 at μ=0.
  const int n_sites = 2;
  const int nsec = (n_sites + 1) * (n_sites + 1);
  std::vector<double> logZ(static_cast<size_t>(nsec), -1e300);
  std::vector<int> nelec(static_cast<size_t>(nsec), 0);
  logZ[0] = 0.0;
  nelec[0] = 0;
  const int idx_11 = 1 * (n_sites + 1) + 1;  // sector (nu,nd)=(1,1), N=2
  logZ[static_cast<size_t>(idx_11)] = 0.0;
  nelec[static_cast<size_t>(idx_11)] = 2;
  std::vector<double> mu_grid = {0.0};
  std::vector<double> dens;
  ftlm::ftlm_grandcanonical_density_mu_grid(n_sites, logZ, nelec, beta, mu_grid, &dens);
  // Z = e^{βμ·0} + e^{βμ·2} = 2 at μ=0. ⟨N⟩ = (0+2)/2 = 1. n̄ = 1/n_sites = 0.5.
  const double expect_n = 0.5;
  if (std::abs(dens[0] - expect_n) > 1e-12) {
    std::cerr << "test_ftlm_thermo: grandcanonical density got " << dens[0] << " expect " << expect_n << "\n";
    return 1;
  }

  std::cout << "test_ftlm_thermo: ok (ritz + grandcanonical)\n";
  return 0;
}
