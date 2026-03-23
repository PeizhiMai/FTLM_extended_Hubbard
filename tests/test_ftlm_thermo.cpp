#include <cmath>
#include <functional>
#include <iostream>

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
  return 0;
}
