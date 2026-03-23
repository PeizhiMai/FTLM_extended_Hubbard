#pragma once

#include <complex>
#include <functional>
#include <vector>

namespace ftlm {

/// Smallest and largest eigenvalues of a Hermitian operator given only `apply(v, out)` with out = H v.
/// Uses Hermitian Lanczos without full reorthogonalization (smoke / moderate step counts).
struct LanczosExtrema {
  double min_eval = 0.0;
  double max_eval = 0.0;
  int steps_used = 0;
};

LanczosExtrema lanczos_extrema(
    int dim,
    const std::function<void(const std::complex<double>* v, std::complex<double>* Hv)>& apply_h,
    int max_steps, unsigned seed = 1);

/// Hermitian Lanczos without reorthogonalization: fills `alpha` (length = steps used) and `beta`
/// (length = steps used − 1, subdiagonal). Returns Krylov dimension used (0 on failure).
int lanczos_tridiagonal(
    int dim,
    const std::function<void(const std::complex<double>* v, std::complex<double>* Hv)>& apply_h,
    int max_steps, unsigned seed, std::vector<double>* alpha, std::vector<double>* beta);

/// Extrema of a real symmetric tridiagonal matrix (main diag `alpha`, off-diag `beta`, length steps).
std::pair<double, double> tridiagonal_extrema(const std::vector<double>& alpha,
                                              const std::vector<double>& beta, int steps);

}  // namespace ftlm
