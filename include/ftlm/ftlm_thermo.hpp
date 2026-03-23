#pragma once

#include <functional>
#include <vector>

namespace ftlm {

/// Controls stochastic Lanczos depth for FTLM-style canonical traces (real-symmetric H).
struct FtlmParams {
  int n_random = 16;
  int lanczos_steps = 96;
  unsigned seed = 1;
};

/// Stochastic estimate of \(\ln \mathrm{Tr}\,e^{-\beta H}\) for **real-symmetric** \(H\), using
/// Lanczos Gaussian quadrature on random Rademacher/Gaussian starts (no reorthogonalization).
///
/// `apply_real` implements \(y \leftarrow Hx\) with \(x,y \in \mathbb{R}^{\mathrm{dim}}\).
/// For \(\beta=0\) the estimate reduces to \(\ln(\mathrm{dim})\) in expectation.
///
/// **Bias / variance**: finite `lanczos_steps` and `n_random` introduce both; increase them for 4×4.
double ftlm_log_partition_real(int dim,
                               const std::function<void(const double* x, double* y)>& apply_real,
                               double beta, const FtlmParams& par);

}  // namespace ftlm
