#pragma once

#include <complex>
#include <functional>
#include <vector>

#include "ftlm/lanczos.hpp"

namespace ftlm {

/// Reusable dense Jacobi buffers for Ritz / tridiagonal quadrature (`ftlm_log_tridiagonal_partition_exp`,
/// `ftlm_tridiagonal_ritz_from_lanczos_coeffs`).
struct FtlmTridiagonalQuadratureScratch {
  std::vector<double> T_flat{};
  std::vector<double> V_flat{};
  void shrink_to_fit() {
    T_flat.shrink_to_fit();
    V_flat.shrink_to_fit();
  }
};

/// Controls stochastic Lanczos depth for FTLM-style canonical traces (real-symmetric H).
struct FtlmParams {
  int n_random = 16;
  int lanczos_steps = 96;
  unsigned seed = 1;
  /// If non-null, `ftlm_log_partition_complex` reuses these O(dim) complex buffers across random starts
  /// (and callers can size once per sector to max block dimension). `ftlm_log_partition_real` may also use it
  /// for the internal complex Lanczos path.
  LanczosComplexWorkspace* lanczos_ws = nullptr;
  /// If non-null, tridiagonal Ritz / Jacobi uses these \(n\times n\) reals (one per sector in k-bench).
  FtlmTridiagonalQuadratureScratch* quad_scratch = nullptr;
};

/// Diagonalize the symmetric tridiagonal from Hermitian Lanczos (`alpha` length \(n\), `beta` length \(n-1\)).
/// On success, `eigenvalues[k]` are Ritz energies \(E_k\) (order follows Jacobi) and
/// `w0_squared[k] = |V_{0,k}|^2` is the squared overlap of the **starting** Lanczos vector with Ritz mode \(k\)
/// (Fortran `vka(1,k)^2` in `cond_spect_omp.f`). Rows of `V` are not sorted by energy.
///
/// For \(n=1\), returns `eigenvalues = {alpha[0]}`, `w0_squared = {1}` without Jacobi.
bool ftlm_tridiagonal_ritz_from_lanczos_coeffs(const std::vector<double>& alpha, const std::vector<double>& beta,
                                             std::vector<double>* eigenvalues, std::vector<double>* w0_squared,
                                             FtlmTridiagonalQuadratureScratch* scratch);

/// \(\ln \sum_k w_k \exp(-\beta E_k)\) with log-sum-exp (Fortran inner sum over Ritz levels in `cond_spect_omp.f`).
double ftlm_log_trace_exp_beta_ritz(const std::vector<double>& eigenvalues,
                                    const std::vector<double>& w0_squared, double beta);

/// \(\ln \sum_k |V_{0,k}|^2 \exp(-\beta \lambda_k)\) for Lanczos tridiagonal \(T\) from `alpha`/`beta`.
/// This is the **single random-vector** contribution inside `ftlm_log_partition_*` before the \(\ln \mathrm{dim}\)
/// and sample average.
double ftlm_log_tridiagonal_partition_exp(const std::vector<double>& alpha, const std::vector<double>& beta,
                                          double beta_temp, FtlmTridiagonalQuadratureScratch* scratch);

/// Grand-canonical average filling \(\bar n(\mu) = \langle N\rangle / N_{\mathrm{sites}}\) from **sector**
/// estimates of \(\ln \mathrm{Tr}_s e^{-\beta H}\) (canonical) and fixed electron counts \(N_s\) per sector.
///
/// Sectors are indexed flat as `nu * (n_sites+1) + nd` for `nu, nd \in [0, n_sites]`, matching `bench_ftlm_nmu_rect`.
/// Entries with non-finite `logZ_trace` are skipped.
void ftlm_grandcanonical_density_mu_grid(int n_sites, const std::vector<double>& logZ_trace,
                                         const std::vector<int>& n_elec, double beta,
                                         const std::vector<double>& mu_grid, std::vector<double>* density_out);

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

/// Stochastic estimate of \(\ln \mathrm{Tr}\,e^{-\beta H}\) for complex Hermitian \(H\), using
/// Lanczos Gaussian quadrature on complex random starts (no reorthogonalization).
double ftlm_log_partition_complex(
    int dim,
    const std::function<void(const std::complex<double>* x, std::complex<double>* y)>& apply_h,
    double beta, const FtlmParams& par);

}  // namespace ftlm
