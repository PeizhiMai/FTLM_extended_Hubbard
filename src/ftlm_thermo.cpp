#include "ftlm/ftlm_thermo.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include "ftlm/lanczos.hpp"

namespace ftlm {
namespace detail {

/// Row-major `A[n*n]`, `V[n*n]` (starts as identity); matches the legacy `vector<vector<double>>` Jacobi.
void jacobi_symmetric_all_flat(double* A, double* V, int n, int max_sweeps, double tol_offdiag) {
  const size_t nn = static_cast<size_t>(n) * static_cast<size_t>(n);
  std::fill(V, V + nn, 0.0);
  for (int i = 0; i < n; ++i) {
    V[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(i)] = 1.0;
  }
  if (n <= 1) {
    return;
  }

  for (int sweep = 0; sweep < max_sweeps; ++sweep) {
    int p = 0;
    int q = 1;
    double max_abs = 0.0;
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        const double v = std::abs(A[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)]);
        if (v > max_abs) {
          max_abs = v;
          p = i;
          q = j;
        }
      }
    }
    if (max_abs < tol_offdiag) {
      break;
    }

    const double app = A[static_cast<size_t>(p) * static_cast<size_t>(n) + static_cast<size_t>(p)];
    const double aqq = A[static_cast<size_t>(q) * static_cast<size_t>(n) + static_cast<size_t>(q)];
    const double apq = A[static_cast<size_t>(p) * static_cast<size_t>(n) + static_cast<size_t>(q)];
    const double phi = 0.5 * std::atan2(2.0 * apq, aqq - app);
    const double c = std::cos(phi);
    const double s = std::sin(phi);

    for (int k = 0; k < n; ++k) {
      if (k == p || k == q) {
        continue;
      }
      const double apk = A[static_cast<size_t>(p) * static_cast<size_t>(n) + static_cast<size_t>(k)];
      const double aqk = A[static_cast<size_t>(q) * static_cast<size_t>(n) + static_cast<size_t>(k)];
      const double rpk = c * apk - s * aqk;
      const double rqk = c * aqk + s * apk;
      A[static_cast<size_t>(p) * static_cast<size_t>(n) + static_cast<size_t>(k)] = rpk;
      A[static_cast<size_t>(k) * static_cast<size_t>(n) + static_cast<size_t>(p)] = rpk;
      A[static_cast<size_t>(q) * static_cast<size_t>(n) + static_cast<size_t>(k)] = rqk;
      A[static_cast<size_t>(k) * static_cast<size_t>(n) + static_cast<size_t>(q)] = rqk;
    }

    const double new_pp = c * c * app - 2.0 * s * c * apq + s * s * aqq;
    const double new_qq = s * s * app + 2.0 * s * c * apq + c * c * aqq;
    A[static_cast<size_t>(p) * static_cast<size_t>(n) + static_cast<size_t>(q)] = 0.0;
    A[static_cast<size_t>(q) * static_cast<size_t>(n) + static_cast<size_t>(p)] = 0.0;
    A[static_cast<size_t>(p) * static_cast<size_t>(n) + static_cast<size_t>(p)] = new_pp;
    A[static_cast<size_t>(q) * static_cast<size_t>(n) + static_cast<size_t>(q)] = new_qq;

    for (int i = 0; i < n; ++i) {
      const double vip = V[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(p)];
      const double viq = V[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(q)];
      V[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(p)] = c * vip - s * viq;
      V[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(q)] = s * vip + c * viq;
    }
  }
}

void fill_tridiagonal_matrix(const std::vector<double>& alpha, const std::vector<double>& beta, int n,
                             double* T_flat) {
  const size_t nn = static_cast<size_t>(n) * static_cast<size_t>(n);
  std::fill(T_flat, T_flat + nn, 0.0);
  for (int i = 0; i < n; ++i) {
    T_flat[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(i)] = alpha[static_cast<size_t>(i)];
  }
  const int nb = static_cast<int>(beta.size());
  for (int i = 0; i < n - 1 && i < nb; ++i) {
    const double b = beta[static_cast<size_t>(i)];
    T_flat[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(i + 1)] = b;
    T_flat[static_cast<size_t>(i + 1) * static_cast<size_t>(n) + static_cast<size_t>(i)] = b;
  }
}

}  // namespace detail

bool ftlm_tridiagonal_ritz_from_lanczos_coeffs(const std::vector<double>& alpha, const std::vector<double>& beta,
                                             std::vector<double>* eigenvalues, std::vector<double>* w0_squared,
                                             FtlmTridiagonalQuadratureScratch* scratch) {
  const int n = static_cast<int>(alpha.size());
  if (n <= 0 || eigenvalues == nullptr || w0_squared == nullptr) {
    return false;
  }
  if (n <= 1) {
    eigenvalues->assign(1, alpha[0]);
    w0_squared->assign(1, 1.0);
    return true;
  }
  const int nb_expected = n - 1;
  if (static_cast<int>(beta.size()) < nb_expected) {
    return false;
  }

  std::vector<double> local_T;
  std::vector<double> local_V;
  std::vector<double>* T_ptr = scratch ? &scratch->T_flat : &local_T;
  std::vector<double>* V_ptr = scratch ? &scratch->V_flat : &local_V;
  std::vector<double>& T_flat = *T_ptr;
  std::vector<double>& V_flat = *V_ptr;
  const size_t nn = static_cast<size_t>(n) * static_cast<size_t>(n);
  if (T_flat.size() < nn) {
    T_flat.resize(nn);
  }
  if (V_flat.size() < nn) {
    V_flat.resize(nn);
  }
  detail::fill_tridiagonal_matrix(alpha, beta, n, T_flat.data());

  const int max_sw = std::max(3000, 120 * n);
  detail::jacobi_symmetric_all_flat(T_flat.data(), V_flat.data(), n, max_sw, 1e-14);

  eigenvalues->resize(static_cast<size_t>(n));
  w0_squared->resize(static_cast<size_t>(n));
  for (int k = 0; k < n; ++k) {
    (*eigenvalues)[static_cast<size_t>(k)] =
        T_flat[static_cast<size_t>(k) * static_cast<size_t>(n) + static_cast<size_t>(k)];
    const double c0k = V_flat[static_cast<size_t>(k)];  // row 0, col k
    (*w0_squared)[static_cast<size_t>(k)] = c0k * c0k;
  }
  return true;
}

double ftlm_log_trace_exp_beta_ritz(const std::vector<double>& eigenvalues,
                                    const std::vector<double>& w0_squared, double beta) {
  const int n = static_cast<int>(eigenvalues.size());
  if (n <= 0 || w0_squared.size() != eigenvalues.size()) {
    return -std::numeric_limits<double>::infinity();
  }
  if (n == 1) {
    return -beta * eigenvalues[0] + std::log(std::max(w0_squared[0], 0.0));
  }

  double max_t = -std::numeric_limits<double>::infinity();
  for (int k = 0; k < n; ++k) {
    const double lam = eigenvalues[static_cast<size_t>(k)];
    const double w = w0_squared[static_cast<size_t>(k)];
    if (w <= 0.0) {
      continue;
    }
    const double t = -beta * lam + std::log(w);
    max_t = std::max(max_t, t);
  }
  if (!std::isfinite(max_t)) {
    return -std::numeric_limits<double>::infinity();
  }
  double sum = 0.0;
  for (int k = 0; k < n; ++k) {
    const double lam = eigenvalues[static_cast<size_t>(k)];
    const double w = w0_squared[static_cast<size_t>(k)];
    if (w <= 0.0) {
      continue;
    }
    const double t = -beta * lam + std::log(w);
    sum += std::exp(t - max_t);
  }
  return max_t + std::log(sum);
}

double ftlm_log_tridiagonal_partition_exp(const std::vector<double>& alpha, const std::vector<double>& beta,
                                          double beta_temp, FtlmTridiagonalQuadratureScratch* scratch) {
  std::vector<double> evals;
  std::vector<double> w0;
  if (!ftlm_tridiagonal_ritz_from_lanczos_coeffs(alpha, beta, &evals, &w0, scratch)) {
    return -std::numeric_limits<double>::infinity();
  }
  return ftlm_log_trace_exp_beta_ritz(evals, w0, beta_temp);
}

void ftlm_grandcanonical_density_mu_grid(int n_sites, const std::vector<double>& logZ_trace,
                                         const std::vector<int>& n_elec, double beta,
                                         const std::vector<double>& mu_grid, std::vector<double>* density_out) {
  if (density_out == nullptr) {
    return;
  }
  const int nsec = (n_sites + 1) * (n_sites + 1);
  if (static_cast<int>(logZ_trace.size()) != nsec || static_cast<int>(n_elec.size()) != nsec) {
    density_out->clear();
    return;
  }
  density_out->resize(mu_grid.size());
  for (size_t k = 0; k < mu_grid.size(); ++k) {
    const double mu = mu_grid[k];
    double mx = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < nsec; ++i) {
      const double lz = logZ_trace[static_cast<size_t>(i)];
      if (!std::isfinite(lz)) {
        continue;
      }
      const int N = n_elec[static_cast<size_t>(i)];
      const double ex = lz + beta * mu * static_cast<double>(N);
      mx = std::max(mx, ex);
    }
    if (!std::isfinite(mx)) {
      (*density_out)[k] = 0.0;
      continue;
    }
    double sum_w = 0.0;
    double sum_Nw = 0.0;
    for (int i = 0; i < nsec; ++i) {
      const double lz = logZ_trace[static_cast<size_t>(i)];
      if (!std::isfinite(lz)) {
        continue;
      }
      const int N = n_elec[static_cast<size_t>(i)];
      const double ex = lz + beta * mu * static_cast<double>(N) - mx;
      const double w = std::exp(ex);
      sum_w += w;
      sum_Nw += static_cast<double>(N) * w;
    }
    (*density_out)[k] = (sum_w > 0.0) ? (sum_Nw / sum_w) / static_cast<double>(n_sites) : 0.0;
  }
}

double ftlm_log_partition_real(int dim, const std::function<void(const double* x, double* y)>& apply_real,
                               double beta, const FtlmParams& par) {
  if (dim <= 0) {
    return -std::numeric_limits<double>::infinity();
  }
  if (dim == 1) {
    double x[1] = {1.0};
    double y[1] = {0.0};
    apply_real(x, y);
    return -beta * y[0];
  }

  std::vector<double> xr(static_cast<size_t>(dim));
  std::vector<double> xi(static_cast<size_t>(dim));
  std::vector<double> yr(static_cast<size_t>(dim));
  std::vector<double> yi(static_cast<size_t>(dim));

  const std::function<void(const std::complex<double>*, std::complex<double>*)> apply_c =
      [&](const std::complex<double>* x, std::complex<double>* y) {
        for (int i = 0; i < dim; ++i) {
          xr[static_cast<size_t>(i)] = x[static_cast<size_t>(i)].real();
          xi[static_cast<size_t>(i)] = x[static_cast<size_t>(i)].imag();
        }
        apply_real(xr.data(), yr.data());
        apply_real(xi.data(), yi.data());
        for (int i = 0; i < dim; ++i) {
          y[static_cast<size_t>(i)] = {yr[static_cast<size_t>(i)], yi[static_cast<size_t>(i)]};
        }
      };

  const int R = std::max(1, par.n_random);
  const int M = std::max(1, par.lanczos_steps);

  std::vector<double> log_br;
  log_br.reserve(static_cast<size_t>(R));
  for (int r = 0; r < R; ++r) {
    std::vector<double> alpha;
    std::vector<double> beta_td;
    const unsigned seed_r = par.seed + static_cast<unsigned>(r) * 100003u;
    const int used =
        lanczos_tridiagonal(dim, apply_c, M, seed_r, &alpha, &beta_td, par.lanczos_ws);
    if (used <= 0) {
      continue;
    }
    log_br.push_back(ftlm_log_tridiagonal_partition_exp(alpha, beta_td, beta, par.quad_scratch));
  }
  if (log_br.empty()) {
    return -std::numeric_limits<double>::infinity();
  }

  double max_l = log_br[0];
  for (double x : log_br) {
    max_l = std::max(max_l, x);
  }
  double s = 0.0;
  for (double x : log_br) {
    s += std::exp(x - max_l);
  }
  const double log_mean_br = max_l + std::log(s / static_cast<double>(log_br.size()));
  return std::log(static_cast<double>(dim)) + log_mean_br;
}

double ftlm_log_partition_complex(
    int dim, const std::function<void(const std::complex<double>* x, std::complex<double>* y)>& apply_h,
    double beta, const FtlmParams& par) {
  if (dim <= 0) {
    return -std::numeric_limits<double>::infinity();
  }
  if (dim == 1) {
    std::complex<double> x[1] = {std::complex<double>(1.0, 0.0)};
    std::complex<double> y[1] = {std::complex<double>(0.0, 0.0)};
    apply_h(x, y);
    return -beta * y[0].real();
  }

  const int R = std::max(1, par.n_random);
  const int M = std::max(1, par.lanczos_steps);

  std::vector<double> log_br;
  log_br.reserve(static_cast<size_t>(R));
  for (int r = 0; r < R; ++r) {
    std::vector<double> alpha;
    std::vector<double> beta_td;
    const unsigned seed_r = par.seed + static_cast<unsigned>(r) * 100003u;
    const int used =
        lanczos_tridiagonal(dim, apply_h, M, seed_r, &alpha, &beta_td, par.lanczos_ws);
    if (used <= 0) {
      continue;
    }
    log_br.push_back(ftlm_log_tridiagonal_partition_exp(alpha, beta_td, beta, par.quad_scratch));
  }
  if (log_br.empty()) {
    return -std::numeric_limits<double>::infinity();
  }

  double max_l = log_br[0];
  for (double x : log_br) {
    max_l = std::max(max_l, x);
  }
  double s = 0.0;
  for (double x : log_br) {
    s += std::exp(x - max_l);
  }
  const double log_mean_br = max_l + std::log(s / static_cast<double>(log_br.size()));
  return std::log(static_cast<double>(dim)) + log_mean_br;
}

}  // namespace ftlm
