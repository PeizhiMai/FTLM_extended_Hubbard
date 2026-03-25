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
namespace {

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

/// \(\ln \sum_k |V_{0k}|^2 \exp(-\beta \lambda_k)\) (log-sum-exp) for Lanczos tridiagonal.
double log_lanczos_tridiagonal_quadrature_exp(const std::vector<double>& alpha, const std::vector<double>& beta,
                                               double beta_temp) {
  const int n = static_cast<int>(alpha.size());
  if (n <= 0) {
    return -std::numeric_limits<double>::infinity();
  }
  if (n == 1) {
    return -beta_temp * alpha[0];
  }

  // One contiguous pair of n×n buffers (reused across FTLM random starts) instead of nested vectors per call.
  thread_local std::vector<double> T_flat;
  thread_local std::vector<double> V_flat;
  const size_t nn = static_cast<size_t>(n) * static_cast<size_t>(n);
  if (T_flat.size() < nn) {
    T_flat.resize(nn);
  }
  if (V_flat.size() < nn) {
    V_flat.resize(nn);
  }
  std::fill(T_flat.begin(), T_flat.begin() + nn, 0.0);
  for (int i = 0; i < n; ++i) {
    T_flat[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(i)] = alpha[static_cast<size_t>(i)];
  }
  const int nb = static_cast<int>(beta.size());
  for (int i = 0; i < n - 1 && i < nb; ++i) {
    const double b = beta[static_cast<size_t>(i)];
    T_flat[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(i + 1)] = b;
    T_flat[static_cast<size_t>(i + 1) * static_cast<size_t>(n) + static_cast<size_t>(i)] = b;
  }

  const int max_sw = std::max(3000, 120 * n);
  jacobi_symmetric_all_flat(T_flat.data(), V_flat.data(), n, max_sw, 1e-14);

  double max_t = -std::numeric_limits<double>::infinity();
  for (int k = 0; k < n; ++k) {
    const double lam = T_flat[static_cast<size_t>(k) * static_cast<size_t>(n) + static_cast<size_t>(k)];
    const double c0k = V_flat[static_cast<size_t>(k)];  // row 0, col k
    const double w = c0k * c0k;
    if (w <= 0.0) {
      continue;
    }
    const double t = -beta_temp * lam + std::log(w);
    max_t = std::max(max_t, t);
  }
  if (!std::isfinite(max_t)) {
    return -std::numeric_limits<double>::infinity();
  }
  double sum = 0.0;
  for (int k = 0; k < n; ++k) {
    const double lam = T_flat[static_cast<size_t>(k) * static_cast<size_t>(n) + static_cast<size_t>(k)];
    const double c0k = V_flat[static_cast<size_t>(k)];
    const double w = c0k * c0k;
    if (w <= 0.0) {
      continue;
    }
    const double t = -beta_temp * lam + std::log(w);
    sum += std::exp(t - max_t);
  }
  return max_t + std::log(sum);
}

}  // namespace

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

  // Real-symmetric H on complex v: H(v_re + i v_im) = H v_re + i H v_im (both from apply_real).
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
    log_br.push_back(log_lanczos_tridiagonal_quadrature_exp(alpha, beta_td, beta));
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
    log_br.push_back(log_lanczos_tridiagonal_quadrature_exp(alpha, beta_td, beta));
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
