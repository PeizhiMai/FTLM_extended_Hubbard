#include "ftlm/lanczos.hpp"

#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace ftlm {

namespace {

std::complex<double> dotc(const std::vector<std::complex<double>>& a,
                            const std::vector<std::complex<double>>& b) {
  std::complex<double> s(0.0, 0.0);
  for (size_t i = 0; i < a.size(); ++i) {
    s += std::conj(a[i]) * b[i];
  }
  return s;
}

double norm2(const std::vector<std::complex<double>>& v) {
  double s = 0.0;
  for (const auto& z : v) {
    s += std::norm(z);
  }
  return s;
}

void scale(std::vector<std::complex<double>>& v, std::complex<double> s) {
  for (auto& z : v) {
    z *= s;
  }
}

/// Classic Jacobi diagonalization (small n): return min/max diagonal after sweeps.
std::pair<double, double> jacobi_min_max(std::vector<std::vector<double>> A) {
  const int n = static_cast<int>(A.size());
  if (n == 0) {
    return {0.0, 0.0};
  }
  if (n == 1) {
    return {A[0][0], A[0][0]};
  }

  for (int sweep = 0; sweep < 120; ++sweep) {
    int p = 0;
    int q = 1;
    double max_abs = 0.0;
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        const double v = std::abs(A[static_cast<size_t>(i)][static_cast<size_t>(j)]);
        if (v > max_abs) {
          max_abs = v;
          p = i;
          q = j;
        }
      }
    }
    if (max_abs < 1e-15) {
      break;
    }

    const double app = A[static_cast<size_t>(p)][static_cast<size_t>(p)];
    const double aqq = A[static_cast<size_t>(q)][static_cast<size_t>(q)];
    const double apq = A[static_cast<size_t>(p)][static_cast<size_t>(q)];
    const double phi = 0.5 * std::atan2(2.0 * apq, aqq - app);
    const double c = std::cos(phi);
    const double s = std::sin(phi);

    for (int k = 0; k < n; ++k) {
      if (k == p || k == q) {
        continue;
      }
      const double apk = A[static_cast<size_t>(p)][static_cast<size_t>(k)];
      const double aqk = A[static_cast<size_t>(q)][static_cast<size_t>(k)];
      const double rpk = c * apk - s * aqk;
      const double rqk = c * aqk + s * apk;
      A[static_cast<size_t>(p)][static_cast<size_t>(k)] = rpk;
      A[static_cast<size_t>(k)][static_cast<size_t>(p)] = rpk;
      A[static_cast<size_t>(q)][static_cast<size_t>(k)] = rqk;
      A[static_cast<size_t>(k)][static_cast<size_t>(q)] = rqk;
    }

    const double new_pp = c * c * app - 2.0 * s * c * apq + s * s * aqq;
    const double new_qq = s * s * app + 2.0 * s * c * apq + c * c * aqq;
    A[static_cast<size_t>(p)][static_cast<size_t>(q)] = 0.0;
    A[static_cast<size_t>(q)][static_cast<size_t>(p)] = 0.0;
    A[static_cast<size_t>(p)][static_cast<size_t>(p)] = new_pp;
    A[static_cast<size_t>(q)][static_cast<size_t>(q)] = new_qq;
  }

  double lo = std::numeric_limits<double>::infinity();
  double hi = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < n; ++i) {
    const double d = A[static_cast<size_t>(i)][static_cast<size_t>(i)];
    lo = std::min(lo, d);
    hi = std::max(hi, d);
  }
  return {lo, hi};
}

}  // namespace

std::pair<double, double> tridiagonal_extrema(const std::vector<double>& alpha,
                                              const std::vector<double>& beta, int steps) {
  const int n = steps;
  if (n <= 0) {
    return {0.0, 0.0};
  }
  std::vector<std::vector<double>> T(static_cast<size_t>(n), std::vector<double>(static_cast<size_t>(n), 0.0));
  for (int i = 0; i < n; ++i) {
    T[static_cast<size_t>(i)][static_cast<size_t>(i)] = alpha[static_cast<size_t>(i)];
  }
  const int nb = static_cast<int>(beta.size());
  for (int i = 0; i < n - 1 && i < nb; ++i) {
    const double b = beta[static_cast<size_t>(i)];
    T[static_cast<size_t>(i)][static_cast<size_t>(i + 1)] = b;
    T[static_cast<size_t>(i + 1)][static_cast<size_t>(i)] = b;
  }
  return jacobi_min_max(std::move(T));
}

int lanczos_tridiagonal(
    int dim,
    const std::function<void(const std::complex<double>* v, std::complex<double>* Hv)>& apply_h,
    int max_steps, unsigned seed, std::vector<double>* alpha, std::vector<double>* beta,
    LanczosComplexWorkspace* ws) {
  alpha->clear();
  beta->clear();
  if (dim <= 0 || max_steps <= 0) {
    return 0;
  }
  alpha->reserve(static_cast<size_t>(max_steps));
  if (max_steps > 1) {
    beta->reserve(static_cast<size_t>(max_steps - 1));
  }

  std::mt19937 rng(seed);
  std::normal_distribution<double> gauss(0.0, 1.0);

  thread_local std::vector<std::complex<double>> tl_q;
  thread_local std::vector<std::complex<double>> tl_q_prev;
  thread_local std::vector<std::complex<double>> tl_w;

  std::vector<std::complex<double>>* p_q = nullptr;
  std::vector<std::complex<double>>* p_qp = nullptr;
  std::vector<std::complex<double>>* p_w = nullptr;
  if (ws != nullptr) {
    ws->ensure(dim);
    p_q = &ws->q;
    p_qp = &ws->q_prev;
    p_w = &ws->w;
  } else {
    tl_q.resize(static_cast<size_t>(dim));
    tl_q_prev.assign(static_cast<size_t>(dim), std::complex<double>(0.0, 0.0));
    tl_w.resize(static_cast<size_t>(dim));
    p_q = &tl_q;
    p_qp = &tl_q_prev;
    p_w = &tl_w;
  }
  std::vector<std::complex<double>>& q = *p_q;
  std::vector<std::complex<double>>& q_prev = *p_qp;
  std::vector<std::complex<double>>& w = *p_w;

  for (int i = 0; i < dim; ++i) {
    q[static_cast<size_t>(i)] = {gauss(rng), gauss(rng)};
  }
  const double n0 = std::sqrt(norm2(q));
  if (n0 < 1e-18) {
    q[0] = {1.0, 0.0};
  } else {
    scale(q, {1.0 / n0, 0.0});
  }

  double beta_prev = 0.0;
  int steps_used = 0;

  for (int k = 0; k < max_steps; ++k) {
    apply_h(q.data(), w.data());
    const std::complex<double> z = dotc(q, w);
    const double a_k = z.real();
    for (int i = 0; i < dim; ++i) {
      w[static_cast<size_t>(i)] -= z * q[static_cast<size_t>(i)];
    }
    if (k > 0) {
      for (int i = 0; i < dim; ++i) {
        w[static_cast<size_t>(i)] -= beta_prev * q_prev[static_cast<size_t>(i)];
      }
    }
    const double nw2 = norm2(w);
    alpha->push_back(a_k);
    if (nw2 < 1e-24) {
      steps_used = k + 1;
      break;
    }
    const double b_k = std::sqrt(nw2);
    beta->push_back(b_k);
    std::swap(q, q_prev);
    for (int i = 0; i < dim; ++i) {
      q[static_cast<size_t>(i)] = w[static_cast<size_t>(i)] / b_k;
    }
    beta_prev = b_k;
    steps_used = k + 1;
  }

  return steps_used;
}

LanczosExtrema lanczos_extrema(
    int dim,
    const std::function<void(const std::complex<double>* v, std::complex<double>* Hv)>& apply_h,
    int max_steps, unsigned seed) {
  LanczosExtrema out{};
  std::vector<double> alpha;
  std::vector<double> beta;
  const int used = lanczos_tridiagonal(dim, apply_h, max_steps, seed, &alpha, &beta);
  out.steps_used = used;
  if (used == 0) {
    return out;
  }
  std::vector<double> beta_trim;
  if (used >= 2) {
    beta_trim.assign(beta.begin(), beta.begin() + (used - 1));
  }
  const auto mm = tridiagonal_extrema(alpha, beta_trim, used);
  out.min_eval = mm.first;
  out.max_eval = mm.second;
  return out;
}

}  // namespace ftlm
