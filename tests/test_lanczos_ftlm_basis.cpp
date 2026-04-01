#include <cmath>
#include <complex>
#include <iostream>
#include <random>
#include <vector>

#include "ftlm/lanczos.hpp"

namespace {

std::complex<double> dotc(int dim, const std::complex<double>* a, const std::complex<double>* b) {
  std::complex<double> s(0.0, 0.0);
  for (int i = 0; i < dim; ++i) {
    s += std::conj(a[i]) * b[i];
  }
  return s;
}

void hemv(int n, const std::vector<std::complex<double>>& H, const std::complex<double>* x,
          std::complex<double>* y) {
  for (int i = 0; i < n; ++i) {
    std::complex<double> s(0.0, 0.0);
    for (int j = 0; j < n; ++j) {
      s += H[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)] * x[j];
    }
    y[i] = s;
  }
}

}  // namespace

int main() {
  constexpr int n = 5;
  std::mt19937 rng(42);
  std::normal_distribution<double> g(0.0, 1.0);
  std::vector<std::complex<double>> H(static_cast<size_t>(n) * static_cast<size_t>(n), {0.0, 0.0});
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      const std::complex<double> z{g(rng), g(rng)};
      H[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)] = z;
      H[static_cast<size_t>(j) * static_cast<size_t>(n) + static_cast<size_t>(i)] = std::conj(z);
    }
    H[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(i)] = {std::abs(g(rng)) + 0.1, 0.0};
  }

  auto apply = [&](const std::complex<double>* x, std::complex<double>* y) { hemv(n, H, x, y); };

  std::vector<double> alpha;
  std::vector<double> beta;
  ftlm::LanczosFullBasisBuffer basis;
  const int steps = ftlm::lanczos_tridiagonal(n, apply, 20, 7u, &alpha, &beta, nullptr, &basis);
  if (steps <= 1) {
    std::cerr << "test_lanczos_ftlm_basis: expected steps > 1\n";
    return 1;
  }

  for (int i = 0; i < steps; ++i) {
    for (int j = 0; j < steps; ++j) {
      const std::complex<double> dij = dotc(n, basis.column(i), basis.column(j));
      const double target = (i == j) ? 1.0 : 0.0;
      if (std::abs(dij.real() - target) > 1e-7 || std::abs(dij.imag()) > 1e-7) {
        std::cerr << "test_lanczos_ftlm_basis: orthogonality fail i=" << i << " j=" << j << " -> " << dij << "\n";
        return 1;
      }
    }
  }

  std::vector<std::complex<double>> hv(static_cast<size_t>(n));
  std::vector<std::complex<double>> r(static_cast<size_t>(n));
  for (int k = 0; k < static_cast<int>(beta.size()); ++k) {
    hemv(n, H, basis.column(k), hv.data());
    for (int i = 0; i < n; ++i) {
      r[static_cast<size_t>(i)] = hv[static_cast<size_t>(i)];
      r[static_cast<size_t>(i)] -= alpha[static_cast<size_t>(k)] * basis.column(k)[static_cast<size_t>(i)];
    }
    if (k > 0) {
      for (int i = 0; i < n; ++i) {
        r[static_cast<size_t>(i)] -= beta[static_cast<size_t>(k - 1)] * basis.column(k - 1)[static_cast<size_t>(i)];
      }
    }
    for (int i = 0; i < n; ++i) {
      r[static_cast<size_t>(i)] -= beta[static_cast<size_t>(k)] * basis.column(k + 1)[static_cast<size_t>(i)];
    }
    double nr = 0.0;
    for (int i = 0; i < n; ++i) {
      nr += std::norm(r[static_cast<size_t>(i)]);
    }
    if (std::sqrt(nr) > 1e-6) {
      std::cerr << "test_lanczos_ftlm_basis: three-term residual k=" << k << " ||r||=" << std::sqrt(nr) << "\n";
      return 1;
    }
  }

  std::vector<double> packed;
  ftlm::lanczos_tridiagonal_pack_coeffs(alpha, beta, &packed);
  std::vector<double> a2;
  std::vector<double> b2;
  if (!ftlm::lanczos_tridiagonal_unpack_coeffs(packed, &a2, &b2)) {
    std::cerr << "test_lanczos_ftlm_basis: unpack failed\n";
    return 1;
  }
  if (a2 != alpha || b2 != beta) {
    std::cerr << "test_lanczos_ftlm_basis: pack roundtrip mismatch\n";
    return 1;
  }

  std::cout << "test_lanczos_ftlm_basis ok\n";
  return 0;
}
