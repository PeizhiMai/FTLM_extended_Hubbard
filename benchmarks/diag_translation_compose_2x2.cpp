// Check U(ex,ey) vs U(ex,0)*U(0,ey) for fermionic translation matrices (sector 0,2) on 2x2.
#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <vector>

#include "ftlm/fock_basis.hpp"
#include "ftlm/symmetry/translation_projector_dense.hpp"

namespace {

void matmul(const std::vector<std::vector<std::complex<double>>>& A,
            const std::vector<std::vector<std::complex<double>>>& B,
            std::vector<std::vector<std::complex<double>>>* C) {
  const int n = static_cast<int>(A.size());
  const int m = static_cast<int>(B[0].size());
  const int k = static_cast<int>(B.size());
  C->assign(static_cast<size_t>(n), std::vector<std::complex<double>>(static_cast<size_t>(m), 0.0));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      std::complex<double> z(0.0, 0.0);
      for (int t = 0; t < k; ++t) {
        z += A[static_cast<size_t>(i)][static_cast<size_t>(t)] * B[static_cast<size_t>(t)][static_cast<size_t>(j)];
      }
      (*C)[static_cast<size_t>(i)][static_cast<size_t>(j)] = z;
    }
  }
}

double max_abs_diff(const std::vector<std::vector<std::complex<double>>>& A,
                      const std::vector<std::vector<std::complex<double>>>& B) {
  double m = 0.0;
  const int n = static_cast<int>(A.size());
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      m = std::max(m, std::abs(A[static_cast<size_t>(i)][static_cast<size_t>(j)] -
                               B[static_cast<size_t>(i)][static_cast<size_t>(j)]));
    }
  }
  return m;
}

}  // namespace

int main() {
  constexpr int Lx = 2;
  constexpr int Ly = 2;
  constexpr int Ns = 4;
  ftlm::FockBasis fb(Ns, 0, 2);
  const int d = fb.dim();

  std::vector<std::vector<std::complex<double>>> U10, U01, U11;
  ftlm::symmetry::detail::build_dense_translation_U(fb, Lx, Ly, 1, 0, &U10);
  ftlm::symmetry::detail::build_dense_translation_U(fb, Lx, Ly, 0, 1, &U01);
  ftlm::symmetry::detail::build_dense_translation_U(fb, Lx, Ly, 1, 1, &U11);

  std::vector<std::vector<std::complex<double>>> prod, prod2;
  matmul(U10, U01, &prod);
  matmul(U01, U10, &prod2);

  std::cout << std::setprecision(17);
  std::cout << "sector (0,2) dim=" << d << "\n";
  std::cout << "max|U(1,0)U(0,1) - U(1,1)|=" << max_abs_diff(prod, U11) << "\n";
  std::cout << "max|U(0,1)U(1,0) - U(1,1)|=" << max_abs_diff(prod2, U11) << "\n";
  std::cout << "max|U(1,0)U(0,1) - U(0,1)U(1,0)|=" << max_abs_diff(prod, prod2) << "\n";

  std::vector<std::vector<std::complex<double>>> I(
      static_cast<size_t>(d), std::vector<std::complex<double>>(static_cast<size_t>(d), 0.0));
  for (int i = 0; i < d; ++i) {
    I[static_cast<size_t>(i)][static_cast<size_t>(i)] = 1.0;
  }
  std::vector<std::vector<std::complex<double>>> S = I;
  for (int i = 0; i < d; ++i) {
    for (int j = 0; j < d; ++j) {
      S[static_cast<size_t>(i)][static_cast<size_t>(j)] +=
          U10[static_cast<size_t>(i)][static_cast<size_t>(j)] + U01[static_cast<size_t>(i)][static_cast<size_t>(j)] +
          U11[static_cast<size_t>(i)][static_cast<size_t>(j)];
    }
  }
  double mx = 0.0;
  for (int i = 0; i < d; ++i) {
    for (int j = 0; j < d; ++j) {
      mx = std::max(mx, std::abs(S[static_cast<size_t>(i)][static_cast<size_t>(j)]));
    }
  }
  std::cout << "max_abs(I+U10+U01+U11)=" << mx << " (expect 0 if Gamma projector sum vanishes)\n";
  return 0;
}
