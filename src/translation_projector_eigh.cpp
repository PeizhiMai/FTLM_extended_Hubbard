#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

#include "ftlm/symmetry/translation_projector_dense.hpp"

#if defined(__APPLE__)
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK 1
#endif
#include <Accelerate/Accelerate.h>
#else
extern "C" {
void zheev_(char* jobz, char* uplo, int* n, std::complex<double>* a, int* lda, double* w,
            std::complex<double>* work, int* lwork, double* rwork, int* info);
}
#endif

namespace ftlm {
namespace symmetry {
namespace detail {

void eigh_hermitian_small(const std::vector<std::vector<std::complex<double>>>& H, std::vector<double>* evals,
                          std::vector<std::vector<std::complex<double>>>* evec_columns, double tol_pair) {
  (void)tol_pair;
  const int n = static_cast<int>(H.size());
  if (n == 0) {
    evals->clear();
    evec_columns->clear();
    return;
  }

#if defined(__APPLE__)
  std::vector<std::complex<double>> a(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      a[static_cast<std::size_t>(j * n + i)] = H[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
    }
  }
  char jobz = 'V';
  char uplo = 'U';
  __LAPACK_int nn = static_cast<__LAPACK_int>(n);
  __LAPACK_int lda = static_cast<__LAPACK_int>(n);
  std::vector<double> w(static_cast<std::size_t>(n));
  std::vector<std::complex<double>> work(1);
  __LAPACK_int lwork = -1;
  std::vector<double> rwork(static_cast<std::size_t>(std::max(1, 3 * n - 2)));
  __LAPACK_int info = 0;
  zheev_(&jobz, &uplo, &nn, reinterpret_cast<__LAPACK_double_complex*>(a.data()), &lda, w.data(),
         reinterpret_cast<__LAPACK_double_complex*>(work.data()), &lwork, rwork.data(), &info);
  if (info != 0) {
    throw std::runtime_error("zheev workspace query failed (info=" + std::to_string(static_cast<int>(info)) + ")");
  }
  lwork = static_cast<__LAPACK_int>(std::llround(work[0].real()));
  if (lwork < 1) {
    lwork = 1;
  }
  work.resize(static_cast<std::size_t>(lwork));
  zheev_(&jobz, &uplo, &nn, reinterpret_cast<__LAPACK_double_complex*>(a.data()), &lda, w.data(),
         reinterpret_cast<__LAPACK_double_complex*>(work.data()), &lwork, rwork.data(), &info);
  if (info != 0) {
    throw std::runtime_error("zheev failed (info=" + std::to_string(static_cast<int>(info)) + ")");
  }
  evals->assign(w.begin(), w.end());
  evec_columns->assign(static_cast<std::size_t>(n), std::vector<std::complex<double>>(static_cast<std::size_t>(n), 0.0));
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      (*evec_columns)[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
          a[static_cast<std::size_t>(j * n + i)];
    }
  }
#else
  std::vector<std::complex<double>> a(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      a[static_cast<std::size_t>(j * n + i)] = H[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
    }
  }
  char jobz = 'V';
  char uplo = 'U';
  int nn = n;
  int lda = n;
  std::vector<double> w(static_cast<std::size_t>(n));
  std::vector<std::complex<double>> work(1);
  int lwork = -1;
  std::vector<double> rwork(static_cast<std::size_t>(std::max(1, 3 * n - 2)));
  int info = 0;
  zheev_(&jobz, &uplo, &nn, a.data(), &lda, w.data(), work.data(), &lwork, rwork.data(), &info);
  if (info != 0) {
    throw std::runtime_error("zheev workspace query failed (info=" + std::to_string(info) + ")");
  }
  lwork = static_cast<int>(std::llround(work[0].real()));
  if (lwork < 1) {
    lwork = 1;
  }
  work.resize(static_cast<std::size_t>(lwork));
  zheev_(&jobz, &uplo, &nn, a.data(), &lda, w.data(), work.data(), &lwork, rwork.data(), &info);
  if (info != 0) {
    throw std::runtime_error("zheev failed (info=" + std::to_string(info) + ")");
  }
  evals->assign(w.begin(), w.end());
  evec_columns->assign(static_cast<std::size_t>(n), std::vector<std::complex<double>>(static_cast<std::size_t>(n), 0.0));
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      (*evec_columns)[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
          a[static_cast<std::size_t>(j * n + i)];
    }
  }
#endif
}

}  // namespace detail
}  // namespace symmetry
}  // namespace ftlm
