#include "ftlm/symmetry/k_block_im_pk_reduced.hpp"

#include "ftlm/symmetry/k_block_pk_tiny_dense.hpp"
#include "ftlm/symmetry/translation_projector_dense.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

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
namespace {

void zheev_hermitian_vectors(int n, std::vector<std::complex<double>>* a_colmajor, std::vector<double>* w) {
  if (n <= 0) {
    w->clear();
    return;
  }
#if defined(__APPLE__)
  char jobz = 'V';
  char uplo = 'U';
  __LAPACK_int nn = static_cast<__LAPACK_int>(n);
  __LAPACK_int lda = static_cast<__LAPACK_int>(n);
  w->resize(static_cast<std::size_t>(n));
  std::vector<std::complex<double>> work(1);
  __LAPACK_int lwork = -1;
  std::vector<double> rwork(static_cast<std::size_t>(std::max(1, 3 * n - 2)));
  __LAPACK_int info = 0;
  zheev_(&jobz, &uplo, &nn, reinterpret_cast<__LAPACK_double_complex*>(a_colmajor->data()), &lda, w->data(),
         reinterpret_cast<__LAPACK_double_complex*>(work.data()), &lwork, rwork.data(), &info);
  if (info != 0) {
    throw std::runtime_error("k_block_im_pk_reduced: zheev workspace query failed");
  }
  lwork = static_cast<__LAPACK_int>(std::llround(work[0].real()));
  if (lwork < 1) {
    lwork = 1;
  }
  work.resize(static_cast<std::size_t>(lwork));
  zheev_(&jobz, &uplo, &nn, reinterpret_cast<__LAPACK_double_complex*>(a_colmajor->data()), &lda, w->data(),
         reinterpret_cast<__LAPACK_double_complex*>(work.data()), &lwork, rwork.data(), &info);
  if (info != 0) {
    throw std::runtime_error("k_block_im_pk_reduced: zheev failed");
  }
#else
  char jobz = 'V';
  char uplo = 'U';
  int nn = n;
  int lda = n;
  w->resize(static_cast<std::size_t>(n));
  std::vector<std::complex<double>> work(1);
  int lwork = -1;
  std::vector<double> rwork(static_cast<std::size_t>(std::max(1, 3 * n - 2)));
  int info = 0;
  zheev_(&jobz, &uplo, &nn, a_colmajor->data(), &lda, w->data(), work.data(), &lwork, rwork.data(), &info);
  if (info != 0) {
    throw std::runtime_error("k_block_im_pk_reduced: zheev workspace query failed");
  }
  lwork = static_cast<int>(std::llround(work[0].real()));
  if (lwork < 1) {
    lwork = 1;
  }
  work.resize(static_cast<std::size_t>(lwork));
  zheev_(&jobz, &uplo, &nn, a_colmajor->data(), &lda, w->data(), work.data(), &lwork, rwork.data(), &info);
  if (info != 0) {
    throw std::runtime_error("k_block_im_pk_reduced: zheev failed");
  }
#endif
}

}  // namespace

bool build_tiny_im_pk_reduced_hamiltonian(const HubbardParams& p, const FockBasis& fb, int lx, int ly,
                                          MomentumSector K, const std::vector<SpinfulHopping>& hoppings,
                                          const std::vector<NearestPair>& nn_pairs, int max_d_full,
                                          TinyImPkReducedHamiltonian* out) {
  const int d = fb.dim();
  out->d_full = d;
  out->r = 0;
  out->u_colmajor.clear();
  out->h_red_colmajor.clear();
  if (d <= 0) {
    return false;
  }
  if (d > max_d_full) {
    throw std::invalid_argument("build_tiny_im_pk_reduced_hamiltonian: d_full exceeds max_d_full (prototype only)");
  }

  std::vector<std::vector<std::complex<double>>> p_vv;
  detail::build_translation_projector_dense(fb, lx, ly, K, &p_vv);
  detail::hermitian_symmetrize_inplace(&p_vv);
  std::vector<std::complex<double>> p_cm;
  tiny_pk::vv_to_colmajor(p_vv, &p_cm);

  std::vector<double> w_p;
  zheev_hermitian_vectors(d, &p_cm, &w_p);

  int r = 0;
  for (int j = 0; j < d; ++j) {
    if (w_p[static_cast<std::size_t>(j)] > 0.5) {
      ++r;
    }
  }
  if (r == 0) {
    return false;
  }

  out->r = r;
  out->u_colmajor.assign(static_cast<std::size_t>(d * r), std::complex<double>{0.0, 0.0});
  int col = 0;
  for (int j = 0; j < d; ++j) {
    if (w_p[static_cast<std::size_t>(j)] <= 0.5) {
      continue;
    }
    for (int i = 0; i < d; ++i) {
      out->u_colmajor[static_cast<std::size_t>(i + col * d)] = p_cm[static_cast<std::size_t>(i + j * d)];
    }
    ++col;
  }

  std::vector<std::complex<double>> h_cm;
  tiny_pk::build_dense_h_sector(p, fb, hoppings, nn_pairs, &h_cm);

  out->h_red_colmajor.assign(static_cast<std::size_t>(r * r), std::complex<double>{0.0, 0.0});
  std::vector<std::complex<double>> hv(static_cast<std::size_t>(d));
  std::vector<std::complex<double>> uj(static_cast<std::size_t>(d));
  for (int jj = 0; jj < r; ++jj) {
    for (int i = 0; i < d; ++i) {
      uj[static_cast<std::size_t>(i)] = out->u_colmajor[static_cast<std::size_t>(i + jj * d)];
    }
    for (int i = 0; i < d; ++i) {
      std::complex<double> s{0.0, 0.0};
      for (int k = 0; k < d; ++k) {
        s += h_cm[static_cast<std::size_t>(i + k * d)] * uj[static_cast<std::size_t>(k)];
      }
      hv[static_cast<std::size_t>(i)] = s;
    }
    for (int ii = 0; ii < r; ++ii) {
      std::complex<double> s{0.0, 0.0};
      for (int p = 0; p < d; ++p) {
        s += std::conj(out->u_colmajor[static_cast<std::size_t>(p + ii * d)]) * hv[static_cast<std::size_t>(p)];
      }
      out->h_red_colmajor[static_cast<std::size_t>(ii + jj * r)] = s;
    }
  }

  tiny_pk::hermitian_symmetrize_flat(r, &out->h_red_colmajor);
  return true;
}

}  // namespace symmetry
}  // namespace ftlm
