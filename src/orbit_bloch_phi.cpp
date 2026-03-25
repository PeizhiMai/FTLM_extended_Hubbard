#include "ftlm/symmetry/orbit_bloch_phi.hpp"

#include "ftlm/symmetry/k_basis.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
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

void zheev_full_hermitian_inplace(std::vector<std::complex<double>>* a_colmajor, int n, std::vector<double>* evals) {
  if (n <= 0) {
    evals->clear();
    return;
  }
  // Reused across calls (per thread) so repeated Gram diagonalizations do not allocate O(lwork) each time.
  thread_local static std::vector<std::complex<double>> work_buf;
  thread_local static std::vector<double> rwork_buf;
#if defined(__APPLE__)
  char jobz = 'V';
  char uplo = 'U';
  __LAPACK_int nn = static_cast<__LAPACK_int>(n);
  __LAPACK_int lda = static_cast<__LAPACK_int>(n);
  evals->resize(static_cast<std::size_t>(n));
  if (work_buf.size() < 1) {
    work_buf.resize(1);
  }
  __LAPACK_int lwork = -1;
  rwork_buf.resize(static_cast<std::size_t>(std::max(1, 3 * n - 2)));
  __LAPACK_int info = 0;
  zheev_(&jobz, &uplo, &nn, reinterpret_cast<__LAPACK_double_complex*>(a_colmajor->data()), &lda, evals->data(),
         reinterpret_cast<__LAPACK_double_complex*>(work_buf.data()), &lwork, rwork_buf.data(), &info);
  if (info != 0) {
    throw std::runtime_error("zheev workspace query failed (orbit_bloch_phi)");
  }
  lwork = static_cast<__LAPACK_int>(std::llround(work_buf[0].real()));
  if (lwork < 1) {
    lwork = 1;
  }
  if (static_cast<std::size_t>(work_buf.size()) < static_cast<std::size_t>(lwork)) {
    work_buf.resize(static_cast<std::size_t>(lwork));
  }
  zheev_(&jobz, &uplo, &nn, reinterpret_cast<__LAPACK_double_complex*>(a_colmajor->data()), &lda, evals->data(),
         reinterpret_cast<__LAPACK_double_complex*>(work_buf.data()), &lwork, rwork_buf.data(), &info);
  if (info != 0) {
    throw std::runtime_error("zheev failed (orbit_bloch_phi)");
  }
#else
  char jobz = 'V';
  char uplo = 'U';
  int nn = n;
  int lda = n;
  evals->resize(static_cast<std::size_t>(n));
  if (work_buf.size() < 1) {
    work_buf.resize(1);
  }
  int lwork = -1;
  rwork_buf.resize(static_cast<std::size_t>(std::max(1, 3 * n - 2)));
  int info = 0;
  zheev_(&jobz, &uplo, &nn, a_colmajor->data(), &lda, evals->data(), work_buf.data(), &lwork, rwork_buf.data(), &info);
  if (info != 0) {
    throw std::runtime_error("zheev workspace query failed (orbit_bloch_phi)");
  }
  lwork = static_cast<int>(std::llround(work_buf[0].real()));
  if (lwork < 1) {
    lwork = 1;
  }
  if (static_cast<std::size_t>(work_buf.size()) < static_cast<std::size_t>(lwork)) {
    work_buf.resize(static_cast<std::size_t>(lwork));
  }
  zheev_(&jobz, &uplo, &nn, a_colmajor->data(), &lda, evals->data(), work_buf.data(), &lwork, rwork_buf.data(), &info);
  if (info != 0) {
    throw std::runtime_error("zheev failed (orbit_bloch_phi)");
  }
#endif
}

}  // namespace

static std::complex<double> dot_cols_conj_left(const std::complex<double>* a, const std::complex<double>* b, int d) {
  std::complex<double> s(0.0, 0.0);
  for (int p = 0; p < d; ++p) {
    s += std::conj(a[static_cast<std::size_t>(p)]) * b[static_cast<std::size_t>(p)];
  }
  return s;
}

bool MomentumPhiGramBasis::build(const MomentumSectorMap& orbit_map, MomentumSector K, int lx, int ly,
                                 const FockBasis& fb, MomentumPhiGramBasis* out) {
  out->seeds = momentum_phi_seeds(orbit_map, K);
  out->d_full = fb.dim();
  out->lx = lx;
  out->ly = ly;
  out->k_in = out->seeds.size();
  out->k_out = 0;
  out->V.clear();
  out->evals.clear();
  const int d = out->d_full;
  const std::size_t k_in = out->k_in;
  if (k_in == 0 || d <= 0) {
    return false;
  }

  std::vector<std::complex<double>> col_i(static_cast<std::size_t>(d));
  std::vector<std::complex<double>> col_j(static_cast<std::size_t>(d));
  std::vector<std::complex<double>> g(static_cast<std::size_t>(k_in) * k_in, std::complex<double>(0.0, 0.0));

  for (std::size_t i = 0; i < k_in; ++i) {
    detail::fill_phi_orbit_bloch_from_seed(orbit_map, K, lx, ly, fb, out->seeds[i], col_i.data());
    for (std::size_t j = i; j < k_in; ++j) {
      detail::fill_phi_orbit_bloch_from_seed(orbit_map, K, lx, ly, fb, out->seeds[j], col_j.data());
      const std::complex<double> s = dot_cols_conj_left(col_i.data(), col_j.data(), d);
      g[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * k_in] = s;
      if (i != j) {
        g[static_cast<std::size_t>(j) + static_cast<std::size_t>(i) * k_in] = std::conj(s);
      }
    }
  }
  for (std::size_t i = 0; i < k_in; ++i) {
    for (std::size_t j = i; j < k_in; ++j) {
      const std::complex<double> a = g[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * k_in];
      const std::complex<double> b = g[static_cast<std::size_t>(j) + static_cast<std::size_t>(i) * k_in];
      const std::complex<double> h = 0.5 * (a + std::conj(b));
      g[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * k_in] = h;
      g[static_cast<std::size_t>(j) + static_cast<std::size_t>(i) * k_in] = std::conj(h);
    }
  }

  std::vector<double> evals_all;
  zheev_full_hermitian_inplace(&g, static_cast<int>(k_in), &evals_all);

  double lam_max = 0.0;
  for (double ev : evals_all) {
    lam_max = std::max(lam_max, ev);
  }
  const double tol_ev = std::max(1e-14 * std::max(1.0, lam_max), 1e-20);

  std::vector<std::size_t> keep_idx;
  keep_idx.reserve(k_in);
  for (std::size_t j = 0; j < k_in; ++j) {
    if (evals_all[j] > tol_ev) {
      keep_idx.push_back(j);
    }
  }
  out->k_out = keep_idx.size();

  out->V.clear();
  out->V.reserve(k_in * out->k_out);
  for (std::size_t r = 0; r < out->k_out; ++r) {
    const std::size_t ej = keep_idx[r];
    for (std::size_t m = 0; m < k_in; ++m) {
      out->V.push_back(g[static_cast<std::size_t>(m) + ej * k_in]);
    }
  }
  out->evals.clear();
  out->evals.reserve(out->k_out);
  for (std::size_t r = 0; r < out->k_out; ++r) {
    out->evals.push_back(evals_all[keep_idx[r]]);
  }
  return true;
}

void MomentumPhiGramBasis::project_block_from_full(const MomentumSectorMap& orbit_map, MomentumSector K,
                                                   const FockBasis& fb, const std::complex<double>* x_full,
                                                   std::complex<double>* y_block,
                                                   MomentumPhiGramApplyScratch* scratch) const {
  if (k_out == 0) {
    return;
  }
  const int d = d_full;
  thread_local std::vector<std::complex<double>> tl_a;
  thread_local std::vector<std::complex<double>> tl_col;
  std::vector<std::complex<double>>* a = nullptr;
  std::vector<std::complex<double>>* col = nullptr;
  if (scratch != nullptr) {
    a = &scratch->a;
    col = &scratch->col;
  } else {
    a = &tl_a;
    col = &tl_col;
  }
  a->resize(k_in);
  col->resize(static_cast<std::size_t>(d));

  for (std::size_t m = 0; m < k_in; ++m) {
    detail::fill_phi_orbit_bloch_from_seed(orbit_map, K, lx, ly, fb, seeds[m], col->data());
    std::complex<double> am(0.0, 0.0);
    for (int p = 0; p < d; ++p) {
      am += std::conj((*col)[static_cast<std::size_t>(p)]) * x_full[static_cast<std::size_t>(p)];
    }
    (*a)[m] = am;
  }
  for (std::size_t r = 0; r < k_out; ++r) {
    std::complex<double> sj(0.0, 0.0);
    for (std::size_t m = 0; m < k_in; ++m) {
      sj += std::conj(V[static_cast<std::size_t>(m) + r * k_in]) * (*a)[m];
    }
    y_block[r] = sj / std::sqrt(evals[r]);
  }
}

void MomentumPhiGramBasis::lift_full_from_block(const MomentumSectorMap& orbit_map, MomentumSector K,
                                               const FockBasis& fb, const std::complex<double>* y_block,
                                               std::complex<double>* x_full,
                                               MomentumPhiGramApplyScratch* scratch) const {
  const int d = d_full;
  std::fill(x_full, x_full + d, std::complex<double>(0.0, 0.0));
  if (k_out == 0) {
    return;
  }
  thread_local std::vector<std::complex<double>> tl_w;
  thread_local std::vector<std::complex<double>> tl_col;
  std::vector<std::complex<double>>* w = nullptr;
  std::vector<std::complex<double>>* col = nullptr;
  if (scratch != nullptr) {
    w = &scratch->w;
    col = &scratch->col;
  } else {
    w = &tl_w;
    col = &tl_col;
  }
  w->assign(k_in, std::complex<double>(0.0, 0.0));
  col->resize(static_cast<std::size_t>(d));

  for (std::size_t r = 0; r < k_out; ++r) {
    const double inv_sqrt = 1.0 / std::sqrt(evals[r]);
    for (std::size_t m = 0; m < k_in; ++m) {
      (*w)[m] += y_block[r] * V[static_cast<std::size_t>(m) + r * k_in] * inv_sqrt;
    }
  }
  for (std::size_t m = 0; m < k_in; ++m) {
    if ((*w)[m].real() == 0.0 && (*w)[m].imag() == 0.0) {
      continue;
    }
    detail::fill_phi_orbit_bloch_from_seed(orbit_map, K, lx, ly, fb, seeds[m], col->data());
    for (int p = 0; p < d; ++p) {
      x_full[p] += (*w)[m] * (*col)[static_cast<std::size_t>(p)];
    }
  }
}

std::size_t MomentumPhiGramBasis::storage_bytes() const noexcept {
  return seeds.size() * sizeof(RawState) + V.size() * sizeof(std::complex<double>) +
         evals.size() * sizeof(double);
}

void build_momentum_phi_orbit_orthonormal(const MomentumSectorMap& orbit_map, MomentumSector K, int lx, int ly,
                                          const FockBasis& fb, std::vector<std::complex<double>>* phi_column_major,
                                          std::size_t* dk_out) {
  const int d = fb.dim();
  const std::vector<RawState> seeds = momentum_phi_seeds(orbit_map, K);
  const std::size_t k_in = seeds.size();
  if (k_in == 0) {
    phi_column_major->clear();
    if (dk_out != nullptr) {
      *dk_out = 0;
    }
    return;
  }

  phi_column_major->assign(static_cast<std::size_t>(d) * k_in, std::complex<double>(0.0, 0.0));
  for (std::size_t j = 0; j < k_in; ++j) {
    detail::fill_phi_orbit_bloch_from_seed(orbit_map, K, lx, ly, fb, seeds[j],
                                           phi_column_major->data() + j * static_cast<std::size_t>(d));
  }

  // G_ij = <col_i | col_j> = sum_p conj(phi[p+i*d]) * phi[p+j*d]
  std::vector<std::complex<double>> g(static_cast<std::size_t>(k_in) * k_in, std::complex<double>(0.0, 0.0));
  for (std::size_t i = 0; i < k_in; ++i) {
    for (std::size_t j = 0; j < k_in; ++j) {
      std::complex<double> s(0.0, 0.0);
      for (int p = 0; p < d; ++p) {
        s += std::conj((*phi_column_major)[static_cast<std::size_t>(p) + i * static_cast<std::size_t>(d)]) *
             (*phi_column_major)[static_cast<std::size_t>(p) + j * static_cast<std::size_t>(d)];
      }
      g[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(k_in)] = s;
    }
  }
  // Hermitian symmetrize (roundoff can make zheev see indefinite G and drop all modes).
  for (std::size_t i = 0; i < k_in; ++i) {
    for (std::size_t j = i; j < k_in; ++j) {
      const std::complex<double> a = g[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * k_in];
      const std::complex<double> b = g[static_cast<std::size_t>(j) + static_cast<std::size_t>(i) * k_in];
      const std::complex<double> h = 0.5 * (a + std::conj(b));
      g[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * k_in] = h;
      g[static_cast<std::size_t>(j) + static_cast<std::size_t>(i) * k_in] = std::conj(h);
    }
  }

  std::vector<double> evals;
  zheev_full_hermitian_inplace(&g, static_cast<int>(k_in), &evals);
  double lam_max = 0.0;
  for (double ev : evals) {
    lam_max = std::max(lam_max, ev);
  }
  const double tol_ev = std::max(1e-14 * std::max(1.0, lam_max), 1e-20);

  // Φ_whiten = Φ @ V @ Λ^{-1/2}: T[p,j] = sum_m phi[p+m*d] * V[m,j], V column-major in g after zheev
  std::vector<std::complex<double>> tmp(static_cast<std::size_t>(d) * k_in, std::complex<double>(0.0, 0.0));
  for (int j = 0; j < static_cast<int>(k_in); ++j) {
    if (evals[static_cast<std::size_t>(j)] <= tol_ev) {
      continue;
    }
    const double inv_sqrt = 1.0 / std::sqrt(evals[static_cast<std::size_t>(j)]);
    for (int p = 0; p < d; ++p) {
      std::complex<double> s(0.0, 0.0);
      for (std::size_t m = 0; m < k_in; ++m) {
        const std::complex<double> vmj = g[static_cast<std::size_t>(m) + static_cast<std::size_t>(j) * k_in];
        s += (*phi_column_major)[static_cast<std::size_t>(p) + m * static_cast<std::size_t>(d)] * vmj;
      }
      tmp[static_cast<std::size_t>(p) + static_cast<std::size_t>(j) * static_cast<std::size_t>(d)] = s * inv_sqrt;
    }
  }

  // Eigenvectors in g are no longer needed; compacting only uses tmp + evals. Releasing g before resizing Φ
  // drops an O(k_in²) peak that would otherwise overlap the final |Φ| allocation during assign below.
  g.clear();
  g.shrink_to_fit();

  // Compact: keep only columns j with eval[j] > tol
  std::vector<std::size_t> keep;
  keep.reserve(k_in);
  for (std::size_t j = 0; j < k_in; ++j) {
    if (evals[j] > tol_ev) {
      keep.push_back(j);
    }
  }
  const std::size_t k_out = keep.size();
  phi_column_major->assign(static_cast<std::size_t>(d) * k_out, std::complex<double>(0.0, 0.0));
  for (std::size_t jc = 0; jc < k_out; ++jc) {
    const int j = static_cast<int>(keep[jc]);
    for (int p = 0; p < d; ++p) {
      (*phi_column_major)[static_cast<std::size_t>(p) + jc * static_cast<std::size_t>(d)] =
          tmp[static_cast<std::size_t>(p) + static_cast<std::size_t>(j) * static_cast<std::size_t>(d)];
    }
  }
  if (dk_out != nullptr) {
    *dk_out = k_out;
  }
}

}  // namespace symmetry
}  // namespace ftlm
