// Step 1 / 3: Compare dim_full, rank(P_K), k_in, dk_Gram, Gram eigenvalue structure; optional Im(P_K) basis check.
#include "ftlm/fock_basis.hpp"
#include "ftlm/hubbard_params.hpp"
#include "ftlm/symmetry/hubbard_momentum_action.hpp"
#include "ftlm/symmetry/k_block_dense_prototype.hpp"
#include "ftlm/symmetry/k_block_im_pk_basis.hpp"
#include "ftlm/symmetry/k_block_matrix_free.hpp"
#include "ftlm/symmetry/k_block_pk_tiny_dense.hpp"
#include "ftlm/symmetry/k_basis.hpp"
#include "ftlm/symmetry/momentum_sector.hpp"
#include "ftlm/symmetry/orbit_bloch_phi.hpp"
#include "ftlm/symmetry/translation_projector_dense.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <iomanip>
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

namespace {

void matrix_to_colmajor(const std::vector<std::vector<std::complex<double>>>& M,
                        std::vector<std::complex<double>>* out) {
  const int n = static_cast<int>(M.size());
  out->resize(static_cast<std::size_t>(n * n));
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      (*out)[static_cast<std::size_t>(i + j * n)] = M[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
    }
  }
}

void zheev_hermitian_inplace_vectors(int n, std::vector<std::complex<double>>* a_colmajor, std::vector<double>* w) {
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
    throw std::runtime_error("zheev workspace query failed (test_k_dim_analysis)");
  }
  lwork = static_cast<__LAPACK_int>(std::llround(work[0].real()));
  if (lwork < 1) {
    lwork = 1;
  }
  work.resize(static_cast<std::size_t>(lwork));
  zheev_(&jobz, &uplo, &nn, reinterpret_cast<__LAPACK_double_complex*>(a_colmajor->data()), &lda, w->data(),
         reinterpret_cast<__LAPACK_double_complex*>(work.data()), &lwork, rwork.data(), &info);
  if (info != 0) {
    throw std::runtime_error("zheev failed (test_k_dim_analysis)");
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
    throw std::runtime_error("zheev workspace query failed (test_k_dim_analysis)");
  }
  lwork = static_cast<int>(std::llround(work[0].real()));
  if (lwork < 1) {
    lwork = 1;
  }
  work.resize(static_cast<std::size_t>(lwork));
  zheev_(&jobz, &uplo, &nn, a_colmajor->data(), &lda, w->data(), work.data(), &lwork, rwork.data(), &info);
  if (info != 0) {
    throw std::runtime_error("zheev failed (test_k_dim_analysis)");
  }
#endif
}

void zheev_hermitian_inplace_evals_only(int n, std::vector<std::complex<double>>* a_colmajor, std::vector<double>* w) {
  if (n <= 0) {
    w->clear();
    return;
  }
#if defined(__APPLE__)
  char jobz = 'N';
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
  lwork = static_cast<__LAPACK_int>(std::llround(work[0].real()));
  if (lwork < 1) {
    lwork = 1;
  }
  work.resize(static_cast<std::size_t>(lwork));
  zheev_(&jobz, &uplo, &nn, reinterpret_cast<__LAPACK_double_complex*>(a_colmajor->data()), &lda, w->data(),
         reinterpret_cast<__LAPACK_double_complex*>(work.data()), &lwork, rwork.data(), &info);
  if (info != 0) {
    throw std::runtime_error("zheev evals failed (Gram)");
  }
#else
  char jobz = 'N';
  char uplo = 'U';
  int nn = n;
  int lda = n;
  w->resize(static_cast<std::size_t>(n));
  std::vector<std::complex<double>> work(1);
  int lwork = -1;
  std::vector<double> rwork(static_cast<std::size_t>(std::max(1, 3 * n - 2)));
  int info = 0;
  zheev_(&jobz, &uplo, &nn, a_colmajor->data(), &lda, w->data(), work.data(), &lwork, rwork.data(), &info);
  lwork = static_cast<int>(std::llround(work[0].real()));
  if (lwork < 1) {
    lwork = 1;
  }
  work.resize(static_cast<std::size_t>(lwork));
  zheev_(&jobz, &uplo, &nn, a_colmajor->data(), &lda, w->data(), work.data(), &lwork, rwork.data(), &info);
  if (info != 0) {
    throw std::runtime_error("zheev evals failed (Gram)");
  }
#endif
}

std::complex<double> dot_cols_conj_left(const std::complex<double>* a, const std::complex<double>* b, int d) {
  std::complex<double> s(0.0, 0.0);
  for (int p = 0; p < d; ++p) {
    s += std::conj(a[static_cast<std::size_t>(p)]) * b[static_cast<std::size_t>(p)];
  }
  return s;
}

/// Full Gram matrix eigenvalues (same construction as `MomentumPhiGramBasis::build`).
void gram_matrix_evals(const ftlm::symmetry::MomentumSectorMap& orbit_map, ftlm::symmetry::MomentumSector K, int lx, int ly,
                       const ftlm::FockBasis& fb, std::vector<double>* evals_out) {
  const std::vector<ftlm::symmetry::RawState> seeds = ftlm::symmetry::momentum_phi_seeds(orbit_map, K);
  const int d = fb.dim();
  const std::size_t k_in = seeds.size();
  if (k_in == 0 || d <= 0) {
    evals_out->clear();
    return;
  }
  std::vector<std::complex<double>> g(static_cast<std::size_t>(k_in) * k_in, std::complex<double>(0.0, 0.0));
  std::vector<std::complex<double>> col_i(static_cast<std::size_t>(d));
  std::vector<std::complex<double>> col_j(static_cast<std::size_t>(d));
  for (std::size_t i = 0; i < k_in; ++i) {
    ftlm::symmetry::detail::fill_phi_orbit_bloch_from_seed(orbit_map, K, lx, ly, fb, seeds[i], col_i.data());
    for (std::size_t j = i; j < k_in; ++j) {
      ftlm::symmetry::detail::fill_phi_orbit_bloch_from_seed(orbit_map, K, lx, ly, fb, seeds[j], col_j.data());
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
  zheev_hermitian_inplace_evals_only(static_cast<int>(k_in), &g, evals_out);
}

double max_abs_diff_sorted_eigs(int n, const std::vector<double>& a, const std::vector<double>& b) {
  if (static_cast<int>(a.size()) != n || static_cast<int>(b.size()) != n) {
    return 1e300;
  }
  std::vector<double> sa = a;
  std::vector<double> sb = b;
  std::sort(sa.begin(), sa.end());
  std::sort(sb.begin(), sb.end());
  double m = 0.0;
  for (int i = 0; i < n; ++i) {
    m = std::max(m, std::abs(sa[static_cast<std::size_t>(i)] - sb[static_cast<std::size_t>(i)]));
  }
  return m;
}

struct Case {
  int Lx;
  int Ly;
  int n_up;
  int n_dn;
  int kx;
  int ky;
};

void analyze(const ftlm::HubbardParams& p, const Case& c) {
  ftlm::HubbardParams pl = p;
  pl.Lx = c.Lx;
  pl.Ly = c.Ly;
  ftlm::FockBasis fb(c.Lx * c.Ly, c.n_up, c.n_dn);
  const int d_full = fb.dim();
  if (d_full <= 0) {
    return;
  }
  std::vector<ftlm::symmetry::RawState> universe;
  universe.reserve(static_cast<std::size_t>(d_full));
  for (int i = 0; i < d_full; ++i) {
    universe.push_back(ftlm::symmetry::RawState{static_cast<std::uint16_t>(fb.up_mask(i)),
                                                  static_cast<std::uint16_t>(fb.down_mask(i))});
  }
  const auto map = ftlm::symmetry::build_momentum_sector_map_rect(std::move(universe), c.Lx, c.Ly);
  const ftlm::symmetry::MomentumSector K{c.kx, c.ky, c.Lx, c.Ly};

  ftlm::symmetry::MomentumPhiGramBasis gram;
  if (!ftlm::symmetry::MomentumPhiGramBasis::build(map, K, c.Lx, c.Ly, fb, &gram)) {
    std::cout << "  sector (" << c.n_up << "," << c.n_dn << ") K=(" << c.kx << "," << c.ky << ")  empty Gram build\n";
    return;
  }
  const std::size_t k_in = gram.k_in;
  const std::size_t dk_gram = gram.k_out;

  std::vector<std::vector<std::complex<double>>> P;
  ftlm::symmetry::detail::build_translation_projector_dense(fb, c.Lx, c.Ly, K, &P);
  ftlm::symmetry::detail::hermitian_symmetrize_inplace(&P);
  std::vector<std::complex<double>> p_cm;
  matrix_to_colmajor(P, &p_cm);
  const double trace_P = ftlm::symmetry::tiny_pk::trace_hermitian_colmajor(d_full, p_cm.data());
  std::vector<double> w_P;
  zheev_hermitian_inplace_vectors(d_full, &p_cm, &w_P);
  int rank_P = 0;
  for (int j = 0; j < d_full; ++j) {
    if (w_P[static_cast<std::size_t>(j)] > 0.5) {
      ++rank_P;
    }
  }

  std::vector<double> g_evals;
  gram_matrix_evals(map, K, c.Lx, c.Ly, fb, &g_evals);
  double lam_max = 0.0;
  for (double ev : g_evals) {
    lam_max = std::max(lam_max, ev);
  }
  const double tol_ev = std::max(1e-14 * std::max(1.0, lam_max), 1e-20);
  int n_gram_above_tol = 0;
  int n_gram_strict_pos = 0;
  for (double ev : g_evals) {
    if (ev > tol_ev) {
      ++n_gram_above_tol;
    }
    if (ev > 1e-30 * std::max(1.0, lam_max)) {
      ++n_gram_strict_pos;
    }
  }

  std::cout << "  L=" << c.Lx << "x" << c.Ly << "  sector(" << c.n_up << "," << c.n_dn << ")  K=(" << c.kx << "," << c.ky
            << ")\n";
  std::cout << "    dim_full=" << d_full << "  k_in(seeds)=" << k_in << "  dk_Gram(k_out)=" << dk_gram << "\n";
  std::cout << "    rank(P_K) [eig>0.5]=" << rank_P << "  trace(P)=" << trace_P << "\n";
  std::cout << "    Gram: lam_max=" << lam_max << "  tol_ev=" << tol_ev << "  #eval>tol_ev=" << n_gram_above_tol
            << "  #eval>1e-30*lam_max=" << n_gram_strict_pos << "\n";

  if (static_cast<int>(dk_gram) != rank_P) {
    std::cout << "    NOTE: dk_Gram != rank(P_K). Mismatch sources: redundant seeds (k_in>rank), Gram tol drops small "
                 "positives, or numerical rank(G)<rank(P) (rare).\n";
  }
  if (static_cast<int>(k_in) < rank_P) {
    std::cout << "    ERROR: k_in < rank(P) — impossible if seeds span Im(P_K).\n";
  }

  // Prototype: orthonormal basis of Im(P_K) from spectral P (eigenvectors for eig>0.5), dimension rank_P.
  if (rank_P == static_cast<int>(dk_gram) && rank_P > 0 && d_full <= 64) {
    ftlm::symmetry::HubbardMomentumAction hub(pl);
    ftlm::symmetry::MomentumBlockScratch scratch;
    ftlm::symmetry::HubbardMomentumBlock block(hub, map, K, c.n_up, c.n_dn, &scratch);
    if (static_cast<int>(block.dim()) != rank_P) {
      std::cout << "    skip spectral match: block.dim=" << block.dim() << " vs rank_P=" << rank_P << "\n";
      return;
    }

    ftlm::symmetry::tiny_pk::TinyPkHpkDense pk =
        ftlm::symmetry::tiny_pk::build_tiny_pk_h_pk_dense(pl, fb, c.Lx, c.Ly, K, hub.hoppings, hub.nn_pairs, 64);

    matrix_to_colmajor(P, &p_cm);
    zheev_hermitian_inplace_vectors(d_full, &p_cm, &w_P);
    std::vector<std::complex<double>> U(static_cast<std::size_t>(d_full) * static_cast<std::size_t>(rank_P));
    int col = 0;
    for (int j = 0; j < d_full; ++j) {
      if (w_P[static_cast<std::size_t>(j)] <= 0.5) {
        continue;
      }
      for (int i = 0; i < d_full; ++i) {
        U[static_cast<std::size_t>(i + col * d_full)] = p_cm[static_cast<std::size_t>(i + j * d_full)];
      }
      ++col;
    }

    const int r = rank_P;
    std::vector<std::complex<double>> h_red(static_cast<std::size_t>(r * r), {0.0, 0.0});
    std::vector<std::complex<double>> hv(static_cast<std::size_t>(d_full));
    std::vector<std::complex<double>> col_j(static_cast<std::size_t>(d_full));
    for (int jj = 0; jj < r; ++jj) {
      for (int i = 0; i < d_full; ++i) {
        col_j[static_cast<std::size_t>(i)] = U[static_cast<std::size_t>(i + jj * d_full)];
      }
      for (int i = 0; i < d_full; ++i) {
        std::complex<double> s{0.0, 0.0};
        for (int k = 0; k < d_full; ++k) {
          s += pk.h_colmajor[static_cast<std::size_t>(i + k * d_full)] * col_j[static_cast<std::size_t>(k)];
        }
        hv[static_cast<std::size_t>(i)] = s;
      }
      for (int ii = 0; ii < r; ++ii) {
        std::complex<double> s{0.0, 0.0};
        for (int p = 0; p < d_full; ++p) {
          s += std::conj(U[static_cast<std::size_t>(p + ii * d_full)]) * hv[static_cast<std::size_t>(p)];
        }
        h_red[static_cast<std::size_t>(ii + jj * r)] = s;
      }
    }

    std::vector<std::complex<double>> h_red_copy = h_red;
    std::vector<double> w_red;
    zheev_hermitian_inplace_vectors(r, &h_red_copy, &w_red);

    ftlm::symmetry::k_block_matrix_free::DenseBlockHamiltonianFromApply h_prod(
        r, [&](const std::complex<double>* x, std::complex<double>* y) { block.apply(x, y); }, 64);
    const std::vector<std::complex<double>>& h_prod_dense = h_prod.dense_cols();
    std::vector<std::complex<double>> h_prod_copy = h_prod_dense;
    std::vector<double> w_prod;
    zheev_hermitian_inplace_vectors(r, &h_prod_copy, &w_prod);

    const double diff_eig = max_abs_diff_sorted_eigs(r, w_red, w_prod);
    std::cout << "    prototype Im(P_K) basis: r=" << r << "  max|eig(U^H H U)-eig(H_prod)|=" << diff_eig << "\n";
    if (diff_eig > 1e-8) {
      std::cerr << "test_k_dim_analysis: spectral mismatch too large\n";
      std::exit(2);
    }
  }
}

}  // namespace

int main() {
  std::cout << std::setprecision(17);
  ftlm::HubbardParams p{};
  p.t = 1.0;
  p.tp = -0.2;
  p.U = 4.0;
  p.V = 0.5;

  std::cout << "=== k-block dimensional analysis (dim_full vs rank(P_K) vs dk_Gram) ===\n";
  std::cout << "Production subspace: orthonormal columns after Gram whitening of raw orbit–Bloch seeds; "
               "dk_Gram = count of Gram eigenvalues > tol_ev (see orbit_bloch_phi.cpp).\n";
  std::cout << "Dense P_K: Hermitian projector after group sum + symmetrize; rank(P_K)=Tr(P) in exact arithmetic.\n\n";

  const std::vector<Case> cases_2x2 = {
      {2, 2, 1, 1, 0, 0}, {2, 2, 1, 1, 1, 0}, {2, 2, 1, 1, 1, 1}, {2, 2, 2, 2, 0, 0},
      {2, 2, 2, 2, 1, 0}, {2, 2, 0, 2, 0, 0},
  };
  p.Lx = 2;
  p.Ly = 2;
  for (const Case& c : cases_2x2) {
    analyze(p, c);
  }

  const std::vector<Case> cases_3x2 = {
      {3, 2, 1, 1, 0, 0},
      {3, 2, 1, 1, 1, 0},
  };
  p.Lx = 3;
  p.Ly = 2;
  for (const Case& c : cases_3x2) {
    analyze(p, c);
  }

  std::cout << "\ntest_k_dim_analysis ok\n";
  return 0;
}
