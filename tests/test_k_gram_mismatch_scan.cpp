// Diagnostic: production Gram k_out vs rank(P_K) on small rectangles (dim_full <= 64 for dense P).
#include "ftlm/fock_basis.hpp"
#include "ftlm/symmetry/k_basis.hpp"
#include "ftlm/symmetry/momentum_sector.hpp"
#include "ftlm/symmetry/orbit_bloch_phi.hpp"
#include "ftlm/symmetry/raw_state.hpp"
#include "ftlm/symmetry/translation_projector_dense.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <iomanip>
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

void zheev_hermitian_vectors(int n, std::vector<std::complex<double>>* a_colmajor, std::vector<double>* w) {
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
  lwork = static_cast<__LAPACK_int>(std::llround(work[0].real()));
  if (lwork < 1) {
    lwork = 1;
  }
  work.resize(static_cast<std::size_t>(lwork));
  zheev_(&jobz, &uplo, &nn, reinterpret_cast<__LAPACK_double_complex*>(a_colmajor->data()), &lda, w->data(),
         reinterpret_cast<__LAPACK_double_complex*>(work.data()), &lwork, rwork.data(), &info);
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
  lwork = static_cast<int>(std::llround(work[0].real()));
  if (lwork < 1) {
    lwork = 1;
  }
  work.resize(static_cast<std::size_t>(lwork));
  zheev_(&jobz, &uplo, &nn, a_colmajor->data(), &lda, w->data(), work.data(), &lwork, rwork.data(), &info);
#endif
}

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

void scan_lattice(int Lx, int Ly) {
  const int n_sites = Lx * Ly;
  std::cout << "\n--- L=" << Lx << "x" << Ly << " (dim_full<=64 only for dense P) ---\n";
  for (int nu = 0; nu <= n_sites; ++nu) {
    for (int nd = 0; nd <= n_sites; ++nd) {
      ftlm::FockBasis fb(n_sites, nu, nd);
      const int d_full = fb.dim();
      if (d_full <= 0 || d_full > 64) {
        continue;
      }
      std::vector<ftlm::symmetry::RawState> universe;
      universe.reserve(static_cast<std::size_t>(d_full));
      for (int i = 0; i < d_full; ++i) {
        universe.push_back(ftlm::symmetry::RawState{static_cast<std::uint16_t>(fb.up_mask(i)),
                                                      static_cast<std::uint16_t>(fb.down_mask(i))});
      }
      const auto map = ftlm::symmetry::build_momentum_sector_map_rect(std::move(universe), Lx, Ly);
      for (int ky = 0; ky < Ly; ++ky) {
        for (int kx = 0; kx < Lx; ++kx) {
          const ftlm::symmetry::MomentumSector K{kx, ky, Lx, Ly};
          const std::size_t k_in = ftlm::symmetry::momentum_phi_seeds(map, K).size();

          ftlm::symmetry::GramKOutDiagnostics gram{};
          const bool gram_ok = ftlm::symmetry::momentum_phi_gram_k_out_only(map, K, Lx, Ly, fb, &gram, nullptr, nullptr);
          const std::size_t k_out = gram_ok ? gram.k_out : 0;
          const double lam_max_p = gram_ok ? gram.lam_max : 0.0;
          const double tol_ev_p = gram_ok ? gram.tol_ev : 0.0;

          std::vector<std::vector<std::complex<double>>> P;
          ftlm::symmetry::detail::build_translation_projector_dense(fb, Lx, Ly, K, &P);
          ftlm::symmetry::detail::hermitian_symmetrize_inplace(&P);
          std::vector<std::complex<double>> p_cm;
          matrix_to_colmajor(P, &p_cm);
          std::vector<double> w_P;
          zheev_hermitian_vectors(d_full, &p_cm, &w_P);
          int rank_P = 0;
          for (int j = 0; j < d_full; ++j) {
            if (w_P[static_cast<std::size_t>(j)] > 0.5) {
              ++rank_P;
            }
          }

          const bool mismatch_a = (rank_P > 0 && k_out == 0);
          const bool mismatch_b = (rank_P != static_cast<int>(k_out)) && (k_out > 0 || rank_P > 0);
          if (mismatch_a || mismatch_b) {
            std::cout << std::setprecision(17);
            std::cout << "  sector(" << nu << "," << nd << ") K=(" << kx << "," << ky << ") dim_full=" << d_full
                      << " k_in=" << k_in << " k_out=" << k_out << " rank(P)=" << rank_P
                      << " lam_max=" << lam_max_p << " tol_ev=" << tol_ev_p;
            if (mismatch_a) {
              std::cout << "  [k_out=0 but rank(P)>0: Gram numerically singular vs dense P]";
            }
            if (mismatch_b && !mismatch_a) {
              std::cout << "  [k_out!=rank(P)]";
            }
            std::cout << "\n";
          }
        }
      }
    }
  }
}

}  // namespace

int main() {
  std::cout << "Gram k_out (momentum_phi_gram_k_out_only) vs rank(P_K) [eig>0.5] on dense symmetrized P_K.\n";
  std::cout << "Production keep/skip uses k_out; mismatch means simple rank(P) gating would differ.\n";
  scan_lattice(2, 2);
  scan_lattice(3, 2);
  std::cout << "\ntest_k_gram_mismatch_scan ok\n";
  return 0;
}
