// Overlap matrix S = O^dagger O for raw orbit Bloch columns O (fill_phi_orbit_bloch_from_seed per OrbitKBlockBasis rep).
// Reports orthogonality / conditioning for Task 1 (Gram-free Lanczos feasibility).

#include "ftlm/fock_basis.hpp"
#include "ftlm/hubbard_params.hpp"
#include "ftlm/symmetry/hubbard_momentum_action.hpp"
#include "ftlm/symmetry/orbit_bloch_phi.hpp"
#include "ftlm/symmetry/orbit_k_block_basis.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

#if defined(__APPLE__)
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK 1
#endif
#include <Accelerate/Accelerate.h>
#else
extern "C" {
void zheev_(char* jobz, char* uplo, int* n, std::complex<double>* a, int* lda, double* w, std::complex<double>* work,
            int* lwork, double* rwork, int* info);
}
#endif

namespace {

using C = std::complex<double>;

void zheev_eigenvalues_hermitian_colmajor(int n, std::vector<C>* a_colmajor, std::vector<double>* evals) {
  evals->resize(static_cast<std::size_t>(n));
  std::vector<C> work(1);
  std::vector<double> rwork(static_cast<std::size_t>(std::max(1, 3 * n - 2)));
#if defined(__APPLE__)
  char jobz = 'N';
  char uplo = 'U';
  __LAPACK_int nn = static_cast<__LAPACK_int>(n);
  __LAPACK_int lda = nn;
  __LAPACK_int lwork_q = -1;
  __LAPACK_int info = 0;
  zheev_(&jobz, &uplo, &nn, reinterpret_cast<__LAPACK_double_complex*>(a_colmajor->data()), &lda, evals->data(),
         reinterpret_cast<__LAPACK_double_complex*>(work.data()), &lwork_q, rwork.data(), &info);
  if (info != 0) {
    throw std::runtime_error("zheev workspace query failed");
  }
  __LAPACK_int lwork = static_cast<__LAPACK_int>(std::llround(work[0].real()));
  if (lwork < 1) {
    lwork = 1;
  }
  work.resize(static_cast<std::size_t>(lwork));
  zheev_(&jobz, &uplo, &nn, reinterpret_cast<__LAPACK_double_complex*>(a_colmajor->data()), &lda, evals->data(),
         reinterpret_cast<__LAPACK_double_complex*>(work.data()), &lwork, rwork.data(), &info);
  if (info != 0) {
    throw std::runtime_error("zheev failed");
  }
#else
  char jobz = 'N';
  char uplo = 'U';
  int nn = n;
  int lda = n;
  int lwork_q = -1;
  int info = 0;
  zheev_(&jobz, &uplo, &nn, a_colmajor->data(), &lda, evals->data(), work.data(), &lwork_q, rwork.data(), &info);
  if (info != 0) {
    throw std::runtime_error("zheev workspace query failed");
  }
  int lwork = static_cast<int>(std::llround(work[0].real()));
  if (lwork < 1) {
    lwork = 1;
  }
  work.resize(static_cast<std::size_t>(lwork));
  zheev_(&jobz, &uplo, &nn, a_colmajor->data(), &lda, evals->data(), work.data(), &lwork, rwork.data(), &info);
  if (info != 0) {
    throw std::runtime_error("zheev failed");
  }
#endif
}

std::vector<C> build_O(const ftlm::FockBasis& fb, const ftlm::symmetry::MomentumSectorMap& map,
                       const ftlm::symmetry::OrbitKBlockBasis& basis) {
  const int d = fb.dim();
  const int dk = basis.dim();
  std::vector<C> O(static_cast<std::size_t>(d * dk), C(0.0, 0.0));
  for (int j = 0; j < dk; ++j) {
    const auto& e = basis.entries()[static_cast<std::size_t>(j)];
    ftlm::symmetry::detail::fill_phi_orbit_bloch_from_seed(map, basis.K, basis.lx, basis.ly, fb, e.rep,
                                                          O.data() + static_cast<std::size_t>(j * d));
  }
  return O;
}

struct OverlapStats {
  double max_offdiag_abs = 0.0;
  double diag_min = 0.0;
  double diag_max = 0.0;
  double frob_off = 0.0;
  double frob_all = 0.0;
  double lambda_min_pos = 0.0;
  double lambda_max = 0.0;
  double cond = 0.0;
};

OverlapStats analyze(const std::vector<C>& O, int d, int dk) {
  OverlapStats st;
  std::vector<C> S(static_cast<std::size_t>(dk * dk), C(0.0, 0.0));
  for (int i = 0; i < dk; ++i) {
    for (int j = 0; j < dk; ++j) {
      C acc(0.0, 0.0);
      for (int p = 0; p < d; ++p) {
        acc += std::conj(O[static_cast<std::size_t>(p + i * d)]) * O[static_cast<std::size_t>(p + j * d)];
      }
      S[static_cast<std::size_t>(i + j * dk)] = acc;
    }
  }

  st.diag_min = 1e300;
  st.diag_max = -1e300;
  for (int i = 0; i < dk; ++i) {
    const double di = S[static_cast<std::size_t>(i + i * dk)].real();
    st.diag_min = std::min(st.diag_min, di);
    st.diag_max = std::max(st.diag_max, di);
  }
  for (int i = 0; i < dk; ++i) {
    for (int j = 0; j < dk; ++j) {
      const double nrm = std::norm(S[static_cast<std::size_t>(i + j * dk)]);
      st.frob_all += nrm;
      if (i != j) {
        st.frob_off += nrm;
        st.max_offdiag_abs = std::max(st.max_offdiag_abs, std::abs(S[static_cast<std::size_t>(i + j * dk)]));
      }
    }
  }
  st.frob_all = std::sqrt(st.frob_all);
  st.frob_off = std::sqrt(st.frob_off);

  std::vector<C> Shem = S;
  std::vector<double> evals;
  zheev_eigenvalues_hermitian_colmajor(dk, &Shem, &evals);
  double lam_pos_min = 1e300;
  st.lambda_max = 0.0;
  for (double ev : evals) {
    if (ev > st.lambda_max) {
      st.lambda_max = ev;
    }
    if (ev > 1e-20) {
      lam_pos_min = std::min(lam_pos_min, ev);
    }
  }
  st.lambda_min_pos = lam_pos_min;
  st.cond = (lam_pos_min < 1e300 && lam_pos_min > 0.0) ? (st.lambda_max / lam_pos_min) : -1.0;
  return st;
}

void run_case(const ftlm::HubbardParams& p, int nu, int nd, int kx, int ky) {
  ftlm::symmetry::HubbardMomentumAction hub(p);
  const int n_sites = p.Lx * p.Ly;
  ftlm::FockBasis fb(n_sites, nu, nd);
  std::vector<ftlm::symmetry::RawState> universe;
  universe.reserve(static_cast<std::size_t>(fb.dim()));
  for (int i = 0; i < fb.dim(); ++i) {
    universe.push_back(ftlm::symmetry::RawState{static_cast<std::uint16_t>(fb.up_mask(i)),
                                                  static_cast<std::uint16_t>(fb.down_mask(i))});
  }
  const auto map = ftlm::symmetry::build_momentum_sector_map_rect(std::move(universe), p.Lx, p.Ly);
  const ftlm::symmetry::MomentumSector K{kx, ky, p.Lx, p.Ly};
  ftlm::symmetry::OrbitKBlockBasis basis;
  basis.build(fb, K);
  const int dk = basis.dim();
  if (dk <= 0) {
    std::cout << "(" << nu << "," << nd << ") K=(" << kx << "," << ky << ")  dk=0  skip\n";
    return;
  }
  const int d = fb.dim();
  const std::vector<C> O = build_O(fb, map, basis);
  const OverlapStats st = analyze(O, d, dk);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "(" << nu << "," << nd << ") K=(" << kx << "," << ky << ")  d_full=" << d << "  dk=" << dk << "\n";
  std::cout << "  max_{i!=j}|S_ij|=" << st.max_offdiag_abs << "  diag[min,max]=[" << st.diag_min << "," << st.diag_max
            << "]\n";
  std::cout << "  ||S_off||_F=" << st.frob_off << "  ||S||_F=" << st.frob_all
            << "  ratio_off=" << (st.frob_all > 0 ? st.frob_off / st.frob_all : 0.0) << "\n";
  std::cout << "  lambda_max(S)=" << st.lambda_max << "  lambda_min_pos(S)=" << st.lambda_min_pos
            << "  cond(S)=" << (st.cond > 0 ? st.cond : -1.0) << "\n\n";
}

}  // namespace

int main() {
  ftlm::HubbardParams p{};
  p.Lx = 3;
  p.Ly = 2;
  p.t = 1.0;
  p.tp = -0.35;
  p.U = 5.75;
  p.V = 0.9;

  std::cout << "=== diag_orbit_overlap_s  (S = O^dagger O, raw orbit Bloch columns) ===\n";
  std::cout << "3x2 default hubbard params\n\n";

  std::cout << "--- (3,3) all K ---\n";
  for (int ky = 0; ky < p.Ly; ++ky) {
    for (int kx = 0; kx < p.Lx; ++kx) {
      run_case(p, 3, 3, kx, ky);
    }
  }

  std::cout << "--- (2,3) K=(1,0) ---\n";
  run_case(p, 2, 3, 1, 0);

  std::cout << "--- (2,4) K=(0,1) (dim-mismatch example) ---\n";
  run_case(p, 2, 4, 0, 1);

  std::cout << "--- (1,1) K=(1,0) small sector ---\n";
  run_case(p, 1, 1, 1, 0);

  return 0;
}
