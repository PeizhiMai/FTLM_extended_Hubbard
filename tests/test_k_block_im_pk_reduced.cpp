// Compare TinyImPkReducedHamiltonian (U from dense P_K, H_red = U† H U) vs production HubbardMomentumBlock
// on tiny sectors where rank(P_K) = dk_Gram.
#include "ftlm/fock_basis.hpp"
#include "ftlm/hubbard_params.hpp"
#include "ftlm/symmetry/hubbard_momentum_action.hpp"
#include "ftlm/symmetry/k_block_dense_prototype.hpp"
#include "ftlm/symmetry/k_block_im_pk_reduced.hpp"
#include "ftlm/symmetry/momentum_sector.hpp"
#include "ftlm/symmetry/orbit_bloch_phi.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
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

void zheev_eigs_only(int n, std::vector<std::complex<double>>* a, std::vector<double>* w) {
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
  zheev_(&jobz, &uplo, &nn, reinterpret_cast<__LAPACK_double_complex*>(a->data()), &lda, w->data(),
         reinterpret_cast<__LAPACK_double_complex*>(work.data()), &lwork, rwork.data(), &info);
  lwork = static_cast<__LAPACK_int>(std::llround(work[0].real()));
  if (lwork < 1) {
    lwork = 1;
  }
  work.resize(static_cast<std::size_t>(lwork));
  zheev_(&jobz, &uplo, &nn, reinterpret_cast<__LAPACK_double_complex*>(a->data()), &lda, w->data(),
         reinterpret_cast<__LAPACK_double_complex*>(work.data()), &lwork, rwork.data(), &info);
  if (info != 0) {
    throw std::runtime_error("zheev evals failed");
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
  zheev_(&jobz, &uplo, &nn, a->data(), &lda, w->data(), work.data(), &lwork, rwork.data(), &info);
  lwork = static_cast<int>(std::llround(work[0].real()));
  if (lwork < 1) {
    lwork = 1;
  }
  work.resize(static_cast<std::size_t>(lwork));
  zheev_(&jobz, &uplo, &nn, a->data(), &lda, w->data(), work.data(), &lwork, rwork.data(), &info);
  if (info != 0) {
    throw std::runtime_error("zheev evals failed");
  }
#endif
}

double max_abs_diff_sorted_eigs(int n, const std::vector<double>& a, const std::vector<double>& b) {
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

double max_abs_mat(int n, const std::complex<double>* A, const std::complex<double>* B) {
  double m = 0.0;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      m = std::max(m, std::abs(A[static_cast<std::size_t>(i + j * n)] - B[static_cast<std::size_t>(i + j * n)]));
    }
  }
  return m;
}

struct Case {
  int Lx, Ly, n_up, n_dn, kx, ky;
};

void run_case(const ftlm::HubbardParams& p0, const Case& c) {
  ftlm::HubbardParams p = p0;
  p.Lx = c.Lx;
  p.Ly = c.Ly;
  ftlm::symmetry::HubbardMomentumAction hub(p);

  ftlm::FockBasis fb(c.Lx * c.Ly, c.n_up, c.n_dn);
  const int d_full = fb.dim();
  if (d_full <= 0 || d_full > 64) {
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
    return;
  }
  const int dk_gram = static_cast<int>(gram.k_out);
  if (dk_gram <= 0) {
    return;
  }

  ftlm::symmetry::TinyImPkReducedHamiltonian im{};
  if (!ftlm::symmetry::build_tiny_im_pk_reduced_hamiltonian(p, fb, c.Lx, c.Ly, K, hub.hoppings, hub.nn_pairs, 64, &im)) {
    std::cerr << "build_tiny_im_pk_reduced failed L=" << c.Lx << "x" << c.Ly << " sector(" << c.n_up << "," << c.n_dn
              << ") K=(" << c.kx << "," << c.ky << ")\n";
    std::exit(2);
  }

  if (im.r != dk_gram) {
    std::cerr << "dim mismatch: r=" << im.r << " dk_Gram=" << dk_gram << "  L=" << c.Lx << "x" << c.Ly << " sector("
              << c.n_up << "," << c.n_dn << ") K=(" << c.kx << "," << c.ky << ")\n";
    std::exit(2);
  }

  ftlm::symmetry::MomentumBlockScratch scratch;
  ftlm::symmetry::HubbardMomentumBlock block(hub, map, K, c.n_up, c.n_dn, &scratch);
  if (static_cast<int>(block.dim()) != im.r) {
    std::cerr << "block.dim != im.r\n";
    std::exit(2);
  }

  ftlm::symmetry::k_block_matrix_free::DenseBlockHamiltonianFromApply h_prod(
      im.r, [&](const std::complex<double>* x, std::complex<double>* y) { block.apply(x, y); }, 64);

  const std::vector<std::complex<double>>& h_prod_dense = h_prod.dense_cols();

  std::vector<std::complex<double>> h_im = im.h_red_colmajor;
  std::vector<double> w_im, w_prod;
  zheev_eigs_only(im.r, &h_im, &w_im);
  std::vector<std::complex<double>> h_prod_copy = h_prod_dense;
  zheev_eigs_only(im.r, &h_prod_copy, &w_prod);

  const double eig_diff = max_abs_diff_sorted_eigs(im.r, w_im, w_prod);
  const double tol_eig = 1e-8;
  if (eig_diff > tol_eig) {
    std::cerr << "sorted eigenvalue mismatch max=" << eig_diff << " case L=" << c.Lx << "x" << c.Ly << " (" << c.n_up
              << "," << c.n_dn << ") K=(" << c.kx << "," << c.ky << ")\n";
    std::exit(2);
  }

  const double mat_diff = max_abs_mat(im.r, im.h_red_colmajor.data(), h_prod_dense.data());

  std::cout << "ok L=" << c.Lx << "x" << c.Ly << " (" << c.n_up << "," << c.n_dn << ") K=(" << c.kx << "," << c.ky
            << ") r=dk=" << im.r << " max|eig_im-eig_prod|=" << eig_diff
            << " max|H_im-H_prod|_entry=" << mat_diff << " (entrywise diff expected: different ON bases of Im(P_K))\n";
}

}  // namespace

int main() {
  ftlm::HubbardParams p{};
  p.t = 1.0;
  p.tp = -0.2;
  p.U = 4.0;
  p.V = 0.5;

  const std::vector<Case> cases = {
      {2, 2, 1, 1, 0, 0}, {2, 2, 1, 1, 1, 0}, {2, 2, 1, 1, 1, 1}, {2, 2, 2, 2, 0, 0}, {2, 2, 2, 2, 1, 0},
      {3, 2, 1, 1, 0, 0}, {3, 2, 1, 1, 1, 0},
  };

  for (const Case& c : cases) {
    run_case(p, c);
  }

  std::cout << "test_k_block_im_pk_reduced ok\n";
  return 0;
}
