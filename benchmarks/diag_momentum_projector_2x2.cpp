// Diagnostics for dense translation projector P_k vs KBasis vs orbit-Bloch Phi on 2x2 failing sectors.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "ftlm/fock_basis.hpp"
#include "ftlm/symmetry/k_basis.hpp"
#include "ftlm/symmetry/momentum_sector.hpp"
#include "ftlm/symmetry/raw_state.hpp"
#include "ftlm/symmetry/orbit_bloch_phi.hpp"
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

namespace {

constexpr int Lx = 2;
constexpr int Ly = 2;
constexpr int Ns = 4;

void hermitian_symmetrize_inplace(std::vector<std::vector<std::complex<double>>>* M) {
  const int n = static_cast<int>(M->size());
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      const std::complex<double> a = (*M)[static_cast<size_t>(i)][static_cast<size_t>(j)];
      const std::complex<double> b = (*M)[static_cast<size_t>(j)][static_cast<size_t>(i)];
      const std::complex<double> h = 0.5 * (a + std::conj(b));
      (*M)[static_cast<size_t>(i)][static_cast<size_t>(j)] = h;
      (*M)[static_cast<size_t>(j)][static_cast<size_t>(i)] = std::conj(h);
    }
  }
}

double max_herm_skew(const std::vector<std::vector<std::complex<double>>>& M) {
  const int n = static_cast<int>(M.size());
  double m = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      const std::complex<double> d = M[static_cast<size_t>(i)][static_cast<size_t>(j)] -
                                      std::conj(M[static_cast<size_t>(j)][static_cast<size_t>(i)]);
      m = std::max(m, std::abs(d));
    }
  }
  return m;
}

std::vector<double> zheev_evals_only(const std::vector<std::vector<std::complex<double>>>& H) {
  const int n = static_cast<int>(H.size());
  if (n == 0) {
    return {};
  }
#if defined(__APPLE__)
  std::vector<std::complex<double>> a(static_cast<size_t>(n) * static_cast<size_t>(n));
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      a[static_cast<size_t>(j * n + i)] = H[static_cast<size_t>(i)][static_cast<size_t>(j)];
    }
  }
  char jobz = 'N';
  char uplo = 'U';
  __LAPACK_int nn = static_cast<__LAPACK_int>(n);
  __LAPACK_int lda = static_cast<__LAPACK_int>(n);
  std::vector<double> w(static_cast<size_t>(n));
  std::vector<std::complex<double>> work(1);
  __LAPACK_int lwork = -1;
  std::vector<double> rwork(static_cast<size_t>(std::max(1, 3 * n - 2)));
  __LAPACK_int info = 0;
  zheev_(&jobz, &uplo, &nn, reinterpret_cast<__LAPACK_double_complex*>(a.data()), &lda, w.data(),
         reinterpret_cast<__LAPACK_double_complex*>(work.data()), &lwork, rwork.data(), &info);
  if (info != 0) {
    throw std::runtime_error("zheev workspace query failed");
  }
  lwork = static_cast<__LAPACK_int>(std::llround(work[0].real()));
  if (lwork < 1) {
    lwork = 1;
  }
  work.resize(static_cast<size_t>(lwork));
  zheev_(&jobz, &uplo, &nn, reinterpret_cast<__LAPACK_double_complex*>(a.data()), &lda, w.data(),
         reinterpret_cast<__LAPACK_double_complex*>(work.data()), &lwork, rwork.data(), &info);
  if (info != 0) {
    throw std::runtime_error("zheev failed");
  }
  return w;
#else
  std::vector<std::complex<double>> a(static_cast<size_t>(n) * static_cast<size_t>(n));
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      a[static_cast<size_t>(j * n + i)] = H[static_cast<size_t>(i)][static_cast<size_t>(j)];
    }
  }
  char jobz = 'N';
  char uplo = 'U';
  int nn = n;
  int lda = n;
  std::vector<double> w(static_cast<size_t>(n));
  std::vector<std::complex<double>> work(1);
  int lwork = -1;
  std::vector<double> rwork(static_cast<size_t>(std::max(1, 3 * n - 2)));
  int info = 0;
  zheev_(&jobz, &uplo, &nn, a.data(), &lda, w.data(), work.data(), &lwork, rwork.data(), &info);
  lwork = static_cast<int>(std::llround(work[0].real()));
  if (lwork < 1) {
    lwork = 1;
  }
  work.resize(static_cast<size_t>(lwork));
  zheev_(&jobz, &uplo, &nn, a.data(), &lda, w.data(), work.data(), &lwork, rwork.data(), &info);
  return w;
#endif
}

double trace_c(const std::vector<std::vector<std::complex<double>>>& M) {
  double t = 0.0;
  for (size_t i = 0; i < M.size(); ++i) {
    t += M[i][i].real();
  }
  return t;
}

}  // namespace

int main() {
  std::cout << std::setprecision(17);
  const std::vector<std::pair<int, int>> sectors = {{0, 2}, {2, 0}, {4, 2}};

  for (const auto& pr : sectors) {
    const int nup = pr.first;
    const int ndn = pr.second;
    ftlm::FockBasis fb(Ns, nup, ndn);
    const int d = fb.dim();
    std::vector<ftlm::symmetry::RawState> universe;
    universe.reserve(static_cast<size_t>(d));
    for (int i = 0; i < d; ++i) {
      universe.push_back(ftlm::symmetry::RawState{static_cast<std::uint16_t>(fb.up_mask(i)),
                                                    static_cast<std::uint16_t>(fb.down_mask(i))});
    }
    const auto map = ftlm::symmetry::build_momentum_sector_map_rect(std::move(universe), Lx, Ly);

    std::cout << "\n========== sector (" << nup << "," << ndn << ") dim=" << d << " ==========\n";
    std::cout << "orbits=" << map.orbits.size() << "\n";

    for (std::size_t oi = 0; oi < map.orbits.size(); ++oi) {
      const auto& o = map.orbits[oi];
      std::cout << "  orbit " << oi << " rep up=0x" << std::hex << o.representative.up << " dn=0x" << o.representative.dn
                << std::dec << " size=" << o.orbit_size << " stab_sz=" << o.stabilizer.size()
                << " compat_mask=0x" << std::hex << o.compatible_momentum_mask << std::dec << "\n";
    }

    for (int ky = 0; ky < Ly; ++ky) {
      for (int kx = 0; kx < Lx; ++kx) {
        ftlm::symmetry::MomentumSector K{kx, ky, Lx, Ly};
        const auto kb = ftlm::symmetry::KBasis::build(map, K);
        const int dk = static_cast<int>(kb.dim());

        std::vector<std::vector<std::complex<double>>> P;
        ftlm::symmetry::detail::build_translation_projector_dense(fb, Lx, Ly, K, &P);

        const double skew = max_herm_skew(P);
        const double tr = trace_c(P);

        std::vector<std::vector<std::complex<double>>> Ps = P;
        hermitian_symmetrize_inplace(&Ps);
        const std::vector<double> evals = zheev_evals_only(Ps);
        std::vector<double> ev_sorted = evals;
        std::sort(ev_sorted.begin(), ev_sorted.end());

        int cnt_05 = 0;
        int cnt_1e8 = 0;
        int cnt_1e3 = 0;
        double emax = 0.0;
        for (double e : evals) {
          emax = std::max(emax, e);
          if (e > 0.5) {
            ++cnt_05;
          }
          if (e > 1e-8) {
            ++cnt_1e8;
          }
          if (e > 1e-3) {
            ++cnt_1e3;
          }
        }

        std::cout << "K=(" << kx << "," << ky << ") dk_KBasis(reps)=" << dk
                  << " seeds=" << ftlm::symmetry::momentum_phi_seeds(map, K).size() << " max_skew(P,H)=" << skew
                  << " trace(P)=" << tr << " eval_max=" << emax << " rank_cnt(lambda>0.5)=" << cnt_05
                  << " rank_cnt(>1e-8)=" << cnt_1e8 << " rank_cnt(>1e-3)=" << cnt_1e3 << "\n";
        std::cout << "  evals ascending:";
        for (double e : ev_sorted) {
          std::cout << " " << e;
        }
        std::cout << "\n";

        const std::vector<ftlm::symmetry::RawState> seeds = ftlm::symmetry::momentum_phi_seeds(map, K);
        const int ns = static_cast<int>(seeds.size());
        if (ns > 0) {
          std::vector<std::complex<double>> phi_flat(static_cast<size_t>(d) * static_cast<size_t>(ns), 0.0);
          for (int j = 0; j < ns; ++j) {
            ftlm::symmetry::detail::fill_phi_orbit_bloch_from_seed(map, K, Lx, Ly, fb, seeds[static_cast<size_t>(j)],
                                                                   phi_flat.data() + static_cast<size_t>(j) * static_cast<size_t>(d));
          }
          std::vector<std::vector<std::complex<double>>> G(static_cast<size_t>(ns),
                                                            std::vector<std::complex<double>>(static_cast<size_t>(ns), 0.0));
          for (int a = 0; a < ns; ++a) {
            for (int b = 0; b < ns; ++b) {
              std::complex<double> z(0.0, 0.0);
              for (int p = 0; p < d; ++p) {
                const std::complex<double> pa = phi_flat[static_cast<size_t>(p) + static_cast<size_t>(a) * static_cast<size_t>(d)];
                const std::complex<double> pb = phi_flat[static_cast<size_t>(p) + static_cast<size_t>(b) * static_cast<size_t>(d)];
                z += std::conj(pa) * pb;
              }
              G[static_cast<size_t>(a)][static_cast<size_t>(b)] = z;
            }
          }
          std::vector<std::vector<std::complex<double>>> Gh = G;
          hermitian_symmetrize_inplace(&Gh);
          const auto gev = zheev_evals_only(Gh);
          double gmin = gev.empty() ? 0.0 : gev[0];
          double gmax = 0.0;
          for (double x : gev) {
            gmax = std::max(gmax, x);
          }
          std::cout << "  orbit-Bloch Gram (seeds) eigenvalues min=" << gmin << " max=" << gmax << " (expect 1 if ONB)\n";
        }
      }
    }
  }
  return 0;
}
