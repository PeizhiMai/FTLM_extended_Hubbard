// Debug: compare Φ from dense translation projector + spectral cut (Φ_old) vs orbit Bloch + orthonormalization (Φ_new).
// Default: 2x2, sector (N_up,N_dn)=(2,2), K=(0,0).
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "ftlm/fock_basis.hpp"
#include "ftlm/hubbard_hamiltonian.hpp"
#include "ftlm/hubbard_params.hpp"
#include "ftlm/lattice.hpp"
#include "ftlm/symmetry/k_basis.hpp"
#include "ftlm/symmetry/momentum_sector.hpp"
#include "ftlm/symmetry/orbit_bloch_phi.hpp"
#include "ftlm/symmetry/raw_state.hpp"
#include "ftlm/symmetry/translation_projector_dense.hpp"

namespace {

std::vector<std::complex<double>> apply_dense_P(const std::vector<std::vector<std::complex<double>>>& P,
                                                const std::vector<std::complex<double>>& v) {
  const int d = static_cast<int>(v.size());
  std::vector<std::complex<double>> out(static_cast<std::size_t>(d), 0.0);
  for (int i = 0; i < d; ++i) {
    std::complex<double> s(0.0, 0.0);
    for (int j = 0; j < d; ++j) {
      s += P[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] * v[static_cast<std::size_t>(j)];
    }
    out[static_cast<std::size_t>(i)] = s;
  }
  return out;
}

}  // namespace

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

void zheev_full_hermitian(const std::vector<std::vector<std::complex<double>>>& H,
                          std::vector<double>* evals,
                          std::vector<std::vector<std::complex<double>>>* evecs_columns) {
  const int n = static_cast<int>(H.size());
  if (n == 0) {
    evals->clear();
    evecs_columns->clear();
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
    throw std::runtime_error("zheev workspace query failed");
  }
  lwork = static_cast<__LAPACK_int>(std::llround(work[0].real()));
  if (lwork < 1) {
    lwork = 1;
  }
  work.resize(static_cast<std::size_t>(lwork));
  zheev_(&jobz, &uplo, &nn, reinterpret_cast<__LAPACK_double_complex*>(a.data()), &lda, w.data(),
         reinterpret_cast<__LAPACK_double_complex*>(work.data()), &lwork, rwork.data(), &info);
  if (info != 0) {
    throw std::runtime_error("zheev failed");
  }
  evals->assign(w.begin(), w.end());
  evecs_columns->assign(static_cast<std::size_t>(n), std::vector<std::complex<double>>(static_cast<std::size_t>(n), 0.0));
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      (*evecs_columns)[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
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
    throw std::runtime_error("zheev workspace query failed");
  }
  lwork = static_cast<int>(std::llround(work[0].real()));
  if (lwork < 1) {
    lwork = 1;
  }
  work.resize(static_cast<std::size_t>(lwork));
  zheev_(&jobz, &uplo, &nn, a.data(), &lda, w.data(), work.data(), &lwork, rwork.data(), &info);
  if (info != 0) {
    throw std::runtime_error("zheev failed");
  }
  evals->assign(w.begin(), w.end());
  evecs_columns->assign(static_cast<std::size_t>(n), std::vector<std::complex<double>>(static_cast<std::size_t>(n), 0.0));
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      (*evecs_columns)[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
          a[static_cast<std::size_t>(j * n + i)];
    }
  }
#endif
}

/// Same spectral extraction as legacy `build_phi_from_translation_projector_dense`.
void build_phi_old_dense_projector(const ftlm::FockBasis& fb, int lx, int ly, ftlm::symmetry::MomentumSector K,
                                   std::vector<std::complex<double>>* phi, std::size_t* dk_out) {
  std::vector<std::vector<std::complex<double>>> P;
  ftlm::symmetry::detail::build_translation_projector_dense(fb, lx, ly, K, &P);
  ftlm::symmetry::detail::hermitian_symmetrize_inplace(&P);
  const int d = fb.dim();
  std::vector<double> evals;
  std::vector<std::vector<std::complex<double>>> evecs;
  zheev_full_hermitian(P, &evals, &evecs);
  double eval_max = 0.0;
  for (int j = 0; j < d; ++j) {
    eval_max = std::max(eval_max, evals[static_cast<std::size_t>(j)]);
  }
  const double tol = std::max(1e-10, 1e-12 * std::max(1.0, eval_max));
  std::vector<std::pair<double, int>> scored;
  scored.reserve(static_cast<std::size_t>(d));
  for (int j = 0; j < d; ++j) {
    const double ev = evals[static_cast<std::size_t>(j)];
    if (ev > tol) {
      scored.push_back({-ev, j});
    }
  }
  std::sort(scored.begin(), scored.end());
  const std::size_t dk = scored.size();
  if (dk_out != nullptr) {
    *dk_out = dk;
  }
  phi->assign(static_cast<std::size_t>(d) * dk, std::complex<double>(0.0, 0.0));
  for (std::size_t jc = 0; jc < dk; ++jc) {
    const int col_src = scored[static_cast<std::size_t>(jc)].second;
    for (int p = 0; p < d; ++p) {
      (*phi)[static_cast<std::size_t>(p) + jc * static_cast<std::size_t>(d)] =
          evecs[static_cast<std::size_t>(p)][static_cast<std::size_t>(col_src)];
    }
  }
}

std::vector<std::vector<std::complex<double>>> gram_matrix(const std::vector<std::complex<double>>& phi_cm, int d,
                                                             std::size_t dk) {
  std::vector<std::vector<std::complex<double>>> G(
      dk, std::vector<std::complex<double>>(dk, std::complex<double>(0.0, 0.0)));
  for (std::size_t a = 0; a < dk; ++a) {
    for (std::size_t b = 0; b < dk; ++b) {
      std::complex<double> s(0.0, 0.0);
      for (int p = 0; p < d; ++p) {
        s += std::conj(phi_cm[static_cast<std::size_t>(p) + a * static_cast<std::size_t>(d)]) *
             phi_cm[static_cast<std::size_t>(p) + b * static_cast<std::size_t>(d)];
      }
      G[a][b] = s;
    }
  }
  return G;
}

std::vector<std::vector<std::complex<double>>> projector_Q(const std::vector<std::complex<double>>& phi_cm, int d,
                                                         std::size_t dk) {
  std::vector<std::vector<std::complex<double>>> Q(static_cast<std::size_t>(d),
                                                     std::vector<std::complex<double>>(static_cast<std::size_t>(d), 0.0));
  for (int p = 0; p < d; ++p) {
    for (int q = 0; q < d; ++q) {
      std::complex<double> s(0.0, 0.0);
      for (std::size_t j = 0; j < dk; ++j) {
        s += phi_cm[static_cast<std::size_t>(p) + j * static_cast<std::size_t>(d)] *
             std::conj(phi_cm[static_cast<std::size_t>(q) + j * static_cast<std::size_t>(d)]);
      }
      Q[static_cast<std::size_t>(p)][static_cast<std::size_t>(q)] = s;
    }
  }
  return Q;
}

double max_abs_mat(const std::vector<std::vector<std::complex<double>>>& A) {
  double m = 0.0;
  for (const auto& row : A) {
    for (const auto& z : row) {
      m = std::max(m, std::abs(z));
    }
  }
  return m;
}

double max_abs_diff_mat(const std::vector<std::vector<std::complex<double>>>& A,
                        const std::vector<std::vector<std::complex<double>>>& B) {
  double m = 0.0;
  const int n = static_cast<int>(A.size());
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      m = std::max(m, std::abs(A[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] -
                               B[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]));
    }
  }
  return m;
}

/// SVD singular values of A = Phi_old^H Phi_new, shapes (dk_old × d) * (d × dk_new) = dk_old × dk_new
std::vector<double> singular_values_overlap(const std::vector<std::complex<double>>& phi_old, int d, std::size_t dk_old,
                                            const std::vector<std::complex<double>>& phi_new, std::size_t dk_new) {
  const int m = static_cast<int>(dk_old);
  const int n = static_cast<int>(dk_new);
  std::vector<std::complex<double>> A(static_cast<std::size_t>(m) * static_cast<std::size_t>(n), 0.0);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      std::complex<double> s(0.0, 0.0);
      for (int p = 0; p < d; ++p) {
        s += std::conj(phi_old[static_cast<std::size_t>(p) + static_cast<std::size_t>(i) * static_cast<std::size_t>(d)]) *
             phi_new[static_cast<std::size_t>(p) + static_cast<std::size_t>(j) * static_cast<std::size_t>(d)];
      }
      A[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(m)] = s;
    }
  }
  // |A|^2 singular values = eigenvalues of A^H A (n×n) or A A^H (m×m) — use smaller
  const int k = std::min(m, n);
  if (k <= 0) {
    return {};
  }
  const bool use_aah = m <= n;
  const int nn = use_aah ? m : n;
  std::vector<std::vector<std::complex<double>>> M(static_cast<std::size_t>(nn),
                                                   std::vector<std::complex<double>>(static_cast<std::size_t>(nn), 0.0));
  if (use_aah) {
    for (int i = 0; i < m; ++i) {
      for (int ip = 0; ip < m; ++ip) {
        std::complex<double> s(0.0, 0.0);
        for (int j = 0; j < n; ++j) {
          const auto aij = A[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(m)];
          const auto aipj = A[static_cast<std::size_t>(ip) + static_cast<std::size_t>(j) * static_cast<std::size_t>(m)];
          s += aij * std::conj(aipj);
        }
        M[static_cast<std::size_t>(i)][static_cast<std::size_t>(ip)] = s;
      }
    }
  } else {
    for (int j = 0; j < n; ++j) {
      for (int jp = 0; jp < n; ++jp) {
        std::complex<double> s(0.0, 0.0);
        for (int i = 0; i < m; ++i) {
          const auto aij = A[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(m)];
          const auto aijp = A[static_cast<std::size_t>(i) + static_cast<std::size_t>(jp) * static_cast<std::size_t>(m)];
          s += std::conj(aij) * aijp;
        }
        M[static_cast<std::size_t>(j)][static_cast<std::size_t>(jp)] = s;
      }
    }
  }
  std::vector<double> evals;
  std::vector<std::vector<std::complex<double>>> evecs_dummy;
  zheev_full_hermitian(M, &evals, &evecs_dummy);
  std::vector<double> svals;
  svals.reserve(static_cast<std::size_t>(k));
  for (double ev : evals) {
    if (ev > 0.0) {
      svals.push_back(std::sqrt(ev));
    }
  }
  std::sort(svals.begin(), svals.end());
  return svals;
}

void print_col_norms(const std::string& name, const std::vector<std::complex<double>>& phi, int d, std::size_t dk) {
  std::cout << "  column norms (" << name << "):\n";
  for (std::size_t j = 0; j < dk; ++j) {
    double s = 0.0;
    for (int p = 0; p < d; ++p) {
      s += std::norm(phi[static_cast<std::size_t>(p) + j * static_cast<std::size_t>(d)]);
    }
    std::cout << "    j=" << j << "  ||col||=" << std::sqrt(s) << "\n";
  }
}

}  // namespace

int main() {
  ftlm::HubbardParams p;
  p.Lx = Lx;
  p.Ly = Ly;
  p.t = 1.0;
  p.tp = -0.35;
  p.U = 5.75;
  p.V = 0.9;
  p.phi_x = 0.0;
  p.phi_y = 0.0;

  constexpr int Nup = 2;
  constexpr int Ndn = 2;
  const int kx = 0;
  const int ky = 0;

  ftlm::RectLattice lat{Lx, Ly};
  std::vector<ftlm::SpinfulHopping> hops;
  std::vector<ftlm::NearestPair> pairs;
  ftlm::build_hubbard_geometry(p, lat, &hops, &pairs);

  ftlm::FockBasis fb(Lx * Ly, Nup, Ndn);
  const int d = fb.dim();
  std::vector<ftlm::symmetry::RawState> universe;
  for (int i = 0; i < d; ++i) {
    universe.push_back(ftlm::symmetry::RawState{static_cast<std::uint16_t>(fb.up_mask(i)),
                                                  static_cast<std::uint16_t>(fb.down_mask(i))});
  }
  const auto map = ftlm::symmetry::build_momentum_sector_map_rect(std::move(universe), Lx, Ly);
  const ftlm::symmetry::MomentumSector K{kx, ky, Lx, Ly};
  const auto kb = ftlm::symmetry::KBasis::build(map, K);
  const std::vector<ftlm::symmetry::RawState> seeds = ftlm::symmetry::momentum_phi_seeds(map, K);

  std::cout << std::setprecision(14);
  std::cout << "=== Φ compare: 2x2 sector (" << Nup << "," << Ndn << ")  K=(" << kx << "," << ky << ")  d_full=" << d
            << "  KBasis.dim(reps)=" << kb.dim() << "  seeds=" << seeds.size() << " ===\n\n";

  std::vector<std::complex<double>> phi_old;
  std::size_t dk_old = 0;
  build_phi_old_dense_projector(fb, Lx, Ly, K, &phi_old, &dk_old);
  std::cout << "Φ_old (dense P + eigh, eval>tol): dk=" << dk_old << "\n";
  print_col_norms("old", phi_old, d, dk_old);

  std::vector<std::complex<double>> phi_direct;
  phi_direct.assign(static_cast<std::size_t>(d) * seeds.size(), 0.0);
  for (std::size_t j = 0; j < seeds.size(); ++j) {
    ftlm::symmetry::detail::fill_phi_orbit_bloch_from_seed(map, K, Lx, Ly, fb, seeds[j],
                                                           phi_direct.data() + j * static_cast<std::size_t>(d));
  }
  std::cout << "\nΦ_direct (orbit Bloch + fermionic U(R), raw cols=" << seeds.size() << "):\n";
  print_col_norms("direct", phi_direct, d, seeds.size());

  std::vector<std::vector<std::complex<double>>> Pdense;
  ftlm::symmetry::detail::build_translation_projector_dense(fb, Lx, Ly, K, &Pdense);
  ftlm::symmetry::detail::hermitian_symmetrize_inplace(&Pdense);
  double worst_ratio = 0.0;
  int worst_j = -1;
  for (std::size_t j = 0; j < seeds.size(); ++j) {
    std::vector<std::complex<double>> v(static_cast<std::size_t>(d));
    for (int p = 0; p < d; ++p) {
      v[static_cast<std::size_t>(p)] =
          phi_direct[static_cast<std::size_t>(p) + j * static_cast<std::size_t>(d)];
    }
    const std::vector<std::complex<double>> Pv = apply_dense_P(Pdense, v);
    double nrm_r = 0.0;
    double nrm_v = 0.0;
    for (int p = 0; p < d; ++p) {
      const std::complex<double> r = Pv[static_cast<std::size_t>(p)] - v[static_cast<std::size_t>(p)];
      nrm_r += std::norm(r);
      nrm_v += std::norm(v[static_cast<std::size_t>(p)]);
    }
    const double ratio = std::sqrt(nrm_r) / std::max(1e-300, std::sqrt(nrm_v));
    if (ratio > worst_ratio) {
      worst_ratio = ratio;
      worst_j = static_cast<int>(j);
    }
  }
  std::cout << "max_j ||P_K phi_j - phi_j|| / ||phi_j|| (raw columns) = " << worst_ratio << "  (worst j=" << worst_j
            << ")\n";

  std::vector<std::complex<double>> phi_gs = phi_direct;
  std::size_t dk_gs = ftlm::symmetry::gram_schmidt_orthonormalize_phi_columns(d, seeds.size(), &phi_gs);
  std::cout << "\nΦ_gs (modified GS + drop): dk=" << dk_gs << "\n";
  print_col_norms("gs", phi_gs, d, dk_gs);

  std::vector<std::complex<double>> phi_prod;
  std::size_t dk_prod = 0;
  ftlm::symmetry::build_momentum_phi_orbit_orthonormal(map, K, Lx, Ly, fb, &phi_prod, &dk_prod);
  std::cout << "\nΦ_prod (production Gram whitening): dk=" << dk_prod << "\n";
  print_col_norms("prod", phi_prod, d, dk_prod);

  const auto G_old = gram_matrix(phi_old, d, dk_old);
  const auto G_direct = gram_matrix(phi_direct, d, seeds.size());
  const auto G_gs = gram_matrix(phi_gs, d, dk_gs);

  std::cout << "\nmax offdiag |G_old - I| (Hermitian offdiag): ";
  double mo = 0.0;
  for (std::size_t i = 0; i < dk_old; ++i) {
    for (std::size_t j = 0; j < dk_old; ++j) {
      const std::complex<double> z = G_old[i][j] - ((i == j) ? std::complex<double>(1.0, 0.0) : std::complex<double>(0.0, 0.0));
      if (i != j) {
        mo = std::max(mo, std::abs(z));
      }
    }
  }
  std::cout << mo << "\n";

  std::cout << "G_direct: max |G_ij| (full matrix max abs) for Gram check: " << max_abs_mat(G_direct) << "\n";

  const auto Q_old = projector_Q(phi_old, d, dk_old);
  const auto Q_gs = projector_Q(phi_gs, d, dk_gs);
  const auto Q_prod = projector_Q(phi_prod, d, dk_prod);

  std::cout << "\n||Q_old - Q_gs||_max = " << max_abs_diff_mat(Q_old, Q_gs) << "\n";
  std::cout << "||Q_old - Q_prod||_max = " << max_abs_diff_mat(Q_old, Q_prod) << "\n";
  std::cout << "||Q_gs - Q_prod||_max (same span => 0): " << max_abs_diff_mat(Q_gs, Q_prod) << "\n";

  const auto svals_gs = singular_values_overlap(phi_old, d, dk_old, phi_gs, dk_gs);
  std::cout << "Singular values of Phi_old^H Phi_gs:\n";
  for (double s : svals_gs) {
    std::cout << "  " << s << "\n";
  }
  const auto svals_prod = singular_values_overlap(phi_old, d, dk_old, phi_prod, dk_prod);
  std::cout << "Singular values of Phi_old^H Phi_prod:\n";
  for (double s : svals_prod) {
    std::cout << "  " << s << "\n";
  }

  // Reduced H for sanity
  std::vector<std::vector<std::complex<double>>> H_full(
      static_cast<std::size_t>(d), std::vector<std::complex<double>>(static_cast<std::size_t>(d), 0.0));
  std::vector<std::complex<double>> e(static_cast<std::size_t>(d), 0.0), y(static_cast<std::size_t>(d), 0.0);
  for (int j = 0; j < d; ++j) {
    std::fill(e.begin(), e.end(), std::complex<double>(0.0, 0.0));
    e[static_cast<std::size_t>(j)] = {1.0, 0.0};
    ftlm::apply_extended_hubbard(p, fb, hops, pairs, e.data(), y.data());
    for (int i = 0; i < d; ++i) {
      H_full[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = y[static_cast<std::size_t>(i)];
    }
  }
  auto H_red = [&](const std::vector<std::complex<double>>& phi, std::size_t dk) {
    std::vector<std::vector<std::complex<double>>> Hr(
        dk, std::vector<std::complex<double>>(dk, std::complex<double>(0.0, 0.0)));
    for (std::size_t a = 0; a < dk; ++a) {
      for (std::size_t b = 0; b < dk; ++b) {
        std::complex<double> s(0.0, 0.0);
        for (int p = 0; p < d; ++p) {
          for (int q = 0; q < d; ++q) {
            s += std::conj(phi[static_cast<std::size_t>(p) + a * static_cast<std::size_t>(d)]) *
                 H_full[static_cast<std::size_t>(p)][static_cast<std::size_t>(q)] *
                 phi[static_cast<std::size_t>(q) + b * static_cast<std::size_t>(d)];
          }
        }
        Hr[a][b] = s;
      }
    }
    return Hr;
  };
  const auto Hro = H_red(phi_old, dk_old);
  const auto Hrg = H_red(phi_gs, dk_gs);
  const auto Hrp = H_red(phi_prod, dk_prod);
  std::cout << "\nmax |H_red_old - H_red_gs| (if dk match): ";
  if (dk_old == dk_gs) {
    double mh = 0.0;
    for (std::size_t a = 0; a < dk_old; ++a) {
      for (std::size_t b = 0; b < dk_old; ++b) {
        mh = std::max(mh, std::abs(Hro[a][b] - Hrg[a][b]));
      }
    }
    std::cout << mh << "\n";
  } else {
    std::cout << "n/a (dk_old=" << dk_old << " dk_gs=" << dk_gs << ")\n";
  }
  std::cout << "max |H_red_old - H_red_prod| (if dk match): ";
  if (dk_old == dk_prod) {
    double mh = 0.0;
    for (std::size_t a = 0; a < dk_old; ++a) {
      for (std::size_t b = 0; b < dk_old; ++b) {
        mh = std::max(mh, std::abs(Hro[a][b] - Hrp[a][b]));
      }
    }
    std::cout << mh << "\n";
  } else {
    std::cout << "n/a (dk_old=" << dk_old << " dk_prod=" << dk_prod << ")\n";
  }

  return 0;
}
