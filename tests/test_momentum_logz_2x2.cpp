#include <cmath>
#include <complex>
#include <functional>
#include <iostream>
#include <limits>
#include <vector>

#include "ftlm/fock_basis.hpp"
#include "ftlm/hubbard_hamiltonian.hpp"
#include "ftlm/hubbard_params.hpp"
#include "ftlm/lattice.hpp"
#include "ftlm/symmetry/k_basis.hpp"
#include "ftlm/symmetry/momentum_sector.hpp"
#include "ftlm/symmetry/packed_raw_state.hpp"
#include "ftlm/symmetry/raw_state.hpp"

namespace {

void jacobi_symmetric_all(std::vector<std::vector<double>>& A, int max_sweeps, double tol_offdiag) {
  const int n = static_cast<int>(A.size());
  if (n <= 1) return;
  for (int sweep = 0; sweep < max_sweeps; ++sweep) {
    int p = 0, q = 1;
    double max_abs = 0.0;
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        const double v = std::abs(A[static_cast<size_t>(i)][static_cast<size_t>(j)]);
        if (v > max_abs) {
          max_abs = v;
          p = i;
          q = j;
        }
      }
    }
    if (max_abs < tol_offdiag) break;
    const double app = A[static_cast<size_t>(p)][static_cast<size_t>(p)];
    const double aqq = A[static_cast<size_t>(q)][static_cast<size_t>(q)];
    const double apq = A[static_cast<size_t>(p)][static_cast<size_t>(q)];
    const double phi = 0.5 * std::atan2(2.0 * apq, aqq - app);
    const double c = std::cos(phi), s = std::sin(phi);
    for (int k = 0; k < n; ++k) {
      if (k == p || k == q) continue;
      const double apk = A[static_cast<size_t>(p)][static_cast<size_t>(k)];
      const double aqk = A[static_cast<size_t>(q)][static_cast<size_t>(k)];
      const double rpk = c * apk - s * aqk;
      const double rqk = c * aqk + s * apk;
      A[static_cast<size_t>(p)][static_cast<size_t>(k)] = rpk;
      A[static_cast<size_t>(k)][static_cast<size_t>(p)] = rpk;
      A[static_cast<size_t>(q)][static_cast<size_t>(k)] = rqk;
      A[static_cast<size_t>(k)][static_cast<size_t>(q)] = rqk;
    }
    const double new_pp = c * c * app - 2.0 * s * c * apq + s * s * aqq;
    const double new_qq = s * s * app + 2.0 * s * c * apq + c * c * aqq;
    A[static_cast<size_t>(p)][static_cast<size_t>(q)] = 0.0;
    A[static_cast<size_t>(q)][static_cast<size_t>(p)] = 0.0;
    A[static_cast<size_t>(p)][static_cast<size_t>(p)] = new_pp;
    A[static_cast<size_t>(q)][static_cast<size_t>(q)] = new_qq;
  }
}

double exact_log_partition_from_apply(
    int dim, const std::function<void(const std::complex<double>*, std::complex<double>*)>& apply_h, double beta) {
  if (dim <= 0) return -std::numeric_limits<double>::infinity();
  std::vector<std::vector<std::complex<double>>> H(
      static_cast<size_t>(dim), std::vector<std::complex<double>>(static_cast<size_t>(dim), {0.0, 0.0}));
  std::vector<std::complex<double>> e(static_cast<size_t>(dim), {0.0, 0.0});
  std::vector<std::complex<double>> y(static_cast<size_t>(dim), {0.0, 0.0});
  for (int j = 0; j < dim; ++j) {
    std::fill(e.begin(), e.end(), std::complex<double>(0.0, 0.0));
    e[static_cast<size_t>(j)] = {1.0, 0.0};
    apply_h(e.data(), y.data());
    for (int i = 0; i < dim; ++i) H[static_cast<size_t>(i)][static_cast<size_t>(j)] = y[static_cast<size_t>(i)];
  }

  const int m = 2 * dim;
  std::vector<std::vector<double>> R(static_cast<size_t>(m), std::vector<double>(static_cast<size_t>(m), 0.0));
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      const auto hij = H[static_cast<size_t>(i)][static_cast<size_t>(j)];
      R[static_cast<size_t>(i)][static_cast<size_t>(j)] = hij.real();
      R[static_cast<size_t>(i)][static_cast<size_t>(j + dim)] = -hij.imag();
      R[static_cast<size_t>(i + dim)][static_cast<size_t>(j)] = hij.imag();
      R[static_cast<size_t>(i + dim)][static_cast<size_t>(j + dim)] = hij.real();
    }
  }
  jacobi_symmetric_all(R, std::max(5000, 120 * m), 1e-13);
  double max_t = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < m; ++i) max_t = std::max(max_t, -beta * R[static_cast<size_t>(i)][static_cast<size_t>(i)]);
  double sum = 0.0;
  for (int i = 0; i < m; ++i) sum += std::exp(-beta * R[static_cast<size_t>(i)][static_cast<size_t>(i)] - max_t);
  return std::log(0.5) + max_t + std::log(sum);
}

/// One column of the stacked Bloch matrix (same lift as `HubbardMomentumAction::apply` / Gram reference).
void fill_phi_column_rect(const ftlm::symmetry::MomentumSectorMap& map, ftlm::symmetry::MomentumSector K,
                          const ftlm::symmetry::KBasis& kb, int jcol, const ftlm::FockBasis& fb, int lx, int ly,
                          std::complex<double>* col) {
  const int d_full = fb.dim();
  std::fill(col, col + d_full, std::complex<double>(0.0, 0.0));
  const ftlm::symmetry::RawState r = kb.representatives[static_cast<size_t>(jcol)];
  std::size_t stab = 1;
  const std::uint32_t pk = ftlm::symmetry::pack_raw_state(r);
  for (const auto& o : map.orbits) {
    if (ftlm::symmetry::pack_raw_state(o.representative) == pk) {
      stab = o.stabilizer.size();
      break;
    }
  }
  const double c = ftlm::symmetry::momentum_orbit_normalization_factor_rect(stab, lx, ly);
  for (int ey = 0; ey < ly; ++ey) {
    for (int ex = 0; ex < lx; ++ex) {
      const ftlm::symmetry::RawState st = ftlm::symmetry::translate_raw_state_rect(r, lx, ly, ex, ey);
      const int idx = fb.index_of(st.up, st.dn);
      if (idx >= 0) {
        col[idx] += c * std::conj(ftlm::symmetry::translation_bloch_phase(K, ex, ey));
      }
    }
  }
}

}  // namespace

int main() {
  ftlm::HubbardParams p;
  p.Lx = 2;
  p.Ly = 2;
  p.t = 1.0;
  p.tp = -0.35;
  p.U = 5.75;
  p.V = 0.9;
  p.phi_x = 0.0;
  p.phi_y = 0.0;
  const double beta = 20.0;
  constexpr int lx = 2;
  constexpr int ly = 2;

  ftlm::RectLattice lat{p.Lx, p.Ly};
  std::vector<ftlm::SpinfulHopping> hops;
  std::vector<ftlm::NearestPair> pairs;
  ftlm::build_hubbard_geometry(p, lat, &hops, &pairs);

  for (int nu = 0; nu <= 4; ++nu) {
    for (int nd = 0; nd <= 4; ++nd) {
      ftlm::FockBasis fb(4, nu, nd);
      const int dim = fb.dim();
      if (dim <= 0) continue;

      auto apply_full = [&](const std::complex<double>* x, std::complex<double>* y) {
        ftlm::apply_extended_hubbard(p, fb, hops, pairs, x, y);
      };
      const double logz_full = exact_log_partition_from_apply(dim, apply_full, beta);

      std::vector<ftlm::symmetry::RawState> universe;
      universe.reserve(static_cast<size_t>(dim));
      for (int i = 0; i < dim; ++i) {
        universe.push_back(ftlm::symmetry::RawState{static_cast<std::uint16_t>(fb.up_mask(i)),
                                                    static_cast<std::uint16_t>(fb.down_mask(i))});
      }
      const auto map = ftlm::symmetry::build_momentum_sector_map_rect(std::move(universe), lx, ly);

      // Stacked Bloch basis: columns = all (K, orbit_index) vectors in ky,kx,j order.  H_red = Phi^H H Phi
      // is dim×dim Hermitian with the same spectrum as H in the Fock sector (Phi is unitary when
      // sum_K dim(K) = dim).  Summing partition functions log Z_K from disjoint K blocks is wrong when
      // H has cross-K matrix elements in this basis (see Gram / refactor diagnostics for sector (0,2)).
      std::vector<std::complex<double>> phi(static_cast<size_t>(dim) * static_cast<size_t>(dim));
      int col = 0;
      for (int ky = 0; ky < ly; ++ky) {
        for (int kx = 0; kx < lx; ++kx) {
          const ftlm::symmetry::MomentumSector K{kx, ky, lx, ly};
          const auto kb = ftlm::symmetry::KBasis::build(map, K);
          for (std::size_t j = 0; j < kb.dim(); ++j) {
            fill_phi_column_rect(map, K, kb, static_cast<int>(j), fb, lx, ly,
                                 phi.data() + static_cast<size_t>(col) * static_cast<size_t>(dim));
            ++col;
          }
        }
      }
      if (col != dim) {
        std::cerr << "stacked Bloch column count mismatch sector (" << nu << "," << nd << "): col=" << col
                  << " dim=" << dim << "\n";
        return 1;
      }

      std::vector<std::vector<std::complex<double>>> Hred(
          static_cast<size_t>(dim), std::vector<std::complex<double>>(static_cast<size_t>(dim)));
      std::vector<std::complex<double>> v(static_cast<size_t>(dim)), w(static_cast<size_t>(dim));
      for (int b = 0; b < dim; ++b) {
        for (int p = 0; p < dim; ++p) {
          v[static_cast<size_t>(p)] = phi[static_cast<size_t>(p) + static_cast<size_t>(b) * static_cast<size_t>(dim)];
        }
        ftlm::apply_extended_hubbard(p, fb, hops, pairs, v.data(), w.data());
        for (int a = 0; a < dim; ++a) {
          std::complex<double> s(0.0, 0.0);
          for (int p = 0; p < dim; ++p) {
            s += std::conj(phi[static_cast<size_t>(p) + static_cast<size_t>(a) * static_cast<size_t>(dim)]) *
                 w[static_cast<size_t>(p)];
          }
          Hred[static_cast<size_t>(a)][static_cast<size_t>(b)] = s;
        }
      }

      auto apply_red = [&](const std::complex<double>* x, std::complex<double>* y) {
        for (int i = 0; i < dim; ++i) {
          std::complex<double> acc(0.0, 0.0);
          for (int j = 0; j < dim; ++j) {
            acc += Hred[static_cast<size_t>(i)][static_cast<size_t>(j)] * x[j];
          }
          y[i] = acc;
        }
      };
      const double logz_red = exact_log_partition_from_apply(dim, apply_red, beta);

      const double err = std::abs(logz_full - logz_red);
      if (err > 1e-8) {
        std::cerr << "logZ mismatch sector (" << nu << "," << nd << "): full=" << logz_full
                  << " stacked_red=" << logz_red << " err=" << err << "\n";
        return 1;
      }
    }
  }

  std::cout << "test_momentum_logz_2x2 ok\n";
  return 0;
}
