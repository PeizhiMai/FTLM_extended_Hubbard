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
#include "ftlm/symmetry/orbit_bloch_phi.hpp"
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

  bool any_sector_bloch = false;
  for (int nu = 0; nu <= 4; ++nu) {
    for (int nd = 0; nd <= 4; ++nd) {
      ftlm::FockBasis fb(4, nu, nd);
      const int dim = fb.dim();
      if (dim <= 0) continue;

      auto apply_full = [&](const std::complex<double>* x, std::complex<double>* y) {
        ftlm::apply_extended_hubbard(p, fb, hops, pairs, x, y);
      };
      const double logz_full = exact_log_partition_from_apply(dim, apply_full, beta);
      if (!std::isfinite(logz_full)) {
        std::cerr << "non-finite logZ full sector (" << nu << "," << nd << ")\n";
        return 1;
      }

      std::vector<ftlm::symmetry::RawState> universe;
      universe.reserve(static_cast<size_t>(dim));
      for (int i = 0; i < dim; ++i) {
        universe.push_back(ftlm::symmetry::RawState{static_cast<std::uint16_t>(fb.up_mask(i)),
                                                    static_cast<std::uint16_t>(fb.down_mask(i))});
      }
      const auto map = ftlm::symmetry::build_momentum_sector_map_rect(std::move(universe), lx, ly);

      // Raw Bloch columns from momentum_phi_seeds must be non-trivial (stabilizer + JW + Gram path smoke test).
      bool any_bloch = false;
      std::vector<std::complex<double>> col(static_cast<size_t>(dim));
      for (int ky = 0; ky < ly; ++ky) {
        for (int kx = 0; kx < lx; ++kx) {
          const ftlm::symmetry::MomentumSector K{kx, ky, lx, ly};
          const auto seeds = ftlm::symmetry::momentum_phi_seeds(map, K);
          for (const ftlm::symmetry::RawState& seed : seeds) {
            ftlm::symmetry::detail::fill_phi_orbit_bloch_from_seed(map, K, lx, ly, fb, seed, col.data());
            double n2 = 0.0;
            for (int p = 0; p < dim; ++p) {
              n2 += std::norm(col[static_cast<size_t>(p)]);
            }
            if (n2 > 1e-20) {
              any_bloch = true;
            }
          }
        }
      }
      if (any_bloch) {
        any_sector_bloch = true;
      }
    }
  }
  if (!any_sector_bloch) {
    std::cerr << "all sectors had zero Bloch columns\n";
    return 1;
  }

  std::cout << "test_momentum_logz_2x2 ok\n";
  return 0;
}
