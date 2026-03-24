#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "ftlm/fock_basis.hpp"
#include "ftlm/hubbard_hamiltonian.hpp"
#include "ftlm/hubbard_params.hpp"
#include "ftlm/lattice.hpp"
#include "ftlm/symmetry/k_basis.hpp"
#include "ftlm/symmetry/momentum_sector.hpp"
#include "ftlm/symmetry/orbit.hpp"
#include "ftlm/symmetry/packed_raw_state.hpp"
#include "ftlm/symmetry/raw_state.hpp"

namespace {

constexpr int kLx = 2;
constexpr int kLy = 2;
constexpr int kNs = kLx * kLy;

double choose_int(int n, int k) {
  if (k < 0 || k > n) return 0.0;
  if (k == 0 || k == n) return 1.0;
  k = std::min(k, n - k);
  double r = 1.0;
  for (int i = 1; i <= k; ++i) {
    r *= static_cast<double>(n - k + i) / static_cast<double>(i);
  }
  return r;
}

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

std::vector<double> eigvals_hermitian_from_apply(
    int dim, const std::function<void(const std::complex<double>*, std::complex<double>*)>& apply_h) {
  if (dim <= 0) return {};
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
  jacobi_symmetric_all(R, std::max(6000, 150 * m), 1e-13);
  std::vector<double> ev(static_cast<size_t>(m));
  for (int i = 0; i < m; ++i) ev[static_cast<size_t>(i)] = R[static_cast<size_t>(i)][static_cast<size_t>(i)];
  std::sort(ev.begin(), ev.end());
  std::vector<double> out;
  out.reserve(static_cast<size_t>(dim));
  for (int i = 0; i < m; i += 2) out.push_back(ev[static_cast<size_t>(i)]);
  return out;
}

double logZ_from_eigs(const std::vector<double>& e, double beta) {
  if (e.empty()) return -std::numeric_limits<double>::infinity();
  double m = -std::numeric_limits<double>::infinity();
  for (double x : e) m = std::max(m, -beta * x);
  double s = 0.0;
  for (double x : e) s += std::exp(-beta * x - m);
  return m + std::log(s);
}

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
  double mx = 0.0;
  for (size_t i = 0; i < a.size(); ++i) mx = std::max(mx, std::abs(a[i] - b[i]));
  return mx;
}

double grand_canonical_n_from_sector_logz(const std::map<std::pair<int, int>, double>& logz, double beta, double mu) {
  double m = -std::numeric_limits<double>::infinity();
  for (const auto& kv : logz) {
    const int n = kv.first.first + kv.first.second;
    m = std::max(m, kv.second + beta * mu * static_cast<double>(n));
  }
  double zw = 0.0;
  double nw = 0.0;
  for (const auto& kv : logz) {
    const int n = kv.first.first + kv.first.second;
    const double w = std::exp(kv.second + beta * mu * static_cast<double>(n) - m);
    zw += w;
    nw += static_cast<double>(n) * w;
  }
  return (nw / zw) / static_cast<double>(kNs);
}

}  // namespace

int main() {
  ftlm::HubbardParams p;
  p.Lx = kLx;
  p.Ly = kLy;
  p.t = 1.0;
  p.tp = -0.35;
  p.U = 5.75;
  p.V = 0.9;
  p.phi_x = 0.0;
  p.phi_y = 0.0;

  ftlm::RectLattice lat{kLx, kLy};
  std::vector<ftlm::SpinfulHopping> hops;
  std::vector<ftlm::NearestPair> pairs;
  ftlm::build_hubbard_geometry(p, lat, &hops, &pairs);

  std::cout << "[Step 1] Hilbert-space completeness per (Nup,Ndn)\n";

  std::map<std::pair<int, int>, std::vector<double>> full_sector_eigs;
  std::map<std::pair<int, int>, std::vector<double>> k_sector_eigs_concat;
  std::map<std::pair<int, int>, std::map<double, double>> logz_old_beta;
  std::map<std::pair<int, int>, std::map<double, double>> logz_k_beta;

  const std::vector<double> betas = {0.0, 2.0, 20.0};
  bool first_fail_reported = false;

  for (int nu = 0; nu <= kNs; ++nu) {
    for (int nd = 0; nd <= kNs; ++nd) {
      const int expected = static_cast<int>(choose_int(kNs, nu) * choose_int(kNs, nd) + 0.5);
      ftlm::FockBasis fb(kNs, nu, nd);
      const int dim_full = fb.dim();
      if (dim_full != expected) {
        std::cerr << "FockBasis dim mismatch at sector (" << nu << "," << nd << "): expected=" << expected
                  << " got=" << dim_full << "\n";
        return 1;
      }
      if (dim_full <= 0) continue;

      std::vector<ftlm::symmetry::RawState> universe;
      universe.reserve(static_cast<size_t>(dim_full));
      for (int i = 0; i < dim_full; ++i) {
        universe.push_back(ftlm::symmetry::RawState{static_cast<std::uint16_t>(fb.up_mask(i)),
                                                    static_cast<std::uint16_t>(fb.down_mask(i))});
      }
      const auto map = ftlm::symmetry::build_momentum_sector_map_rect(std::move(universe), kLx, kLy);

      int sum_k_dim = 0;
      std::set<std::uint32_t> reps_seen;
      bool dup_in_block = false;
      for (int ky = 0; ky < kLy; ++ky) {
        for (int kx = 0; kx < kLx; ++kx) {
          const ftlm::symmetry::MomentumSector K{kx, ky, kLx, kLy};
          const auto kb = ftlm::symmetry::KBasis::build(map, K);
          sum_k_dim += static_cast<int>(kb.dim());
          std::set<std::uint32_t> local;
          for (const auto& rep : kb.representatives) {
            const auto pk = ftlm::symmetry::pack_raw_state(rep);
            if (!local.insert(pk).second) dup_in_block = true;
            reps_seen.insert(pk);
          }
        }
      }
      const bool complete = (sum_k_dim == dim_full);
      std::cout << "sector(" << nu << "," << nd << ") expected=" << expected << " sum_k_dim=" << sum_k_dim
                << " match=" << (complete ? "YES" : "NO") << "\n";
      if (!complete || dup_in_block) {
        std::cout << "  details: dup_in_block=" << (dup_in_block ? "YES" : "NO") << " unique_reps_seen="
                  << reps_seen.size() << "\n";
      }

      auto apply_full = [&](const std::complex<double>* x, std::complex<double>* y) {
        ftlm::apply_extended_hubbard(p, fb, hops, pairs, x, y);
      };
      const auto eig_full = eigvals_hermitian_from_apply(dim_full, apply_full);
      full_sector_eigs[{nu, nd}] = eig_full;

      std::vector<double> eig_k_all;
      for (int ky = 0; ky < kLy; ++ky) {
        for (int kx = 0; kx < kLx; ++kx) {
          const ftlm::symmetry::MomentumSector K{kx, ky, kLx, kLy};
          const auto kb = ftlm::symmetry::KBasis::build(map, K);
          const int dk = static_cast<int>(kb.dim());
          if (dk <= 0) continue;

          std::vector<double> c_norm(static_cast<size_t>(dk), 0.0);
          for (int i = 0; i < dk; ++i) {
            const auto rep = kb.representatives[static_cast<size_t>(i)];
            const auto pk = ftlm::symmetry::pack_raw_state(rep);
            std::size_t s_sz = 1;
            for (const auto& o : map.orbits) {
              if (ftlm::symmetry::pack_raw_state(o.representative) == pk) {
                s_sz = o.stabilizer.size();
                break;
              }
            }
            c_norm[static_cast<size_t>(i)] = ftlm::symmetry::momentum_orbit_normalization_factor_rect(s_sz, kLx, kLy);
          }

          std::vector<std::complex<double>> full_in(static_cast<size_t>(dim_full), {0.0, 0.0});
          std::vector<std::complex<double>> full_out(static_cast<size_t>(dim_full), {0.0, 0.0});
          auto apply_k = [&](const std::complex<double>* x, std::complex<double>* y) {
            std::fill(full_in.begin(), full_in.end(), std::complex<double>(0.0, 0.0));
            for (int i = 0; i < dk; ++i) {
              const auto xi = x[static_cast<size_t>(i)];
              if (std::norm(xi) < 1e-30) continue;
              const auto rep = kb.representatives[static_cast<size_t>(i)];
              const double c = c_norm[static_cast<size_t>(i)];
              for (int dy = 0; dy < kLy; ++dy) {
                for (int dx = 0; dx < kLx; ++dx) {
                  const auto st = ftlm::symmetry::translate_raw_state_rect(rep, kLx, kLy, dx, dy);
                  const int idx = fb.index_of(st.up, st.dn);
                  if (idx >= 0) {
                    const auto ph = ftlm::symmetry::translation_bloch_phase(K, dx, dy);
                    full_in[static_cast<size_t>(idx)] += xi * c * std::conj(ph);
                  }
                }
              }
            }
            ftlm::apply_extended_hubbard(p, fb, hops, pairs, full_in.data(), full_out.data());
            for (int i = 0; i < dk; ++i) {
              std::complex<double> yi(0.0, 0.0);
              const auto rep = kb.representatives[static_cast<size_t>(i)];
              const double c = c_norm[static_cast<size_t>(i)];
              for (int dy = 0; dy < kLy; ++dy) {
                for (int dx = 0; dx < kLx; ++dx) {
                  const auto st = ftlm::symmetry::translate_raw_state_rect(rep, kLx, kLy, dx, dy);
                  const int idx = fb.index_of(st.up, st.dn);
                  if (idx >= 0) {
                    const auto ph = ftlm::symmetry::translation_bloch_phase(K, dx, dy);
                    yi += (c * ph) * full_out[static_cast<size_t>(idx)];
                  }
                }
              }
              y[static_cast<size_t>(i)] = yi;
            }
          };

          const auto eig_k = eigvals_hermitian_from_apply(dk, apply_k);
          eig_k_all.insert(eig_k_all.end(), eig_k.begin(), eig_k.end());
        }
      }
      std::sort(eig_k_all.begin(), eig_k_all.end());
      k_sector_eigs_concat[{nu, nd}] = eig_k_all;

      // Step 2
      const double emax = max_abs_diff(eig_full, eig_k_all);
      std::cout << "  [Step 2] eig_max_abs_diff=" << emax << "\n";
      if (!first_fail_reported && emax > 1e-8) {
        first_fail_reported = true;
        std::cout << "  ==> FIRST DISAGREEMENT at sector(" << nu << "," << nd << ") in eigenvalue equivalence.\n";
      }

      // Step 3 + Step 6
      for (double beta : betas) {
        const double lz_old = logZ_from_eigs(eig_full, beta);
        const double lz_k = logZ_from_eigs(eig_k_all, beta);
        logz_old_beta[{nu, nd}][beta] = lz_old;
        logz_k_beta[{nu, nd}][beta] = lz_k;
        const double z_old = std::exp(lz_old);
        const double z_k = std::exp(lz_k);
        const double rel = (z_old > 0.0) ? std::abs(z_old - z_k) / z_old : 0.0;
        std::cout << "  [Step 3] beta=" << beta << " Z_old=" << z_old << " Z_k=" << z_k << " rel_err=" << rel
                  << "\n";
        if (beta == 0.0) {
          std::cout << "  [Step 6] beta=0 check: Z_old=" << z_old << " Z_k=" << z_k << " dim_full=" << dim_full
                    << "\n";
        }
      }

      // Step 7: orbit compatibility and projected norm sanity
      std::cout << "  [Step 7] orbit diagnostics (first few)\n";
      int shown = 0;
      for (const auto& orb : map.orbits) {
        if (shown >= 3) break;
        const int comp = ftlm::symmetry::popcount_momentum_mask(orb.compatible_momentum_mask);
        std::cout << "    orbit_size=" << orb.orbit_size << " compat_k_count=" << comp << "\n";
        ++shown;
      }
    }
  }

  // Step 4: n(mu) via sector traces only
  std::cout << "[Step 4] n(mu) comparison from sector traces only\n";
  for (double mu : {-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0}) {
    const double n_old = grand_canonical_n_from_sector_logz(
        [&]() {
          std::map<std::pair<int, int>, double> z;
          for (const auto& kv : logz_old_beta) z[kv.first] = kv.second.at(20.0);
          return z;
        }(),
        20.0, mu);
    const double n_k = grand_canonical_n_from_sector_logz(
        [&]() {
          std::map<std::pair<int, int>, double> z;
          for (const auto& kv : logz_k_beta) z[kv.first] = kv.second.at(20.0);
          return z;
        }(),
        20.0, mu);
    std::cout << "  mu=" << mu << " n_old=" << n_old << " n_k=" << n_k << " diff=" << std::abs(n_old - n_k)
              << "\n";
  }

  std::cout << "[Step 5] mu convention audit summary:\n";
  std::cout << "  This diagnostic uses H without mu and applies exp(beta*mu*N) outside sector traces.\n";
  std::cout << "  No double counting of mu in this diagnostic path.\n";

  return 0;
}
