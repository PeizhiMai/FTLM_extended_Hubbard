// Focused diagnostics: sector (nu,nd)=(3,3) on 3x2, every K — dk, dim gate, orbit vs production apply in full space,
// then FTLM log partition per K if apply matches (same beta/FTLM params as k-bench defaults for 3x2).
#include "ftlm/fock_basis.hpp"
#include "ftlm/ftlm_thermo.hpp"
#include "ftlm/hubbard_params.hpp"
#include "ftlm/symmetry/hubbard_momentum_action.hpp"
#include "ftlm/symmetry/orbit_bloch_phi.hpp"
#include "ftlm/symmetry/orbit_k_block_basis.hpp"

#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

namespace {

using Cc = std::complex<double>;

void swap_rows_cm(int nrows, int ncols, Cc* a, int r1, int r2) {
  if (r1 == r2) {
    return;
  }
  for (int j = 0; j < ncols; ++j) {
    std::swap(a[r1 + j * nrows], a[r2 + j * nrows]);
  }
}

bool invert_colmajor_gauss_jordan(int n, std::vector<Cc>* a) {
  const int ncol = 2 * n;
  std::vector<Cc> aug(static_cast<std::size_t>(n * ncol), Cc(0.0, 0.0));
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      aug[static_cast<std::size_t>(i + j * n)] = (*a)[static_cast<std::size_t>(i + j * n)];
    }
  }
  for (int j = 0; j < n; ++j) {
    aug[static_cast<std::size_t>(j + (n + j) * n)] = Cc(1.0, 0.0);
  }
  for (int col = 0; col < n; ++col) {
    int piv = col;
    double best = std::abs(aug[static_cast<std::size_t>(col + col * n)]);
    for (int r = col + 1; r < n; ++r) {
      const double v = std::abs(aug[static_cast<std::size_t>(r + col * n)]);
      if (v > best) {
        best = v;
        piv = r;
      }
    }
    if (best < 1e-18) {
      return false;
    }
    swap_rows_cm(n, ncol, aug.data(), col, piv);
    const Cc invd = Cc(1.0, 0.0) / aug[static_cast<std::size_t>(col + col * n)];
    for (int j = 0; j < ncol; ++j) {
      aug[static_cast<std::size_t>(col + j * n)] *= invd;
    }
    for (int r = 0; r < n; ++r) {
      if (r == col) {
        continue;
      }
      const Cc f = aug[static_cast<std::size_t>(r + col * n)];
      if (std::abs(f) < 1e-28) {
        continue;
      }
      for (int j = 0; j < ncol; ++j) {
        aug[static_cast<std::size_t>(r + j * n)] -= f * aug[static_cast<std::size_t>(col + j * n)];
      }
    }
  }
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      (*a)[static_cast<std::size_t>(i + j * n)] = aug[static_cast<std::size_t>(i + (n + j) * n)];
    }
  }
  return true;
}

double max_abs_diff(const std::vector<std::complex<double>>& a, const std::vector<std::complex<double>>& b) {
  double m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    m = std::max(m, std::abs(a[i] - b[i]));
  }
  return m;
}

void matvec_colmajor(int rows, int cols, const std::vector<std::complex<double>>& A,
                     const std::vector<std::complex<double>>& x, std::vector<std::complex<double>>* y) {
  y->assign(static_cast<std::size_t>(rows), std::complex<double>(0.0, 0.0));
  for (int j = 0; j < cols; ++j) {
    const std::complex<double> xj = x[static_cast<std::size_t>(j)];
    for (int i = 0; i < rows; ++i) {
      (*y)[static_cast<std::size_t>(i)] += A[static_cast<std::size_t>(i + j * rows)] * xj;
    }
  }
}

std::vector<std::complex<double>> build_O(const ftlm::FockBasis& fb, const ftlm::symmetry::MomentumSectorMap& map,
                                         const ftlm::symmetry::OrbitKBlockBasis& basis) {
  const int d = fb.dim();
  const int dk = basis.dim();
  std::vector<std::complex<double>> O(static_cast<std::size_t>(d * dk), std::complex<double>(0.0, 0.0));
  for (int j = 0; j < dk; ++j) {
    const auto& e = basis.entries()[static_cast<std::size_t>(j)];
    ftlm::symmetry::detail::fill_phi_orbit_bloch_from_seed(map, basis.K, basis.lx, basis.ly, fb, e.rep,
                                                          O.data() + static_cast<std::size_t>(j * d));
  }
  return O;
}

constexpr int sector_index(int nu, int nd, int n_sites) { return nu * (n_sites + 1) + nd; }

}  // namespace

int main() {
  constexpr int Lx = 3;
  constexpr int Ly = 2;
  constexpr int nu = 3;
  constexpr int nd = 3;
  constexpr int n_sites = Lx * Ly;
  constexpr int idx = sector_index(nu, nd, n_sites);
  constexpr unsigned fseed = 7u;
  constexpr double beta = 20.0;

  ftlm::HubbardParams p{};
  p.Lx = Lx;
  p.Ly = Ly;
  p.t = 1.0;
  p.tp = -0.35;
  p.U = 5.75;
  p.V = 0.9;
  p.phi_x = 0.0;
  p.phi_y = 0.0;

  ftlm::symmetry::HubbardMomentumAction hub(p);
  ftlm::FockBasis fb(n_sites, nu, nd);
  std::vector<ftlm::symmetry::RawState> universe;
  universe.reserve(static_cast<std::size_t>(fb.dim()));
  for (int i = 0; i < fb.dim(); ++i) {
    universe.push_back(ftlm::symmetry::RawState{static_cast<std::uint16_t>(fb.up_mask(i)),
                                                  static_cast<std::uint16_t>(fb.down_mask(i))});
  }
  const auto map = ftlm::symmetry::build_momentum_sector_map_rect(std::move(universe), Lx, Ly);

  std::cout << "=== diag_orbit_sector_33_3x2 ===\n";
  std::cout << "sector (" << nu << "," << nd << ") dim_full=" << fb.dim() << " beta=" << beta
            << " ftlm_random=12 lanczos_steps=72 seed_base=" << fseed << " sector_idx=" << idx << "\n\n";

  double worst_apply = 0.0;
  double worst_lz_diff = 0.0;
  int worst_apply_kx = -1, worst_apply_ky = -1;
  int worst_lz_kx = -1, worst_lz_ky = -1;

  for (int ky = 0; ky < Ly; ++ky) {
    for (int kx = 0; kx < Lx; ++kx) {
      const ftlm::symmetry::MomentumSector K{kx, ky, Lx, Ly};
      ftlm::symmetry::OrbitKBlockBasis orbit_basis;
      orbit_basis.build(fb, K);
      const int dk_orbit = orbit_basis.dim();
      const std::size_t dk_prod_dim = hub.momentum_block_dim(map, K, nu, nd);
      const bool dim_gate = (dk_orbit > 0 && static_cast<std::size_t>(dk_orbit) == dk_prod_dim);
      const bool would_use_orbit_mf =
          dim_gate && (nu > 0 && nd > 0 && nu < n_sites && nd < n_sites) && fb.dim() > 0 && fb.dim() <= 512;

      std::cout << "K=(" << kx << "," << ky << ")  dk_orbit=" << dk_orbit << "  dk_prod=" << dk_prod_dim
                << "  dim_gate_match=" << (dim_gate ? "yes" : "no")
                << "  bench_orbit_mf_eligible=" << (would_use_orbit_mf ? "yes" : "no") << "\n";

      if (!dim_gate || dk_orbit <= 0) {
        std::cout << "  skip apply/lz compare (no common dk)\n\n";
        continue;
      }

      ftlm::symmetry::MomentumBlockScratch scratch;
      ftlm::symmetry::HubbardMomentumBlock prod_block(hub, map, K, nu, nd, &scratch);
      const int dk_prod = static_cast<int>(prod_block.dim());
      if (dk_prod != dk_orbit) {
        std::cout << "  ERROR: HubbardMomentumBlock.dim() != dk_orbit\n\n";
        continue;
      }
      const int d = fb.dim();
      const int dk = dk_orbit;
      std::vector<std::complex<double>> O = build_O(fb, map, orbit_basis);

      std::function<void(const std::complex<double>*, std::complex<double>*)> apply_orbit =
          [&](const std::complex<double>* x, std::complex<double>* y) {
            ftlm::symmetry::apply_orbit_k_block(p, fb, orbit_basis, hub.hoppings, hub.nn_pairs, x, y);
          };
      std::function<void(const std::complex<double>*, std::complex<double>*)> apply_prod =
          [&](const std::complex<double>* x, std::complex<double>* y) { prod_block.apply(x, y); };

      double max_apply = 0.0;
      auto compare_apply = [&](const std::vector<std::complex<double>>& x_orb) {
        std::vector<std::complex<double>> y_orb(static_cast<std::size_t>(dk));
        apply_orbit(x_orb.data(), y_orb.data());
        std::vector<std::complex<double>> v_full;
        matvec_colmajor(d, dk, O, x_orb, &v_full);
        std::vector<std::complex<double>> x_prod(static_cast<std::size_t>(dk));
        prod_block.project_full_to_block(v_full.data(), x_prod.data());
        std::vector<std::complex<double>> y_prod(static_cast<std::size_t>(dk));
        apply_prod(x_prod.data(), y_prod.data());
        std::vector<std::complex<double>> w_orb_full;
        matvec_colmajor(d, dk, O, y_orb, &w_orb_full);
        std::vector<std::complex<double>> w_prod_full(static_cast<std::size_t>(d));
        prod_block.lift_block_to_full(y_prod.data(), w_prod_full.data());
        max_apply = std::max(max_apply, max_abs_diff(w_orb_full, w_prod_full));
      };

      for (int j = 0; j < dk; ++j) {
        std::vector<std::complex<double>> e(static_cast<std::size_t>(dk), std::complex<double>(0.0, 0.0));
        e[static_cast<std::size_t>(j)] = std::complex<double>(1.0, 0.0);
        compare_apply(e);
      }
      std::mt19937 rng(static_cast<std::uint32_t>(kx + 17 * ky + 99));
      std::normal_distribution<double> gauss(0.0, 1.0);
      for (int r = 0; r < 8; ++r) {
        std::vector<std::complex<double>> xr(static_cast<std::size_t>(dk));
        for (int i = 0; i < dk; ++i) {
          xr[static_cast<std::size_t>(i)] = std::complex<double>(gauss(rng), gauss(rng));
        }
        compare_apply(xr);
      }

      std::cout << "  max|H_orb - H_prod| in embedded full space (basis+8 rand): " << max_apply << "\n";
      if (max_apply > worst_apply) {
        worst_apply = max_apply;
        worst_apply_kx = kx;
        worst_apply_ky = ky;
      }

      const unsigned seed_k = fseed + static_cast<unsigned>(idx * 257u + static_cast<unsigned>(ky) * 17u +
                                                              static_cast<unsigned>(kx));
      ftlm::FtlmParams fpar;
      fpar.n_random = 12;
      fpar.lanczos_steps = 72;
      fpar.seed = seed_k;

      scratch.ensure_d_full(d);
      std::vector<Cc> T(static_cast<std::size_t>(dk * dk), Cc(0.0, 0.0));
      for (int jc = 0; jc < dk; ++jc) {
        const auto& rep = orbit_basis.entries()[static_cast<std::size_t>(jc)].rep;
        ftlm::symmetry::detail::fill_phi_orbit_bloch_from_seed(map, K, Lx, Ly, fb, rep, scratch.vin.data());
        prod_block.project_full_to_block(scratch.vin.data(), T.data() + jc * dk);
      }
      std::vector<Cc> Tinv = T;
      const bool t_ok = invert_colmajor_gauss_jordan(dk, &Tinv);
      std::vector<Cc> xo(static_cast<std::size_t>(dk));
      std::vector<Cc> yo(static_cast<std::size_t>(dk));

      std::function<void(const std::complex<double>*, std::complex<double>*)> apply_orbit_gram =
          [&](const std::complex<double>* x, std::complex<double>* y) {
            for (int i = 0; i < dk; ++i) {
              Cc acc(0.0, 0.0);
              for (int k = 0; k < dk; ++k) {
                acc += Tinv[static_cast<std::size_t>(i + k * dk)] * x[k];
              }
              xo[static_cast<std::size_t>(i)] = acc;
            }
            apply_orbit(xo.data(), yo.data());
            for (int i = 0; i < dk; ++i) {
              Cc acc(0.0, 0.0);
              for (int k = 0; k < dk; ++k) {
                acc += T[static_cast<std::size_t>(i + k * dk)] * yo[static_cast<std::size_t>(k)];
              }
              y[i] = acc;
            }
          };

      ftlm::LanczosComplexWorkspace ws_orb_raw;
      ftlm::LanczosComplexWorkspace ws_orb_wrap;
      ftlm::LanczosComplexWorkspace ws_prod;
      ftlm::FtlmTridiagonalQuadratureScratch qs_orb_raw;
      ftlm::FtlmTridiagonalQuadratureScratch qs_orb_wrap;
      ftlm::FtlmTridiagonalQuadratureScratch qs_prod;
      fpar.lanczos_ws = &ws_orb_raw;
      fpar.quad_scratch = &qs_orb_raw;
      const double lz_orb_raw = ftlm::ftlm_log_partition_complex(dk, apply_orbit, beta, fpar);
      fpar.lanczos_ws = &ws_prod;
      fpar.quad_scratch = &qs_prod;
      const double lz_prod = ftlm::ftlm_log_partition_complex(dk, apply_prod, beta, fpar);
      double lz_orb_wrap = lz_prod;
      double lz_wrap_diff = 0.0;
      if (t_ok) {
        fpar.lanczos_ws = &ws_orb_wrap;
        fpar.quad_scratch = &qs_orb_wrap;
        lz_orb_wrap = ftlm::ftlm_log_partition_complex(dk, apply_orbit_gram, beta, fpar);
        lz_wrap_diff = std::abs(lz_orb_wrap - lz_prod);
      }

      std::cout << "  FTLM (raw orbit coords for random starts — wrong vs prod): lz=" << lz_orb_raw << "  lz_prod="
                << lz_prod << "  |diff|=" << std::abs(lz_orb_raw - lz_prod) << "\n";
      if (!t_ok) {
        std::cout << "  FTLM Gram-bridge: T invert failed\n\n";
      } else {
        std::cout << "  FTLM (T H_orb T^-1 in Gram coords, matches bench): lz=" << lz_orb_wrap << "  lz_prod=" << lz_prod
                  << "  |diff|=" << lz_wrap_diff << "  seed_k=" << seed_k << "\n\n";
      }

      if (t_ok && lz_wrap_diff > worst_lz_diff) {
        worst_lz_diff = lz_wrap_diff;
        worst_lz_kx = kx;
        worst_lz_ky = ky;
      }
    }
  }

  std::cout << "=== summary ===\n";
  std::cout << "worst apply diff (embedded full space): " << worst_apply << " at K=(" << worst_apply_kx << ","
            << worst_apply_ky << ")\n";
  std::cout << "worst |lz_wrap - lz_prod| (Gram-bridge FTLM): " << worst_lz_diff << " at K=(" << worst_lz_kx << ","
            << worst_lz_ky << ")\n";
  return 0;
}
