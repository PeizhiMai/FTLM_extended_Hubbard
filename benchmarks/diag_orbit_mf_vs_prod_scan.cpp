// Diagnostic: compare orbit-matrix-free apply vs production HubbardMomentumBlock on the same K block,
// in full Fock space (same convention as tests/test_orbit_k_block_apply_vs_production.cpp).
// Default scan: 3x2, mixed-spin sectors only, dim_full in [dim_full_min, dim_full_max] (defaults 257..512).
// Stops at the first block with dk mismatch or |Δy|_inf > fail_tol (default 1e-8) on basis or random probes.

#include "ftlm/fock_basis.hpp"
#include "ftlm/hubbard_hamiltonian.hpp"
#include "ftlm/hubbard_params.hpp"
#include "ftlm/lattice.hpp"
#include "ftlm/symmetry/hubbard_momentum_action.hpp"
#include "ftlm/symmetry/momentum_sector.hpp"
#include "ftlm/symmetry/orbit_bloch_phi.hpp"
#include "ftlm/symmetry/orbit_k_block_basis.hpp"
#include "ftlm/symmetry/raw_state.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

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

std::vector<std::complex<double>> build_orbit_full_basis_matrix(
    const ftlm::FockBasis& fb, const ftlm::symmetry::MomentumSectorMap& orbit_map,
    const ftlm::symmetry::OrbitKBlockBasis& basis) {
  const int d = fb.dim();
  const int dk = basis.dim();
  std::vector<std::complex<double>> O(static_cast<std::size_t>(d * dk), std::complex<double>(0.0, 0.0));
  for (int j = 0; j < dk; ++j) {
    const auto& e = basis.entries()[static_cast<std::size_t>(j)];
    ftlm::symmetry::detail::fill_phi_orbit_bloch_from_seed(orbit_map, basis.K, basis.lx, basis.ly, fb, e.rep,
                                                            O.data() + static_cast<std::size_t>(j * d));
  }
  return O;
}

int parse_int_arg(int argc, char** argv, const char* key, int default_value) {
  const std::string prefix = std::string("--") + key + "=";
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    if (std::strncmp(a, prefix.c_str(), prefix.size()) == 0) {
      return std::atoi(a + prefix.size());
    }
  }
  return default_value;
}

double parse_double_arg(int argc, char** argv, const char* key, double default_value) {
  const std::string prefix = std::string("--") + key + "=";
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    if (std::strncmp(a, prefix.c_str(), prefix.size()) == 0) {
      return std::strtod(a + prefix.size(), nullptr);
    }
  }
  return default_value;
}

struct ScanStats {
  double max_basis = 0.0;
  double max_rand = 0.0;
  int dk_orbit = 0;
  int dk_prod = 0;
  bool dk_phi_match = true;
  int dk_phi = 0;
  double min_abs_A = std::numeric_limits<double>::infinity();
  double min_margin_A = std::numeric_limits<double>::infinity();
  int max_stabilizer_size = 0;
  std::size_t n_orbits_surviving = 0;
};

bool scan_block(const ftlm::HubbardParams& p, const ftlm::symmetry::HubbardMomentumAction& hub,
                const ftlm::symmetry::MomentumSectorMap& map, int nu, int nd, ftlm::symmetry::MomentumSector K,
                const ftlm::FockBasis& fb, ftlm::symmetry::MomentumBlockScratch* scratch, int n_rand,
                ScanStats* out) {
  ftlm::symmetry::OrbitKBlockBasis orbit_basis;
  orbit_basis.build(fb, K);
  const int dk_orbit = orbit_basis.dim();
  if (dk_orbit <= 0) {
    return false;
  }

  ftlm::symmetry::HubbardMomentumBlock prod_block(hub, map, K, nu, nd, scratch);
  const int dk_prod = static_cast<int>(prod_block.dim());
  if (dk_prod <= 0) {
    return false;
  }

  const int d = fb.dim();

  out->dk_orbit = dk_orbit;
  out->dk_prod = dk_prod;
  out->dk_phi = static_cast<int>(dk_prod);
  out->dk_phi_match = true;

  const double tol_surv = orbit_basis.survive_tol();
  out->n_orbits_surviving = orbit_basis.entries().size();
  for (const auto& e : orbit_basis.entries()) {
    out->min_abs_A = std::min(out->min_abs_A, e.abs_A);
    out->min_margin_A = std::min(out->min_margin_A, e.abs_A - tol_surv);
    out->max_stabilizer_size = std::max(out->max_stabilizer_size, static_cast<int>(e.stabilizer_size));
  }

  if (dk_orbit != dk_prod) {
    return true;
  }

  std::vector<std::complex<double>> O = build_orbit_full_basis_matrix(fb, map, orbit_basis);

  auto run_one = [&](const std::vector<std::complex<double>>& x_orb, double* max_here) {
    std::vector<std::complex<double>> y_orb(static_cast<std::size_t>(dk_orbit));
    ftlm::symmetry::apply_orbit_k_block(p, fb, orbit_basis, hub.hoppings, hub.nn_pairs, x_orb.data(), y_orb.data());

    std::vector<std::complex<double>> v_full;
    matvec_colmajor(d, dk_orbit, O, x_orb, &v_full);
    std::vector<std::complex<double>> x_prod(static_cast<std::size_t>(dk_prod));
    prod_block.project_full_to_block(v_full.data(), x_prod.data());

    std::vector<std::complex<double>> y_prod(static_cast<std::size_t>(dk_prod));
    prod_block.apply(x_prod.data(), y_prod.data());

    std::vector<std::complex<double>> w_orb_full;
    matvec_colmajor(d, dk_orbit, O, y_orb, &w_orb_full);
    std::vector<std::complex<double>> w_prod_full(static_cast<std::size_t>(d));
    prod_block.lift_block_to_full(y_prod.data(), w_prod_full.data());
    *max_here = std::max(*max_here, max_abs_diff(w_orb_full, w_prod_full));
  };

  for (int j = 0; j < dk_orbit; ++j) {
    std::vector<std::complex<double>> e(static_cast<std::size_t>(dk_orbit), std::complex<double>(0.0, 0.0));
    e[static_cast<std::size_t>(j)] = std::complex<double>(1.0, 0.0);
    run_one(e, &out->max_basis);
  }
  std::mt19937 rng(13 + nu * 17 + nd * 31 + K.kx * 3 + K.ky * 5);
  std::normal_distribution<double> gauss(0.0, 1.0);
  for (int r = 0; r < n_rand; ++r) {
    std::vector<std::complex<double>> xr(static_cast<std::size_t>(dk_orbit));
    for (int i = 0; i < dk_orbit; ++i) {
      xr[static_cast<std::size_t>(i)] = std::complex<double>(gauss(rng), gauss(rng));
    }
    run_one(xr, &out->max_rand);
  }

  return true;
}

}  // namespace

int main(int argc, char** argv) {
  const int Lx = parse_int_arg(argc, argv, "Lx", 3);
  const int Ly = parse_int_arg(argc, argv, "Ly", 2);
  const int dim_full_min = parse_int_arg(argc, argv, "dim-full-min", 257);
  const int dim_full_max = parse_int_arg(argc, argv, "dim-full-max", 512);
  const int n_rand = parse_int_arg(argc, argv, "random-probes", 4);
  const double fail_tol = parse_double_arg(argc, argv, "fail-tol", 1e-8);
  const int stop_after_first = parse_int_arg(argc, argv, "stop-after-first", 1);

  ftlm::HubbardParams p{};
  p.Lx = Lx;
  p.Ly = Ly;
  p.t = 1.0;
  p.tp = -0.35;
  p.U = 5.75;
  p.V = 0.9;
  p.phi_x = 0.0;
  p.phi_y = 0.0;

  ftlm::RectLattice lat{p.Lx, p.Ly};
  std::vector<ftlm::SpinfulHopping> hops;
  std::vector<ftlm::NearestPair> pairs;
  ftlm::build_hubbard_geometry(p, lat, &hops, &pairs);
  ftlm::symmetry::HubbardMomentumAction hub(p);

  const int n_sites = Lx * Ly;

  std::cout << "[diag_orbit_mf_vs_prod_scan] Lx=" << Lx << " Ly=" << Ly << " dim_full in [" << dim_full_min << ", "
            << dim_full_max << "]  random_probes=" << n_rand << " fail_tol=" << fail_tol << "\n";

  for (int nu = 0; nu <= n_sites; ++nu) {
    for (int nd = 0; nd <= n_sites; ++nd) {
      const bool orbit_mf_sector_safe = (nu > 0 && nd > 0 && nu < n_sites && nd < n_sites);
      if (!orbit_mf_sector_safe) {
        continue;
      }
      ftlm::FockBasis fb(n_sites, nu, nd);
      const int dim_full = fb.dim();
      if (dim_full < dim_full_min || dim_full > dim_full_max) {
        continue;
      }

      std::vector<ftlm::symmetry::RawState> universe;
      universe.reserve(static_cast<std::size_t>(dim_full));
      for (int i = 0; i < dim_full; ++i) {
        universe.push_back(ftlm::symmetry::RawState{static_cast<std::uint16_t>(fb.up_mask(i)),
                                                    static_cast<std::uint16_t>(fb.down_mask(i))});
      }
      const auto map = ftlm::symmetry::build_momentum_sector_map_rect(std::move(universe), Lx, Ly);
      ftlm::symmetry::MomentumBlockScratch scratch;

      for (int ky = 0; ky < Ly; ++ky) {
        for (int kx = 0; kx < Lx; ++kx) {
          const ftlm::symmetry::MomentumSector K{kx, ky, Lx, Ly};
          ScanStats st{};
          if (!scan_block(p, hub, map, nu, nd, K, fb, &scratch, n_rand, &st)) {
            continue;
          }

          const bool dk_mismatch = (st.dk_orbit != st.dk_prod);
          const bool phi_mismatch = !st.dk_phi_match;
          const bool bad = dk_mismatch || phi_mismatch || st.max_basis > fail_tol || st.max_rand > fail_tol;

          std::cout << "block nu=" << nu << " nd=" << nd << " K=(" << kx << "," << ky << ")"
                    << " dim_full=" << dim_full << " dk_orbit=" << st.dk_orbit << " dk_prod=" << st.dk_prod
                    << " dk_phi=" << st.dk_phi << " max|dy|_basis=" << st.max_basis << " max|dy|_rand=" << st.max_rand
                    << " min|A|=" << st.min_abs_A << " min(|A|-tol)=" << st.min_margin_A
                    << " max_stab=" << st.max_stabilizer_size << " n_surv=" << st.n_orbits_surviving
                    << (bad ? "  <-- BAD" : "  ok") << "\n";

          if (bad && stop_after_first != 0) {
            std::cout << "\n[diag_orbit_mf_vs_prod_scan] FIRST BAD BLOCK (see line above).\n";
            if (st.max_basis <= fail_tol && st.max_rand > fail_tol) {
              std::cout << "  Note: basis probes OK, random probes fail -> likely numerical or subspace mismatch.\n";
            } else if (st.max_basis > fail_tol) {
              std::cout << "  Note: basis probes already fail -> mismatch on standard orbit basis vectors.\n";
            }
            if (dk_mismatch) {
              std::cout << "  Note: dk_orbit != dk_prod -> different effective subspace dimension.\n";
            }
            if (phi_mismatch) {
              std::cout << "  Note: dk_phi != dk_prod (Gram orbit-orthonormal vs HubbardMomentumBlock dim).\n";
            }
            return 1;
          }
        }
      }
    }
  }

  std::cout << "[diag_orbit_mf_vs_prod_scan] no failing block in range (or no blocks scanned).\n";
  return 0;
}
