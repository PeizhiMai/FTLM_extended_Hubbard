#include "ftlm/fock_basis.hpp"
#include "ftlm/hubbard_params.hpp"
#include "ftlm/symmetry/hubbard_momentum_action.hpp"
#include "ftlm/symmetry/momentum_sector.hpp"
#include "ftlm/symmetry/orbit_bloch_phi.hpp"
#include "ftlm/symmetry/orbit_k_block_basis.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <random>
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

bool run_case(const ftlm::HubbardParams& p, int n_up, int n_dn, ftlm::symmetry::MomentumSector K,
              double* out_max_diff, double* out_min_margin, int* out_dk_orbit, int* out_dk_prod) {
  ftlm::symmetry::HubbardMomentumAction hub(p);
  ftlm::FockBasis fb(p.Lx * p.Ly, n_up, n_dn);
  std::vector<ftlm::symmetry::RawState> universe;
  universe.reserve(static_cast<std::size_t>(fb.dim()));
  for (int i = 0; i < fb.dim(); ++i) {
    universe.push_back(ftlm::symmetry::RawState{static_cast<std::uint16_t>(fb.up_mask(i)),
                                                static_cast<std::uint16_t>(fb.down_mask(i))});
  }
  const auto map = ftlm::symmetry::build_momentum_sector_map_rect(std::move(universe), p.Lx, p.Ly);

  ftlm::symmetry::OrbitKBlockBasis orbit_basis;
  orbit_basis.build(fb, K);
  const int dk_orbit = orbit_basis.dim();
  if (dk_orbit <= 0) {
    return false;
  }

  ftlm::symmetry::MomentumBlockScratch scratch;
  ftlm::symmetry::HubbardMomentumBlock prod_block(hub, map, K, n_up, n_dn, &scratch);
  const int dk_prod = static_cast<int>(prod_block.dim());
  if (dk_prod <= 0) {
    return false;
  }

  const int d = fb.dim();

  std::vector<std::complex<double>> O = build_orbit_full_basis_matrix(fb, map, orbit_basis);

  double max_diff = 0.0;
  std::mt19937 rng(7);
  std::normal_distribution<double> gauss(0.0, 1.0);

  auto run_one = [&](const std::vector<std::complex<double>>& x_orb) {
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
    max_diff = std::max(max_diff, max_abs_diff(w_orb_full, w_prod_full));
  };

  for (int j = 0; j < dk_orbit; ++j) {
    std::vector<std::complex<double>> e(static_cast<std::size_t>(dk_orbit), std::complex<double>(0.0, 0.0));
    e[static_cast<std::size_t>(j)] = std::complex<double>(1.0, 0.0);
    run_one(e);
  }
  {
    std::vector<std::complex<double>> ones(static_cast<std::size_t>(dk_orbit), std::complex<double>(1.0, 0.0));
    run_one(ones);
  }
  for (int r = 0; r < 5; ++r) {
    std::vector<std::complex<double>> xr(static_cast<std::size_t>(dk_orbit));
    for (int i = 0; i < dk_orbit; ++i) {
      xr[static_cast<std::size_t>(i)] = std::complex<double>(gauss(rng), gauss(rng));
    }
    run_one(xr);
  }

  double min_margin = 1e300;
  for (const auto& e : orbit_basis.entries()) {
    min_margin = std::min(min_margin, e.abs_A - orbit_basis.survive_tol());
  }

  *out_max_diff = max_diff;
  *out_min_margin = min_margin;
  *out_dk_orbit = dk_orbit;
  *out_dk_prod = dk_prod;
  return true;
}

}  // namespace

int main() {
  ftlm::HubbardParams p{};
  p.Lx = 2;
  p.Ly = 2;
  p.t = 1.0;
  p.tp = -0.2;
  p.U = 4.0;
  p.V = 0.5;

  double max_all = 0.0;
  double min_margin_all = 1e300;

  struct Case { int nu; int nd; int kx; int ky; };
  const std::vector<Case> cases = {
      {1, 1, 0, 0},
      {1, 1, 1, 0},
      {2, 1, 1, 1},
  };

  for (const auto& c : cases) {
    const ftlm::symmetry::MomentumSector K{c.kx, c.ky, p.Lx, p.Ly};
    double md = 0.0;
    double margin = 0.0;
    int dk_orbit = 0;
    int dk_prod = 0;
    if (!run_case(p, c.nu, c.nd, K, &md, &margin, &dk_orbit, &dk_prod)) {
      std::cerr << "case failed nu=" << c.nu << " nd=" << c.nd << " K=(" << c.kx << "," << c.ky << ")\n";
      return 2;
    }
    std::cout << "case nu=" << c.nu << " nd=" << c.nd << " K=(" << c.kx << "," << c.ky << ")"
              << " dk_orbit=" << dk_orbit << " dk_prod=" << dk_prod
              << " max_abs_full_diff=" << md
              << " min(|A|-tol)=" << margin << "\n";
    max_all = std::max(max_all, md);
    min_margin_all = std::min(min_margin_all, margin);
  }

  // 3x2 regression: sector (2,3), K=(1,0) — O from fill_phi; projection/lift uses same Gram as HubbardMomentumBlock.
  {
    ftlm::HubbardParams p32{};
    p32.Lx = 3;
    p32.Ly = 2;
    p32.t = 1.0;
    p32.tp = -0.35;
    p32.U = 5.75;
    p32.V = 0.9;
    p32.phi_x = 0.0;
    p32.phi_y = 0.0;
    const ftlm::symmetry::MomentumSector K32{1, 0, p32.Lx, p32.Ly};
    double md = 0.0;
    double margin = 0.0;
    int dk_o = 0;
    int dk_p = 0;
    if (!run_case(p32, 2, 3, K32, &md, &margin, &dk_o, &dk_p)) {
      std::cerr << "3x2 regression case failed\n";
      return 2;
    }
    std::cout << "3x2 regression (nu,nd)=(2,3) K=(1,0) dk=" << dk_o << " max_abs_full_diff=" << md
              << " min(|A|-tol)=" << margin << "\n";
    max_all = std::max(max_all, md);
    min_margin_all = std::min(min_margin_all, margin);
    if (md > 1e-10) {
      std::cerr << "3x2 regression apply mismatch\n";
      return 2;
    }
  }

  std::cout << "max_abs_full_diff_all=" << max_all << " min_margin_all=" << min_margin_all << "\n";
  if (max_all > 1e-10) {
    std::cerr << "apply mismatch too large\n";
    return 2;
  }
  return 0;
}

