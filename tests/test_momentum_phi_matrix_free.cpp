// Matrix-free Φ lift/project vs legacy dense Φ; projected H columns vs dense reference.
#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "ftlm/fock_basis.hpp"
#include "ftlm/hubbard_hamiltonian.hpp"
#include "ftlm/hubbard_params.hpp"
#include "ftlm/symmetry/hubbard_momentum_action.hpp"
#include "ftlm/symmetry/orbit_bloch_phi.hpp"

namespace {

using ftlm::FockBasis;
using ftlm::HubbardParams;
using ftlm::symmetry::HubbardMomentumAction;
using ftlm::symmetry::MomentumPhiGramBasis;
using ftlm::symmetry::MomentumSector;
using ftlm::symmetry::RawState;
using ftlm::symmetry::build_momentum_phi_orbit_orthonormal;
using ftlm::symmetry::build_momentum_sector_map_rect;

constexpr double tol_lift = 1e-11;
constexpr double tol_proj = 1e-11;
constexpr double tol_Hcol = 1e-10;

void dense_lift(const std::vector<std::complex<double>>& phi_cm, int d, std::size_t dk,
                const std::complex<double>* y_block, std::complex<double>* x_full) {
  std::fill(x_full, x_full + d, std::complex<double>(0.0, 0.0));
  for (std::size_t j = 0; j < dk; ++j) {
    for (int p = 0; p < d; ++p) {
      x_full[p] += phi_cm[static_cast<std::size_t>(p) + j * static_cast<std::size_t>(d)] * y_block[j];
    }
  }
}

void dense_project(const std::vector<std::complex<double>>& phi_cm, int d, std::size_t dk,
                   const std::complex<double>* x_full, std::complex<double>* y_block) {
  for (std::size_t j = 0; j < dk; ++j) {
    std::complex<double> s(0.0, 0.0);
    for (int p = 0; p < d; ++p) {
      s += std::conj(phi_cm[static_cast<std::size_t>(p) + j * static_cast<std::size_t>(d)]) * x_full[p];
    }
    y_block[j] = s;
  }
}

}  // namespace

int main() {
  const int Lx = 2;
  const int Ly = 2;
  HubbardParams p;
  p.Lx = Lx;
  p.Ly = Ly;
  p.t = 1.0;
  p.tp = -0.35;
  p.U = 5.75;
  p.V = 0.9;
  p.phi_x = 0.0;
  p.phi_y = 0.0;

  const int nup = 1;
  const int ndn = 1;
  FockBasis fb(Lx * Ly, nup, ndn);
  const int d = fb.dim();
  std::vector<RawState> universe;
  universe.reserve(static_cast<std::size_t>(d));
  for (int i = 0; i < d; ++i) {
    universe.push_back(RawState{static_cast<std::uint16_t>(fb.up_mask(i)), static_cast<std::uint16_t>(fb.down_mask(i))});
  }
  const auto map = build_momentum_sector_map_rect(std::move(universe), Lx, Ly);
  const MomentumSector K{0, 0, Lx, Ly};

  HubbardMomentumAction hub(p);

  std::vector<std::complex<double>> phi_dense;
  std::size_t dk_dense = 0;
  build_momentum_phi_orbit_orthonormal(map, K, Lx, Ly, fb, &phi_dense, &dk_dense);

  MomentumPhiGramBasis gram;
  if (!MomentumPhiGramBasis::build(map, K, Lx, Ly, fb, &gram)) {
    std::cerr << "matrix-free build failed\n";
    return 2;
  }
  if (gram.k_out != dk_dense) {
    std::cerr << "dim mismatch dense=" << dk_dense << " mf=" << gram.k_out << "\n";
    return 2;
  }

  std::mt19937 rng(42);
  std::normal_distribution<double> gauss(0.0, 1.0);

  for (int trial = 0; trial < 5; ++trial) {
    std::vector<std::complex<double>> yb(dk_dense);
    for (std::size_t j = 0; j < dk_dense; ++j) {
      yb[j] = {gauss(rng), gauss(rng)};
    }
    std::vector<std::complex<double>> xf_dense(static_cast<std::size_t>(d));
    std::vector<std::complex<double>> xf_mf(static_cast<std::size_t>(d));
    dense_lift(phi_dense, d, dk_dense, yb.data(), xf_dense.data());
    gram.lift_full_from_block(map, K, fb, yb.data(), xf_mf.data());
    double max_err = 0.0;
    for (int p = 0; p < d; ++p) {
      max_err = std::max(max_err, std::abs(xf_dense[static_cast<std::size_t>(p)] - xf_mf[static_cast<std::size_t>(p)]));
    }
    if (max_err > tol_lift) {
      std::cerr << "lift mismatch max_err=" << max_err << "\n";
      return 2;
    }
  }

  for (int trial = 0; trial < 5; ++trial) {
    std::vector<std::complex<double>> xf(static_cast<std::size_t>(d));
    for (int p = 0; p < d; ++p) {
      xf[static_cast<std::size_t>(p)] = {gauss(rng), gauss(rng)};
    }
    std::vector<std::complex<double>> yd(dk_dense), ym(dk_dense);
    dense_project(phi_dense, d, dk_dense, xf.data(), yd.data());
    gram.project_block_from_full(map, K, fb, xf.data(), ym.data());
    double max_err = 0.0;
    for (std::size_t j = 0; j < dk_dense; ++j) {
      max_err = std::max(max_err, std::abs(yd[j] - ym[j]));
    }
    if (max_err > tol_proj) {
      std::cerr << "project mismatch max_err=" << max_err << "\n";
      return 2;
    }
  }

  for (std::size_t a = 0; a < dk_dense; ++a) {
    std::vector<std::complex<double>> ea(dk_dense, std::complex<double>(0.0, 0.0));
    ea[a] = {1.0, 0.0};
    std::vector<std::complex<double>> full(static_cast<std::size_t>(d));
    gram.lift_full_from_block(map, K, fb, ea.data(), full.data());
    std::vector<std::complex<double>> out(dk_dense);
    gram.project_block_from_full(map, K, fb, full.data(), out.data());
    for (std::size_t b = 0; b < dk_dense; ++b) {
      const double expected = (a == b) ? 1.0 : 0.0;
      const double err = std::abs(out[b] - std::complex<double>(expected, 0.0));
      if (err > 1e-10) {
        std::cerr << "orthonormal roundtrip mismatch at a=" << a << " b=" << b << " err=" << err << "\n";
        return 2;
      }
    }
  }

  // Column j of Φ†HΦ: y = Φ† H Φ e_j
  std::vector<std::complex<double>> v(static_cast<std::size_t>(d));
  std::vector<std::complex<double>> Hv(static_cast<std::size_t>(d));
  std::vector<std::complex<double>> y_mf(dk_dense);
  std::vector<std::complex<double>> y_ds(dk_dense);
  for (std::size_t j = 0; j < dk_dense; ++j) {
    std::vector<std::complex<double>> ej(dk_dense, std::complex<double>(0.0, 0.0));
    ej[j] = {1.0, 0.0};
    dense_lift(phi_dense, d, dk_dense, ej.data(), v.data());
    ftlm::apply_extended_hubbard(p, fb, hub.hoppings, hub.nn_pairs, v.data(), Hv.data());
    dense_project(phi_dense, d, dk_dense, Hv.data(), y_ds.data());

    gram.lift_full_from_block(map, K, fb, ej.data(), v.data());
    ftlm::apply_extended_hubbard(p, fb, hub.hoppings, hub.nn_pairs, v.data(), Hv.data());
    gram.project_block_from_full(map, K, fb, Hv.data(), y_mf.data());

    for (std::size_t i = 0; i < dk_dense; ++i) {
      const double err = std::abs(y_ds[i] - y_mf[i]);
      if (err > tol_Hcol) {
        std::cerr << "H column mismatch i=" << i << " j=" << j << " err=" << err << "\n";
        return 2;
      }
    }
  }

  const std::size_t dense_bytes = phi_dense.size() * sizeof(std::complex<double>);
  const std::size_t gram_bytes = gram.storage_bytes();
  // Dense Φ is O(d·d_k); matrix-free keeps a k_in×k_out slice of V (no d_full × d_k array).

  std::cout << "ok matrix-free vs dense: dk=" << dk_dense << " k_in=" << gram.k_in << " dense_phi_bytes=" << dense_bytes
            << " gram_storage_bytes=" << gram_bytes << "\n";
  return 0;
}
