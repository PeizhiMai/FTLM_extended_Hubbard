#include "ftlm/symmetry/hubbard_momentum_action.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "ftlm/fock_basis.hpp"
#include "ftlm/hubbard_hamiltonian.hpp"
#include "ftlm/symmetry/k_basis.hpp"
#include "ftlm/symmetry/translation_projector_dense.hpp"
#include "ftlm/symmetry/raw_state.hpp"

namespace ftlm {
namespace symmetry {
namespace {

constexpr int kMaxDenseProjectorDim = 512;

}  // namespace

HubbardMomentumAction::HubbardMomentumAction(HubbardParams p) : params(std::move(p)) {
  if (params.Lx <= 0 || params.Ly <= 0 || params.Lx * params.Ly > 16) {
    throw std::invalid_argument("HubbardMomentumAction: requires 1 <= Lx*Ly <= 16");
  }
  lat.Lx = params.Lx;
  lat.Ly = params.Ly;
  build_hubbard_geometry(params, lat, &hoppings, &nn_pairs);
}

std::size_t HubbardMomentumAction::momentum_block_dim(const MomentumSectorMap& orbit_map, MomentumSector K, int n_up,
                                                      int n_dn) const {
  const int lx = params.Lx;
  const int ly = params.Ly;
  FockBasis fb(lx * ly, n_up, n_dn);
  const int d_full = fb.dim();
  if (d_full <= kMaxDenseProjectorDim) {
    std::vector<std::complex<double>> phi;
    std::size_t dk = 0;
    detail::build_phi_from_translation_projector_dense(fb, lx, ly, K, &phi, &dk);
    return dk;
  }
  const KBasis kb = KBasis::build(orbit_map, K);
  return kb.dim();
}

void HubbardMomentumAction::apply(const MomentumSectorMap& orbit_map, MomentumSector K, int n_up, int n_dn,
                                  const std::complex<double>* x, std::complex<double>* y) const {
  const int lx = params.Lx;
  const int ly = params.Ly;
  FockBasis fb(lx * ly, n_up, n_dn);
  const int d_full = fb.dim();

  std::vector<std::complex<double>> phi;
  std::size_t dk = 0;
  if (d_full <= kMaxDenseProjectorDim) {
    detail::build_phi_from_translation_projector_dense(fb, lx, ly, K, &phi, &dk);
  } else {
    const KBasis basis = KBasis::build(orbit_map, K);
    dk = basis.dim();
    phi.assign(static_cast<std::size_t>(d_full) * dk, std::complex<double>(0.0, 0.0));
    for (std::size_t j = 0; j < dk; ++j) {
      detail::fill_phi_orbit_bloch_bitonly(orbit_map, K, basis, lx, ly, fb, static_cast<int>(j),
                                             phi.data() + j * static_cast<std::size_t>(d_full));
    }
  }

  std::fill(y, y + dk, std::complex<double>(0.0, 0.0));
  if (dk == 0) {
    return;
  }

  std::vector<std::complex<double>> vin(static_cast<std::size_t>(d_full));
  std::vector<std::complex<double>> wout(static_cast<std::size_t>(d_full));
  for (int p = 0; p < d_full; ++p) {
    std::complex<double> acc(0.0, 0.0);
    for (std::size_t j = 0; j < dk; ++j) {
      acc += phi[static_cast<std::size_t>(p) + j * static_cast<std::size_t>(d_full)] * x[j];
    }
    vin[static_cast<std::size_t>(p)] = acc;
  }

  apply_extended_hubbard(params, fb, hoppings, nn_pairs, vin.data(), wout.data());

  for (std::size_t i = 0; i < dk; ++i) {
    std::complex<double> acc(0.0, 0.0);
    for (int p = 0; p < d_full; ++p) {
      acc += std::conj(phi[static_cast<std::size_t>(p) + i * static_cast<std::size_t>(d_full)]) *
             wout[static_cast<std::size_t>(p)];
    }
    y[i] = acc;
  }
}

}  // namespace symmetry
}  // namespace ftlm
