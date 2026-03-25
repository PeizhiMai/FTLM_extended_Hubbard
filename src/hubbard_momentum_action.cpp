#include "ftlm/symmetry/hubbard_momentum_action.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <vector>

#include "ftlm/hubbard_hamiltonian.hpp"
#include "ftlm/symmetry/orbit_bloch_phi.hpp"
#include "ftlm/symmetry/raw_state.hpp"

namespace ftlm {
namespace symmetry {

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
  MomentumPhiGramBasis gram;
  if (!MomentumPhiGramBasis::build(orbit_map, K, lx, ly, fb, &gram, nullptr)) {
    return 0;
  }
  return gram.k_out;
}

HubbardMomentumBlock::HubbardMomentumBlock(const HubbardMomentumAction& hub, const MomentumSectorMap& orbit_map,
                                           MomentumSector K, int n_up, int n_dn, MomentumBlockScratch* scratch)
    : hub_(&hub),
      orbit_map_(&orbit_map),
      K_(K),
      scratch_(scratch),
      fb_(hub.params.Lx * hub.params.Ly, n_up, n_dn) {
  const int lx = hub.params.Lx;
  const int ly = hub.params.Ly;
  d_full_ = fb_.dim();
  if (!MomentumPhiGramBasis::build(orbit_map, K, lx, ly, fb_, &gram_, scratch_ ? &scratch_->zheev : nullptr)) {
    dk_ = 0;
  } else {
    dk_ = gram_.k_out;
  }
  if (std::getenv("FTLM_MEM_REPORT_DETAIL")) {
    std::cerr << "[mem] HubbardMomentumBlock d_full=" << d_full_ << " d_k=" << dk_
              << " gram_storage_bytes=" << phi_bytes() << "\n";
  }
}

HubbardMomentumBlock::~HubbardMomentumBlock() = default;

void HubbardMomentumBlock::apply(const std::complex<double>* x, std::complex<double>* y) const {
  if (dk_ == 0) {
    return;
  }
  std::fill(y, y + dk_, std::complex<double>(0.0, 0.0));

  std::vector<std::complex<double>>* pvin = nullptr;
  std::vector<std::complex<double>>* pwout = nullptr;
  if (scratch_ != nullptr) {
    scratch_->ensure_d_full(d_full_);
    scratch_->gram_apply.ensure(gram_.k_in, d_full_);
    pvin = &scratch_->vin;
    pwout = &scratch_->wout;
  } else {
    thread_local std::vector<std::complex<double>> tl_vin;
    thread_local std::vector<std::complex<double>> tl_wout;
    tl_vin.resize(static_cast<std::size_t>(d_full_));
    tl_wout.resize(static_cast<std::size_t>(d_full_));
    pvin = &tl_vin;
    pwout = &tl_wout;
  }

  MomentumPhiGramApplyScratch* gram_scr = scratch_ != nullptr ? &scratch_->gram_apply : nullptr;
  gram_.lift_full_from_block(*orbit_map_, K_, fb_, x, pvin->data(), gram_scr);

  apply_extended_hubbard(hub_->params, fb_, hub_->hoppings, hub_->nn_pairs, pvin->data(), pwout->data());

  gram_.project_block_from_full(*orbit_map_, K_, fb_, pwout->data(), y, gram_scr);
}

void HubbardMomentumAction::apply(const MomentumSectorMap& orbit_map, MomentumSector K, int n_up, int n_dn,
                                  const std::complex<double>* x, std::complex<double>* y) const {
  HubbardMomentumBlock block(*this, orbit_map, K, n_up, n_dn, nullptr);
  block.apply(x, y);
}

}  // namespace symmetry
}  // namespace ftlm
