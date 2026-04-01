// Diagnostic: for raw orbit columns, verify P_K φ = φ and print a step-by-step trace for the worst column.
#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

#include "ftlm/fock_basis.hpp"
#include "ftlm/symmetry/fermionic_translation.hpp"
#include "ftlm/symmetry/k_basis.hpp"
#include "ftlm/symmetry/orbit_bloch_phi.hpp"
#include "ftlm/symmetry/momentum_sector.hpp"
#include "ftlm/symmetry/orbit.hpp"
#include "ftlm/symmetry/packed_raw_state.hpp"
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

int main() {
  constexpr int Lx = 2;
  constexpr int Ly = 2;
  constexpr int Nup = 2;
  constexpr int Ndn = 2;
  constexpr int kx = 0;
  constexpr int ky = 0;

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

  std::vector<std::vector<std::complex<double>>> Pdense;
  ftlm::symmetry::detail::build_translation_projector_dense(fb, Lx, Ly, K, &Pdense);
  ftlm::symmetry::detail::hermitian_symmetrize_inplace(&Pdense);

  const std::vector<ftlm::symmetry::RawState> seeds = ftlm::symmetry::momentum_phi_seeds(map, K);
  std::vector<std::complex<double>> phi_direct(static_cast<std::size_t>(d) * seeds.size(), 0.0);
  for (std::size_t j = 0; j < seeds.size(); ++j) {
    ftlm::symmetry::detail::fill_phi_orbit_bloch_from_seed(map, K, Lx, Ly, fb, seeds[j],
                                                           phi_direct.data() + j * static_cast<std::size_t>(d));
  }

  double worst_ratio = 0.0;
  int worst_j = 0;
  for (std::size_t j = 0; j < seeds.size(); ++j) {
    std::vector<std::complex<double>> v(static_cast<std::size_t>(d));
    for (int p = 0; p < d; ++p) {
      v[static_cast<std::size_t>(p)] =
          phi_direct[static_cast<std::size_t>(p) + j * static_cast<std::size_t>(d)];
    }
    const auto Pv = apply_dense_P(Pdense, v);
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

  std::cout << std::setprecision(17);
  std::cout << "=== diag_orbit_pk_column_trace: 2x2 (" << Nup << "," << Ndn << ") K=(" << kx << "," << ky << ") ===\n";
  std::cout << "max_j ||P_K phi_j - phi_j||/||phi_j|| = " << worst_ratio << " at j=" << worst_j << "\n\n";

  const ftlm::symmetry::RawState r = seeds[static_cast<std::size_t>(worst_j)];
  std::cout << "Worst column seed r: up=0x" << std::hex << r.up << " dn=0x" << r.dn << std::dec << "\n";

  const ftlm::symmetry::OrbitRecord* orec = nullptr;
  for (const auto& o : map.orbits) {
    if (ftlm::symmetry::pack_raw_state(o.representative) == ftlm::symmetry::pack_raw_state(r) ||
        ftlm::symmetry::pack_raw_state(o.representative) ==
            ftlm::symmetry::pack_raw_state(r)) {  // same orbit: match rep or walk orbits
      orec = &o;
      break;
    }
  }
  if (orec == nullptr) {
    std::cerr << "orbit not found\n";
    return 1;
  }
  std::cout << "orbit_size=" << static_cast<int>(orec->orbit_size) << "  stabilizer_size=" << orec->stabilizer.size()
            << "\n";

  std::complex<double> stab_sum_chi(0.0, 0.0);
  std::complex<double> stab_sum_chi_conj(0.0, 0.0);
  for (const auto& e : orec->stabilizer) {
    const auto ph = ftlm::symmetry::translation_bloch_phase(K, e.first, e.second);
    stab_sum_chi += ph;
    stab_sum_chi_conj += std::conj(ph);
  }
  std::cout << "sum_{R in stab} exp(-i K·R)  (translation_bloch_phase) = " << stab_sum_chi.real() << " + i"
            << stab_sum_chi.imag() << "\n";
  std::cout << "sum_{R in stab} chi*(R) (conj phase) = " << stab_sum_chi_conj.real() << " + i" << stab_sum_chi_conj.imag()
            << "\n";

  const std::size_t stab_sz = ftlm::symmetry::detail::stabilizer_size_of_rep(map, r);
  const double c = ftlm::symmetry::momentum_orbit_normalization_factor_rect(stab_sz, Lx, Ly);
  std::cout << "normalization c = 1/sqrt(|G|*|S|) = " << c << "  (|G|=" << (Lx * Ly) << ")\n\n";

  std::cout << "Per-translation contributions (ex,ey) -> idx, eta, phase=exp(-iK·R), c*eta*conj(phase):\n";
  for (int ey = 0; ey < Ly; ++ey) {
    for (int ex = 0; ex < Lx; ++ex) {
      const ftlm::symmetry::RawState st = ftlm::symmetry::translate_raw_state_rect(r, Lx, Ly, ex, ey);
      const int idx = fb.index_of(static_cast<std::uint64_t>(st.up), static_cast<std::uint64_t>(st.dn));
      const double eta = ftlm::symmetry::fermionic_translation_sign_rect(r, Lx, Ly, ex, ey);
      const auto ph = ftlm::symmetry::translation_bloch_phase(K, ex, ey);
      const std::complex<double> term = c * eta * std::conj(ph);
      std::cout << "  (" << ex << "," << ey << ") idx=" << idx << " eta=" << eta << " phase=(" << ph.real() << ","
                << ph.imag() << ") term=(" << term.real() << "," << term.imag() << ")\n";
    }
  }
  return 0;
}
