// Step 1: OrbitKBlockBasis metadata — fermionic stabilizer character, surviving filter, raw→rep map.
#include "ftlm/fock_basis.hpp"
#include "ftlm/symmetry/k_basis.hpp"
#include "ftlm/symmetry/momentum_sector.hpp"
#include "ftlm/symmetry/orbit.hpp"
#include "ftlm/symmetry/orbit_k_block_basis.hpp"
#include "ftlm/symmetry/packed_raw_state.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {

bool approx_eq(double a, double b, double tol) { return std::abs(a - b) <= tol * (1.0 + std::max(std::abs(a), std::abs(b))); }

}  // namespace

int main() {
  constexpr double tol = 1e-11;

  // 2x2 torus, half-filling sector: exercise multiple orbit sizes / stabilizers.
  const int lx = 2;
  const int ly = 2;
  ftlm::FockBasis fb(4, 2, 2);

  ftlm::symmetry::MomentumSector K0{0, 0, lx, ly};
  ftlm::symmetry::OrbitKBlockBasis b0;
  if (!b0.build(fb, K0)) {
    std::cerr << "build failed\n";
    return 2;
  }
  if (b0.dim() <= 0) {
    std::cerr << "unexpected dim=0 at Gamma\n";
    return 2;
  }

  std::size_t sum_surviving_sizes = 0;
  int n_map_ok = 0;
  for (const auto& e : b0.entries()) {
    sum_surviving_sizes += e.orbit_size;
    // |A| should match |Stab| in magnitude for surviving orbits on tiny lattices (noncanceling cases).
    if (!approx_eq(e.abs_A, static_cast<double>(e.stabilizer_size), 1e-9)) {
      std::cerr << "|A| vs |Stab| mismatch: abs_A=" << e.abs_A << " stab=" << e.stabilizer_size << "\n";
      return 2;
    }
  }

  for (int i = 0; i < fb.dim(); ++i) {
    const ftlm::symmetry::RawState s{static_cast<std::uint16_t>(fb.up_mask(i)),
                                     static_cast<std::uint16_t>(fb.down_mask(i))};
    int ri = 0;
    int dx = 0;
    int dy = 0;
    double eta = 0.0;
    if (b0.map_raw_to_rep(s, &ri, &dx, &dy, &eta)) {
      ++n_map_ok;
      if (s.up == b0.entries()[static_cast<std::size_t>(ri)].rep.up &&
          s.dn == b0.entries()[static_cast<std::size_t>(ri)].rep.dn) {
        if (dx != 0 || dy != 0 || std::abs(eta - 1.0) > tol) {
          std::cerr << "rep self-map wrong\n";
          return 2;
        }
      }
    }
  }
  if (static_cast<std::size_t>(n_map_ok) != sum_surviving_sizes) {
    std::cerr << "map count " << n_map_ok << " vs sum orbit sizes " << sum_surviving_sizes << "\n";
    return 2;
  }

  // KBasis (bit mask) vs fermionic basis: dimensions may differ when fermionic sum cancels.
  std::vector<ftlm::symmetry::RawState> universe;
  for (int i = 0; i < fb.dim(); ++i) {
    universe.push_back(ftlm::symmetry::RawState{static_cast<std::uint16_t>(fb.up_mask(i)),
                                                static_cast<std::uint16_t>(fb.down_mask(i))});
  }
  const ftlm::symmetry::MomentumSectorMap map =
      ftlm::symmetry::build_momentum_sector_map_rect(std::move(universe), lx, ly);
  const auto kb = ftlm::symmetry::KBasis::build(map, K0);
  if (static_cast<int>(kb.dim()) < b0.dim()) {
    std::cerr << "KBasis dim should be >= fermionic dim at Gamma (got kb=" << kb.dim() << " b0=" << b0.dim() << ")\n";
    return 2;
  }

  // 3x2 smoke: build at one nontrivial K without crashing.
  ftlm::FockBasis fb32(6, 3, 3);
  ftlm::symmetry::MomentumSector K1{1, 0, 3, 2};
  ftlm::symmetry::OrbitKBlockBasis b32;
  if (!b32.build(fb32, K1)) {
    std::cerr << "3x2 build failed\n";
    return 2;
  }
  if (b32.dim() <= 0) {
    std::cerr << "3x2 unexpected dim=0\n";
    return 2;
  }

  std::cout << "ok orbit_k_block_basis dim(2x2 Gamma)=" << b0.dim() << " kb_dim=" << kb.dim()
            << " dim(3x2 K=(1,0))=" << b32.dim() << "\n";
  return 0;
}

