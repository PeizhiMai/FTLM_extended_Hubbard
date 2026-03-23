#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

#include "ftlm/symmetry/k_basis.hpp"
#include "ftlm/symmetry/momentum_sector.hpp"
#include "ftlm/symmetry/orbit.hpp"
#include "ftlm/symmetry/raw_state.hpp"

namespace {

using ftlm::symmetry::KBasis;
using ftlm::symmetry::MomentumSector;
using ftlm::symmetry::MomentumSectorMap;
using ftlm::symmetry::RawState;
using ftlm::symmetry::build_momentum_sector_map;
using ftlm::symmetry::canonical_raw_state;
using ftlm::symmetry::momentum_character_trivial;
using ftlm::symmetry::momentum_orbit_normalization_factor;
using ftlm::symmetry::translation_bloch_phase;
using ftlm::symmetry::translation_stabilizer;

void test_momentum_compatibility_explicit_stabilizer() {
  // Stabilizer contains (2,0): need kx*2 ≡ 0 (mod 4) ⇒ kx even.
  std::vector<std::pair<std::int8_t, std::int8_t>> stab = {{0, 0}, {2, 0}};
  for (int kx = 0; kx < 4; ++kx) {
    for (int ky = 0; ky < 4; ++ky) {
      bool all_triv = true;
      for (const auto& e : stab) {
        if (!momentum_character_trivial(kx, ky, e.first, e.second)) {
          all_triv = false;
        }
      }
      const bool even_kx = (kx % 2) == 0;
      assert(all_triv == even_kx);
    }
  }
}

void test_translation_phase_matches_triviality() {
  const MomentumSector K{1, 2};
  for (int dy = 0; dy < 4; ++dy) {
    for (int dx = 0; dx < 4; ++dx) {
      const auto z = translation_bloch_phase(K, dx, dy);
      const bool triv = momentum_character_trivial(K.kx, K.ky, dx, dy);
      if (triv) {
        assert(std::abs(z.real() - 1.0) < 1e-14 && std::abs(z.imag()) < 1e-14);
      } else {
        assert(std::abs(std::abs(z) - 1.0) < 1e-14);
      }
    }
  }
}

void test_momentum_sector_compatibility_bit() {
  std::vector<RawState> seeds;
  seeds.push_back({0, 0});
  seeds.push_back({1u << 0, 0});
  const MomentumSectorMap map = build_momentum_sector_map(std::move(seeds));
  const MomentumSector k0{0, 0};
  const MomentumSector k1{1, 0};
  assert(k0.is_compatible(map.orbits[0]));
  assert(!k1.is_compatible(map.orbits[0]));
  assert(k1.is_compatible(map.orbits[1]));
}

void test_k_basis_dimensions() {
  std::vector<RawState> seeds;
  seeds.push_back({0, 0});
  seeds.push_back({1u << 0, 0});
  seeds.push_back({0x000Fu, 0});
  const MomentumSectorMap map = build_momentum_sector_map(std::move(seeds));

  const KBasis b00 = KBasis::build(map, MomentumSector{0, 0});
  assert(b00.dim() == 3u);
  const RawState vac_rep = canonical_raw_state(RawState{0, 0});
  assert(b00.index_of(vac_rep) != KBasis::npos);

  const KBasis b10 = KBasis::build(map, MomentumSector{1, 0});
  assert(b10.dim() == 1u);
  assert(b10.index_of(canonical_raw_state({1u << 0, 0})) != KBasis::npos);

  // Vacuum is Γ-only (stabilizer = full group); row stripe needs kx ≡ 0 (mod 4).
  const KBasis b01 = KBasis::build(map, MomentumSector{0, 1});
  assert(b01.dim() == 2u);
  assert(b01.index_of(vac_rep) == KBasis::npos);
}

void test_normalization_factor_squared_times_sum_weights() {
  RawState row{0x000Fu, 0};
  const auto stab = translation_stabilizer(canonical_raw_state(row));
  const double f = momentum_orbit_normalization_factor(stab.size());
  assert(std::abs(f - 1.0 / std::sqrt(16.0 * static_cast<double>(stab.size()))) < 1e-14);
}

}  // namespace

int main() {
  test_momentum_compatibility_explicit_stabilizer();
  test_translation_phase_matches_triviality();
  test_momentum_sector_compatibility_bit();
  test_k_basis_dimensions();
  test_normalization_factor_squared_times_sum_weights();
  std::cout << "test_symmetry_stage3 ok\n";
  return 0;
}
