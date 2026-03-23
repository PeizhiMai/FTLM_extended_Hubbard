#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

#include "ftlm/symmetry/momentum_sector.hpp"
#include "ftlm/symmetry/orbit.hpp"
#include "ftlm/symmetry/packed_raw_state.hpp"
#include "ftlm/symmetry/raw_state.hpp"
#include "ftlm/symmetry/translation_handler.hpp"

namespace {

using ftlm::symmetry::MomentumSectorMap;
using ftlm::symmetry::OrbitInfo;
using ftlm::symmetry::OrbitRecord;
using ftlm::symmetry::RawState;
using ftlm::symmetry::TranslationHandler;
using ftlm::symmetry::build_momentum_sector_map;
using ftlm::symmetry::canonical_raw_state;
using ftlm::symmetry::check_orbit_stabilizer_count_identity;
using ftlm::symmetry::enumerate_translation_orbit;
using ftlm::symmetry::make_orbit_info;
using ftlm::symmetry::pack_raw_state;
using ftlm::symmetry::popcount_momentum_mask;
using ftlm::symmetry::primitive_period_tx;
using ftlm::symmetry::primitive_period_ty;
using ftlm::symmetry::translate_raw_state;
using ftlm::symmetry::translation_stabilizer;

void test_single_up_orbit() {
  RawState s{1u << 0, 0};
  std::vector<RawState> orb;
  enumerate_translation_orbit(s, &orb);
  assert(orb.size() == 16u);
  const RawState c = canonical_raw_state(s);
  assert(c == canonical_raw_state(c));
  const auto stab = translation_stabilizer(c);
  assert(stab.size() == 1u);
  assert(stab[0].first == 0 && stab[0].second == 0);
  assert(primitive_period_tx(c) == 4u && primitive_period_ty(c) == 4u);
  const OrbitInfo oi = make_orbit_info(s);
  assert(oi.representative == c && oi.orbit_size == 16u);
  assert(oi.period_tx == 4u && oi.period_ty == 4u);
}

void test_full_row_stripe() {
  // Up electrons on row y=0 only: bits 0..3 set.
  RawState s{0x000Fu, 0};
  std::vector<RawState> orb;
  enumerate_translation_orbit(s, &orb);
  assert(orb.size() == 4u);
  const RawState rep = canonical_raw_state(s);
  const auto stab = translation_stabilizer(rep);
  assert(stab.size() == 4u);
  for (const auto& e : stab) {
    assert(e.second == 0);  // only x-translations fix this pattern as a set on the same row mask
  }
  assert(primitive_period_tx(rep) == 1u);
  assert(primitive_period_ty(rep) == 4u);
}

void test_orbit_stabilizer_identity_all_reps() {
  std::vector<RawState> seeds;
  seeds.push_back({1u << 0, 0});
  seeds.push_back({0x000Fu, 0});
  seeds.push_back({1u << 0, 1u << 1});
  seeds.push_back({0, 0});  // vacuum
  const MomentumSectorMap map = build_momentum_sector_map(std::move(seeds));
  for (const OrbitRecord& r : map.orbits) {
    assert(check_orbit_stabilizer_count_identity(r));
    assert(popcount_momentum_mask(r.compatible_momentum_mask) == static_cast<int>(r.orbit_size));
  }
}

void test_same_canonical_for_orbit_members() {
  RawState a{1u << 0, 1u << 4};
  RawState b = translate_raw_state(a, 1, 0);
  assert(canonical_raw_state(a) == canonical_raw_state(b));
}

void test_vacuum_orbit_and_periods() {
  RawState vac{0, 0};
  std::vector<RawState> orb;
  TranslationHandler::orbit(vac, &orb);
  assert(orb.size() == 1u);
  const OrbitInfo inf = TranslationHandler::info(vac);
  assert(inf.orbit_size == 1u);
  assert(inf.period_tx == 1u && inf.period_ty == 1u);
  assert(TranslationHandler::canonical(vac) == vac);
}

void test_period_divides_four_and_fixes_rep() {
  RawState s{1u << 0, 1u << 5};
  const RawState rep = canonical_raw_state(s);
  const std::uint8_t px = primitive_period_tx(rep);
  const std::uint8_t py = primitive_period_ty(rep);
  assert(px == 1u || px == 2u || px == 4u);
  assert(py == 1u || py == 2u || py == 4u);
  assert(translate_raw_state(rep, px, 0) == rep);
  assert(translate_raw_state(rep, 0, py) == rep);
}

}  // namespace

int main() {
  test_single_up_orbit();
  test_full_row_stripe();
  test_orbit_stabilizer_identity_all_reps();
  test_same_canonical_for_orbit_members();
  test_vacuum_orbit_and_periods();
  test_period_divides_four_and_fixes_rep();
  std::cout << "test_symmetry_stage2 ok\n";
  return 0;
}
