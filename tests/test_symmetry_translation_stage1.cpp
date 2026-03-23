#include <cassert>
#include <cstdint>
#include <iostream>

#include "ftlm/symmetry/lattice_4x4.hpp"
#include "ftlm/symmetry/occup_bits_translate.hpp"
#include "ftlm/symmetry/raw_state.hpp"

namespace {

using ftlm::symmetry::Lattice4x4;
using ftlm::symmetry::RawState;
using ftlm::symmetry::translate_occupation_bits;
using ftlm::symmetry::translate_raw_state;

void test_indexing_convention() {
  assert(Lattice4x4::site_from_xy(0, 0) == 0);
  assert(Lattice4x4::site_from_xy(1, 0) == 1);
  assert(Lattice4x4::site_from_xy(0, 1) == 4);
  assert(Lattice4x4::site_from_xy(3, 3) == 15);
  int x = -1;
  int y = -1;
  Lattice4x4::xy_from_site(6, &x, &y);
  assert(x == 2 && y == 1);
}

void test_site_translation_periodicity() {
  assert(Lattice4x4::translate_site_index(0, 4, 0) == 0);
  assert(Lattice4x4::translate_site_index(0, -1, 0) == 3);
  assert(Lattice4x4::translate_site_index(15, 1, 1) == 0);
  assert(Lattice4x4::translate_site_Tx(3) == 0);
  assert(Lattice4x4::translate_site_Ty(12) == 0);
  assert(Lattice4x4::translate_site_Tx(0) == Lattice4x4::translate_site_index(0, 1, 0));
  assert(Lattice4x4::translate_site_Ty(0) == Lattice4x4::translate_site_index(0, 0, 1));
}

void test_site_translation_homomorphism() {
  const int s = 7;
  const int dx1 = 2;
  const int dy1 = -3;
  const int dx2 = -5;
  const int dy2 = 8;
  const int s1 = Lattice4x4::translate_site_index(s, dx1, dy1);
  const int s2 = Lattice4x4::translate_site_index(s1, dx2, dy2);
  const int s12 = Lattice4x4::translate_site_index(s, dx1 + dx2, dy1 + dy2);
  assert(s2 == s12);
}

void test_occupation_translation_moves_particle() {
  const std::uint16_t b0 = 1u << 0;
  const std::uint16_t b1 = translate_occupation_bits(b0, 1, 0);
  assert(b1 == (1u << 1));
  const std::uint16_t wrap = translate_occupation_bits(1u << 3, 1, 0);
  assert(wrap == (1u << 0));
}

void test_occupation_translation_is_group_action() {
  const std::uint16_t bits = (1u << 0) | (1u << 5) | (1u << 15);
  const int dx = 2;
  const int dy = -7;
  std::uint16_t once = translate_occupation_bits(bits, dx, dy);
  std::uint16_t twice = translate_occupation_bits(once, dx, dy);
  std::uint16_t direct = translate_occupation_bits(bits, 2 * dx, 2 * dy);
  assert(twice == direct);
}

void test_raw_state_translation() {
  RawState s{1u << 0, 1u << 4};
  RawState t = translate_raw_state(s, 1, 0);
  assert(t.up == (1u << 1));
  assert(t.dn == (1u << 5));
  assert(translate_raw_state(translate_raw_state(s, 1, 0), -1, 0) == s);
  assert(translate_raw_state_Tx(s) == t);
  RawState ty = translate_raw_state_Ty(s);
  assert(ty.up == (1u << 4));
  assert(ty.dn == (1u << 8));
}

}  // namespace

int main() {
  test_indexing_convention();
  test_site_translation_periodicity();
  test_site_translation_homomorphism();
  test_occupation_translation_moves_particle();
  test_occupation_translation_is_group_action();
  test_raw_state_translation();
  std::cout << "test_symmetry_translation_stage1 ok\n";
  return 0;
}
