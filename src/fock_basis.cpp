#include "ftlm/fock_basis.hpp"

#include <stdexcept>

namespace ftlm {

void FockBasis::enumerate_bit_combinations(int n, int k, int pos, uint64_t cur, std::vector<uint64_t>* out) {
  if (k == 0) {
    out->push_back(cur);
    return;
  }
  for (int i = pos; i <= n - k; ++i) {
    enumerate_bit_combinations(n, k - 1, i + 1, cur | (1ULL << i), out);
  }
}

FockBasis::FockBasis(int n_sites, int n_up, int n_down)
    : n_sites_(n_sites), n_up_(n_up), n_down_(n_down) {
  if (n_sites <= 0 || n_sites > 32) {
    throw std::invalid_argument("FockBasis: require 1 <= n_sites <= 32 (bit packing)");
  }
  if (n_up < 0 || n_down < 0 || n_up > n_sites || n_down > n_sites) {
    throw std::invalid_argument("FockBasis: particle numbers out of range");
  }
  std::vector<uint64_t> ups;
  std::vector<uint64_t> downs;
  enumerate_bit_combinations(n_sites, n_up, 0, 0, &ups);
  enumerate_bit_combinations(n_sites, n_down, 0, 0, &downs);
  up_mask_.clear();
  down_mask_.clear();
  packed_to_index_.clear();
  int idx = 0;
  for (uint64_t u : ups) {
    for (uint64_t d : downs) {
      up_mask_.push_back(u);
      down_mask_.push_back(d);
      const std::uint64_t key = (u << n_sites_) | d;
      packed_to_index_[key] = idx;
      ++idx;
    }
  }
}

int FockBasis::index_of(uint64_t up, uint64_t down) const {
  const std::uint64_t key = (up << n_sites_) | down;
  const auto it = packed_to_index_.find(key);
  if (it == packed_to_index_.end()) {
    return -1;
  }
  return it->second;
}

}  // namespace ftlm
