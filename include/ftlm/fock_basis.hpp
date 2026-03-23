#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace ftlm {

/// Fixed (N_up, N_down) sector in the occupation-number basis.
/// Site i spin-σ is bit i of `up` (σ=↑) or `down` (σ=↓); up to 32 sites packed in `uint64_t`.
class FockBasis {
 public:
  FockBasis(int n_sites, int n_up, int n_down);

  int n_sites() const { return n_sites_; }
  int n_up() const { return n_up_; }
  int n_down() const { return n_down_; }
  int dim() const { return static_cast<int>(up_mask_.size()); }

  uint64_t up_mask(int idx) const { return up_mask_.at(static_cast<size_t>(idx)); }
  uint64_t down_mask(int idx) const { return down_mask_.at(static_cast<size_t>(idx)); }

  /// Lexicographic index of (up, down), or -1 if outside this sector.
  int index_of(uint64_t up, uint64_t down) const;

 private:
  static void enumerate_bit_combinations(int n, int k, int pos, uint64_t cur, std::vector<uint64_t>* out);

  int n_sites_;
  int n_up_;
  int n_down_;
  std::vector<uint64_t> up_mask_;
  std::vector<uint64_t> down_mask_;
  std::unordered_map<std::uint64_t, int> packed_to_index_;  ///< key = (up << n_sites) | down
};

}  // namespace ftlm
