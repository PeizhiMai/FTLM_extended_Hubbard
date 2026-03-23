#include "ftlm/hubbard_hamiltonian.hpp"

#include <algorithm>
#include <cmath>

namespace ftlm {

namespace {

#if defined(__GNUC__) || defined(__clang__)
int popcount64(uint64_t x) { return __builtin_popcountll(x); }
#else
int popcount64(uint64_t x) {
  int c = 0;
  while (x) {
    ++c;
    x &= x - 1;
  }
  return c;
}
#endif

/// Fermions strictly between mode indices `ma` and `mb` (exclusive), JW order (2*site + spin).
int fermions_between_modes(int ma, int mb, uint64_t up, uint64_t down) {
  if (ma > mb) {
    std::swap(ma, mb);
  }
  int c = 0;
  for (int m = ma + 1; m < mb; ++m) {
    const int site = m / 2;
    const int sp = m % 2;
    if (sp == 0) {
      if ((up >> site) & 1ULL) {
        ++c;
      }
    } else {
      if ((down >> site) & 1ULL) {
        ++c;
      }
    }
  }
  return c;
}

int site_occupancy(uint64_t up, uint64_t down, int site) {
  return static_cast<int>(((up >> site) & 1ULL) + ((down >> site) & 1ULL));
}

}  // namespace

void apply_extended_hubbard(const HubbardParams& p, const FockBasis& basis,
                            const std::vector<SpinfulHopping>& hoppings,
                            const std::vector<NearestPair>& nn_pairs,
                            const std::complex<double>* x, std::complex<double>* y) {
  const int dim = basis.dim();
  std::fill(y, y + dim, std::complex<double>(0.0, 0.0));

  for (int col = 0; col < dim; ++col) {
    const uint64_t u = basis.up_mask(col);
    const uint64_t d = basis.down_mask(col);
    const std::complex<double> vc = x[col];
    if (std::norm(vc) < 1e-28) {
      continue;
    }

    std::complex<double> diag(static_cast<double>(popcount64(u & d)) * p.U, 0.0);
    for (const auto& pr : nn_pairs) {
      const int oi = site_occupancy(u, d, pr.i);
      const int oj = site_occupancy(u, d, pr.j);
      diag += p.V * static_cast<double>(oi * oj);
    }
    y[col] += diag * vc;

    for (const auto& hop : hoppings) {
      const int from = hop.from;
      const int to = hop.to;
      const int spin = hop.spin;
      uint64_t upp = u;
      uint64_t downp = d;
      if (spin == 0) {
        if (((u >> from) & 1ULL) == 0) {
          continue;
        }
        if ((u >> to) & 1ULL) {
          continue;
        }
        upp = u ^ (1ULL << from) ^ (1ULL << to);
      } else {
        if (((d >> from) & 1ULL) == 0) {
          continue;
        }
        if ((d >> to) & 1ULL) {
          continue;
        }
        downp = d ^ (1ULL << from) ^ (1ULL << to);
      }
      const int ma = 2 * from + spin;
      const int mb = 2 * to + spin;
      const int between = fermions_between_modes(ma, mb, u, d);
      const double sign = (between & 1) ? -1.0 : 1.0;
      const int row = basis.index_of(upp, downp);
      if (row < 0) {
        continue;
      }
      y[row] += (sign * hop.coeff) * vc;
    }
  }
}

}  // namespace ftlm
