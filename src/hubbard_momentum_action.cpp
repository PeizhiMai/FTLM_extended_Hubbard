#include "ftlm/symmetry/hubbard_momentum_action.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "ftlm/symmetry/orbit.hpp"
#include "ftlm/symmetry/packed_raw_state.hpp"
#include "ftlm/symmetry/raw_state.hpp"

namespace ftlm {
namespace symmetry {
namespace {

int popcount16(std::uint16_t x) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_popcount(static_cast<unsigned>(x));
#else
  int c = 0;
  while (x) {
    ++c;
    x &= static_cast<std::uint16_t>(x - 1);
  }
  return c;
#endif
}

/// JW string: fermions strictly between mode indices `ma` and `mb` (exclusive), mode \(2\cdot s+\sigma\).
int fermions_between_modes16(int ma, int mb, std::uint16_t up, std::uint16_t dn) {
  if (ma > mb) {
    std::swap(ma, mb);
  }
  int c = 0;
  for (int m = ma + 1; m < mb; ++m) {
    const int site = m / 2;
    if ((m & 1) == 0) {
      if ((up >> site) & 1) {
        ++c;
      }
    } else {
      if ((dn >> site) & 1) {
        ++c;
      }
    }
  }
  return c;
}

int site_occupancy16(std::uint16_t up, std::uint16_t dn, int site) {
  return static_cast<int>(((up >> site) & 1u) + ((dn >> site) & 1u));
}

int double_occupancy_count(std::uint16_t up, std::uint16_t dn) { return popcount16(static_cast<std::uint16_t>(up & dn)); }

std::complex<double> diagonal_energy(const HubbardParams& p, RawState s,
                                      const std::vector<NearestPair>& nn_pairs) {
  double re = static_cast<double>(double_occupancy_count(s.up, s.dn)) * p.U;
  for (const auto& pr : nn_pairs) {
    const int oi = site_occupancy16(s.up, s.dn, pr.i);
    const int oj = site_occupancy16(s.up, s.dn, pr.j);
    re += p.V * static_cast<double>(oi * oj);
  }
  return {re, 0.0};
}

std::size_t stabilizer_size_of_rep(const MomentumSectorMap& map, RawState r) {
  const std::uint32_t pk = pack_raw_state(r);
  for (const auto& o : map.orbits) {
    if (pack_raw_state(o.representative) == pk) {
      return o.stabilizer.size();
    }
  }
  throw std::out_of_range("hubbard_momentum_action: representative not in orbit_map");
}

/// Some `(dx,dy)` with `translate_raw_state(canon, dx, dy) == target` (χ is well-defined on sector).
bool translation_map(RawState canon, RawState target, int* dx, int* dy) {
  for (int ey = 0; ey < 4; ++ey) {
    for (int ex = 0; ex < 4; ++ex) {
      if (translate_raw_state(canon, ex, ey) == target) {
        *dx = ex;
        *dy = ey;
        return true;
      }
    }
  }
  return false;
}

}  // namespace

HubbardMomentumAction::HubbardMomentumAction(HubbardParams p) : params(std::move(p)) {
  if (params.Lx != 4 || params.Ly != 4) {
    throw std::invalid_argument("HubbardMomentumAction: requires Lx=Ly=4 for Z4xZ4 symmetry code");
  }
  lat.Lx = 4;
  lat.Ly = 4;
  build_hubbard_geometry(params, lat, &hoppings, &nn_pairs);
}

void HubbardMomentumAction::apply(const MomentumSectorMap& orbit_map, MomentumSector K, const KBasis& basis,
                                  const std::complex<double>* x, std::complex<double>* y) const {
  const std::size_t n = basis.dim();
  std::fill(y, y + n, std::complex<double>(0.0, 0.0));

  for (std::size_t j = 0; j < n; ++j) {
    const std::complex<double> vc = x[j];
    if (std::norm(vc) < 1e-28) {
      continue;
    }
    const RawState r = basis.representatives[j];
    const std::size_t s_a = stabilizer_size_of_rep(orbit_map, r);
    const std::complex<double> diag = diagonal_energy(params, r, nn_pairs);
    y[j] += diag * vc;

    for (const SpinfulHopping& hop : hoppings) {
      const int from = hop.from;
      const int to = hop.to;
      const int spin = hop.spin;
      std::uint16_t upp = r.up;
      std::uint16_t dnp = r.dn;
      if (spin == 0) {
        if (((r.up >> from) & 1u) == 0) {
          continue;
        }
        if ((r.up >> to) & 1u) {
          continue;
        }
        upp = static_cast<std::uint16_t>(r.up ^ (1u << from) ^ (1u << to));
      } else {
        if (((r.dn >> from) & 1u) == 0) {
          continue;
        }
        if ((r.dn >> to) & 1u) {
          continue;
        }
        dnp = static_cast<std::uint16_t>(r.dn ^ (1u << from) ^ (1u << to));
      }
      const int ma = 2 * from + spin;
      const int mb = 2 * to + spin;
      const double sign = (fermions_between_modes16(ma, mb, r.up, r.dn) & 1) ? -1.0 : 1.0;

      const RawState jumped{upp, dnp};
      const RawState r_can = canonical_raw_state(jumped);
      int dx = 0;
      int dy = 0;
      if (!translation_map(r_can, jumped, &dx, &dy)) {
        continue;
      }
      const std::size_t ib = basis.index_of(r_can);
      if (ib == KBasis::npos) {
        continue;
      }
      const std::size_t s_b = stabilizer_size_of_rep(orbit_map, r_can);
      const double ratio = std::sqrt(static_cast<double>(s_b) / static_cast<double>(s_a));
      const std::complex<double> phase = translation_bloch_phase(K, dx, dy);
      y[ib] += vc * (sign * hop.coeff * phase * ratio);
    }
  }
}

}  // namespace symmetry
}  // namespace ftlm
