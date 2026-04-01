#include "ftlm/symmetry/orbit_k_block_basis.hpp"

#include "ftlm/symmetry/fermionic_translation.hpp"
#include "ftlm/symmetry/momentum_sector.hpp"  // translation_bloch_phase
#include "ftlm/symmetry/orbit.hpp"
#include "ftlm/symmetry/packed_raw_state.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ftlm {
namespace symmetry {
namespace {

#if defined(__GNUC__) || defined(__clang__)
int popcount64(std::uint64_t x) { return __builtin_popcountll(x); }
#else
int popcount64(std::uint64_t x) {
  int c = 0;
  while (x) {
    ++c;
    x &= x - 1;
  }
  return c;
}
#endif

int fermions_between_modes(int ma, int mb, std::uint64_t up, std::uint64_t down) {
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

int site_occupancy(std::uint64_t up, std::uint64_t down, int site) {
  return static_cast<int>(((up >> site) & 1ULL) + ((down >> site) & 1ULL));
}

}  // namespace

std::complex<double> fermionic_stabilizer_character_A_rect(
    const RawState& rep, MomentumSector K, int lx, int ly,
    const std::vector<std::pair<std::int8_t, std::int8_t>>& stabilizer) {
  std::complex<double> A{0.0, 0.0};
  for (const auto& e : stabilizer) {
    const int dx = static_cast<int>(e.first);
    const int dy = static_cast<int>(e.second);
    const double eta = fermionic_translation_sign_rect(rep, lx, ly, dx, dy);
    const std::complex<double> chi_star = translation_bloch_phase(K, dx, dy);
    A += std::complex<double>(eta, 0.0) * chi_star;
  }
  return A;
}

bool OrbitKBlockBasis::build(const FockBasis& fb, MomentumSector Ksec, double survive_tol) {
  entries_.clear();
  rep_pack_to_row_.clear();
  survive_tol_ = survive_tol;
  K = Ksec;
  lx = K.lx;
  ly = K.ly;
  if (fb.n_sites() > 16) {
    throw std::invalid_argument("OrbitKBlockBasis::build: n_sites must be <= 16 (RawState bit width)");
  }

  std::vector<RawState> universe;
  universe.reserve(static_cast<std::size_t>(fb.dim()));
  for (int i = 0; i < fb.dim(); ++i) {
    universe.push_back(RawState{static_cast<std::uint16_t>(fb.up_mask(i)), static_cast<std::uint16_t>(fb.down_mask(i))});
  }

  MomentumSectorMap map = build_momentum_sector_map_rect(std::move(universe), lx, ly);

  std::vector<OrbitKRepEntry> cand;
  cand.reserve(map.orbits.size());
  for (std::size_t oi = 0; oi < map.orbits.size(); ++oi) {
    const OrbitRecord& rec = map.orbits[oi];
    const std::complex<double> A = fermionic_stabilizer_character_A_rect(rec.representative, K, lx, ly, rec.stabilizer);
    const double aabs = std::abs(A);
    if (aabs <= survive_tol_) {
      continue;
    }
    OrbitKRepEntry e;
    e.rep = rec.representative;
    e.orbit_index = oi;
    e.orbit_size = rec.orbit_size;
    e.stabilizer_size = static_cast<std::uint16_t>(rec.stabilizer.size());
    e.A_k = A;
    e.abs_A = aabs;
    cand.push_back(std::move(e));
  }

  std::sort(cand.begin(), cand.end(), [](const OrbitKRepEntry& a, const OrbitKRepEntry& b) {
    return pack_raw_state(a.rep) < pack_raw_state(b.rep);
  });

  entries_ = std::move(cand);
  for (std::size_t r = 0; r < entries_.size(); ++r) {
    rep_pack_to_row_[pack_raw_state(entries_[r].rep)] = r;
  }
  return true;
}

int OrbitKBlockBasis::reduced_index_of_rep(RawState rep) const {
  const auto it = rep_pack_to_row_.find(pack_raw_state(rep));
  if (it == rep_pack_to_row_.end()) {
    return -1;
  }
  return static_cast<int>(it->second);
}

bool OrbitKBlockBasis::map_raw_to_rep(RawState s, int* reduced_index, int* dx, int* dy, double* eta) const {
  if (reduced_index == nullptr || dx == nullptr || dy == nullptr || eta == nullptr) {
    return false;
  }
  const RawState rep = canonical_raw_state_rect(s, lx, ly);
  const auto it = rep_pack_to_row_.find(pack_raw_state(rep));
  if (it == rep_pack_to_row_.end()) {
    return false;
  }
  for (int tdx = 0; tdx < lx; ++tdx) {
    for (int tdy = 0; tdy < ly; ++tdy) {
      const RawState im = translate_raw_state_rect(rep, lx, ly, tdx, tdy);
      if (im.up == s.up && im.dn == s.dn) {
        *reduced_index = static_cast<int>(it->second);
        *dx = tdx;
        *dy = tdy;
        *eta = fermionic_translation_sign_rect(rep, lx, ly, tdx, tdy);
        return true;
      }
    }
  }
  return false;
}

void apply_orbit_k_block(const HubbardParams& p, const FockBasis& fb, const OrbitKBlockBasis& basis,
                         const std::vector<SpinfulHopping>& hoppings,
                         const std::vector<NearestPair>& nn_pairs,
                         const std::complex<double>* x,
                         std::complex<double>* y) {
  const int dk = basis.dim();
  std::fill(y, y + dk, std::complex<double>(0.0, 0.0));
  if (dk <= 0) {
    return;
  }

  const auto& entries = basis.entries();
  for (int b = 0; b < dk; ++b) {
    const std::complex<double> xb = x[static_cast<std::size_t>(b)];
    if (std::norm(xb) < 1e-28) {
      continue;
    }
    const OrbitKRepEntry& beta = entries[static_cast<std::size_t>(b)];
    const std::uint64_t u = static_cast<std::uint64_t>(beta.rep.up);
    const std::uint64_t d = static_cast<std::uint64_t>(beta.rep.dn);

    // Diagonal contribution: alpha=beta, R=(0,0), eta=1, chi_K=1, sqrt(|A_a|/|A_b|)=1.
    std::complex<double> diag(static_cast<double>(popcount64(u & d)) * p.U, 0.0);
    for (const auto& pr : nn_pairs) {
      const int oi = site_occupancy(u, d, pr.i);
      const int oj = site_occupancy(u, d, pr.j);
      diag += p.V * static_cast<double>(oi * oj);
    }
    y[static_cast<std::size_t>(b)] += diag * xb;

    for (const auto& hop : hoppings) {
      const int from = hop.from;
      const int to = hop.to;
      const int spin = hop.spin;
      std::uint64_t upp = u;
      std::uint64_t dnp = d;
      if (spin == 0) {
        if (((u >> from) & 1ULL) == 0 || ((u >> to) & 1ULL) != 0) {
          continue;
        }
        upp = u ^ (1ULL << from) ^ (1ULL << to);
      } else {
        if (((d >> from) & 1ULL) == 0 || ((d >> to) & 1ULL) != 0) {
          continue;
        }
        dnp = d ^ (1ULL << from) ^ (1ULL << to);
      }
      const int ma = 2 * from + spin;
      const int mb = 2 * to + spin;
      const int between = fermions_between_modes(ma, mb, u, d);
      const double jw_sign = (between & 1) ? -1.0 : 1.0;
      const std::complex<double> h_spb = jw_sign * hop.coeff;

      const RawState sprime{static_cast<std::uint16_t>(upp), static_cast<std::uint16_t>(dnp)};
      int a = -1;
      int dx = 0;
      int dy = 0;
      double eta = 0.0;
      if (!basis.map_raw_to_rep(sprime, &a, &dx, &dy, &eta)) {
        continue;
      }
      const OrbitKRepEntry& alpha = entries[static_cast<std::size_t>(a)];
      const double norm = std::sqrt(alpha.abs_A / beta.abs_A);
      // Bloch factor for T_R α = s′: must match `translation_bloch_phase` (χ_K^* in momentum_sector.hpp), same as in
      // `fill_phi_orbit_bloch_from_seed`. Using `conj(translation_bloch_phase)` (χ_K) skewed Im(H_K) vs Φ† H Φ.
      const std::complex<double> chi = translation_bloch_phase(basis.K, dx, dy);
      y[static_cast<std::size_t>(a)] += h_spb * std::complex<double>(eta * norm, 0.0) * chi * xb;
    }
  }
}

}  // namespace symmetry
}  // namespace ftlm

