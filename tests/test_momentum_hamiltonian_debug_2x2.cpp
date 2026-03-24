#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "ftlm/fock_basis.hpp"
#include "ftlm/hubbard_hamiltonian.hpp"
#include "ftlm/hubbard_params.hpp"
#include "ftlm/lattice.hpp"
#include "ftlm/symmetry/hubbard_momentum_action.hpp"
#include "ftlm/symmetry/k_basis.hpp"
#include "ftlm/symmetry/momentum_sector.hpp"
#include "ftlm/symmetry/orbit.hpp"
#include "ftlm/symmetry/packed_raw_state.hpp"
#include "ftlm/symmetry/raw_state.hpp"

namespace {

constexpr int Lx = 2;
constexpr int Ly = 2;
constexpr int Ns = 4;
constexpr int Nup = 0;
constexpr int Ndn = 2;

std::string rs(const ftlm::symmetry::RawState& s) {
  std::ostringstream oss;
  oss << "(up=0x" << std::hex << static_cast<int>(s.up) << ",dn=0x" << static_cast<int>(s.dn) << std::dec << ")";
  return oss.str();
}

int fermions_between_modes16(int ma, int mb, std::uint16_t up, std::uint16_t dn) {
  if (ma > mb) std::swap(ma, mb);
  int c = 0;
  for (int m = ma + 1; m < mb; ++m) {
    const int site = m / 2;
    if ((m & 1) == 0) {
      if ((up >> site) & 1) ++c;
    } else {
      if ((dn >> site) & 1) ++c;
    }
  }
  return c;
}

bool translation_map_rect(int lx, int ly, ftlm::symmetry::RawState canon, ftlm::symmetry::RawState target, int* dx,
                          int* dy) {
  for (int ey = 0; ey < ly; ++ey) {
    for (int ex = 0; ex < lx; ++ex) {
      if (ftlm::symmetry::translate_raw_state_rect(canon, lx, ly, ex, ey) == target) {
        *dx = ex;
        *dy = ey;
        return true;
      }
    }
  }
  return false;
}

std::vector<std::vector<std::complex<double>>> build_full_H(const ftlm::HubbardParams& p, const ftlm::FockBasis& fb,
                                                            const std::vector<ftlm::SpinfulHopping>& hops,
                                                            const std::vector<ftlm::NearestPair>& pairs) {
  const int dim = fb.dim();
  std::vector<std::vector<std::complex<double>>> H(static_cast<size_t>(dim),
                                                   std::vector<std::complex<double>>(static_cast<size_t>(dim), 0.0));
  std::vector<std::complex<double>> e(static_cast<size_t>(dim), 0.0), y(static_cast<size_t>(dim), 0.0);
  for (int j = 0; j < dim; ++j) {
    std::fill(e.begin(), e.end(), std::complex<double>(0.0, 0.0));
    e[static_cast<size_t>(j)] = {1.0, 0.0};
    ftlm::apply_extended_hubbard(p, fb, hops, pairs, e.data(), y.data());
    for (int i = 0; i < dim; ++i) H[static_cast<size_t>(i)][static_cast<size_t>(j)] = y[static_cast<size_t>(i)];
  }
  return H;
}

void print_matrix(const std::string& name, const std::vector<std::vector<std::complex<double>>>& M) {
  std::cout << name << " (" << M.size() << "x" << M.size() << ")\n";
  std::cout << std::setprecision(8);
  for (size_t i = 0; i < M.size(); ++i) {
    for (size_t j = 0; j < M.size(); ++j) {
      const auto z = M[i][j];
      std::cout << "(" << z.real() << (z.imag() >= 0 ? "+" : "") << z.imag() << "i)";
      if (j + 1 < M.size()) std::cout << " ";
    }
    std::cout << "\n";
  }
}

}  // namespace

int main() {
  ftlm::HubbardParams p;
  p.Lx = Lx;
  p.Ly = Ly;
  p.t = 1.0;
  p.tp = -0.35;
  p.U = 5.75;
  p.V = 0.9;
  p.phi_x = 0.0;
  p.phi_y = 0.0;

  ftlm::RectLattice lat{Lx, Ly};
  std::vector<ftlm::SpinfulHopping> hops;
  std::vector<ftlm::NearestPair> pairs;
  ftlm::build_hubbard_geometry(p, lat, &hops, &pairs);

  ftlm::FockBasis fb(Ns, Nup, Ndn);
  const int dim_full = fb.dim();
  std::vector<ftlm::symmetry::RawState> universe;
  universe.reserve(static_cast<size_t>(dim_full));
  for (int i = 0; i < dim_full; ++i) {
    universe.push_back(ftlm::symmetry::RawState{static_cast<std::uint16_t>(fb.up_mask(i)),
                                                static_cast<std::uint16_t>(fb.down_mask(i))});
  }
  const auto map = ftlm::symmetry::build_momentum_sector_map_rect(std::move(universe), Lx, Ly);
  ftlm::symmetry::HubbardMomentumAction hub(p);

  const auto H_full = build_full_H(p, fb, hops, pairs);
  print_matrix("H_full (old full-sector)", H_full);

  std::vector<std::vector<std::complex<double>>> H_recon(
      static_cast<size_t>(dim_full), std::vector<std::complex<double>>(static_cast<size_t>(dim_full), 0.0));

  std::cout << "\n[Step 4] Normalization diagnostics for projected basis\n";
  std::cout << "Convention check: group-sum projected norm^2 should be |G|*|S|, orbit-sum norm^2 should be |orbit|.\n";

  for (int ky = 0; ky < Ly; ++ky) {
    for (int kx = 0; kx < Lx; ++kx) {
      const ftlm::symmetry::MomentumSector K{kx, ky, Lx, Ly};
      const auto kb = ftlm::symmetry::KBasis::build(map, K);
      const int dk = static_cast<int>(hub.momentum_block_dim(map, K, Nup, Ndn));
      if (dk <= 0) continue;

      std::cout << "\n=== K=(" << kx << "," << ky << "), dim=" << dk << " (KBasis.dim=" << kb.dim() << ") ===\n";
      std::vector<std::vector<std::complex<double>>> Hk(
          static_cast<size_t>(dk), std::vector<std::complex<double>>(static_cast<size_t>(dk), 0.0));

      for (int j = 0; j < dk && j < static_cast<int>(kb.dim()); ++j) {
        const auto r = kb.representatives[static_cast<size_t>(j)];
        std::size_t s_a = 1;
        for (const auto& o : map.orbits) {
          if (ftlm::symmetry::pack_raw_state(o.representative) == ftlm::symmetry::pack_raw_state(r)) {
            s_a = o.stabilizer.size();
            break;
          }
        }
        const int orbit_size = (Lx * Ly) / static_cast<int>(s_a);
        const double norm2_group = static_cast<double>(Lx * Ly) * static_cast<double>(s_a);
        const double norm2_orbit = static_cast<double>(orbit_size);
        std::cout << "rep j=" << j << " " << rs(r) << " stabilizer=" << s_a << " orbit=" << orbit_size
                  << " norm2_group=" << norm2_group << " norm2_orbit=" << norm2_orbit << "\n";
      }

      std::vector<std::complex<double>> x(static_cast<size_t>(dk), 0.0), y(static_cast<size_t>(dk), 0.0);
      for (int j = 0; j < dk; ++j) {
        std::fill(x.begin(), x.end(), std::complex<double>(0.0, 0.0));
        x[static_cast<size_t>(j)] = {1.0, 0.0};
        hub.apply(map, K, Nup, Ndn, x.data(), y.data());
        for (int i = 0; i < dk; ++i) Hk[static_cast<size_t>(i)][static_cast<size_t>(j)] = y[static_cast<size_t>(i)];

        if (j >= static_cast<int>(kb.dim())) {
          continue;
        }
        const auto r = kb.representatives[static_cast<size_t>(j)];
        std::size_t s_a = 1;
        for (const auto& o : map.orbits) {
          if (ftlm::symmetry::pack_raw_state(o.representative) == ftlm::symmetry::pack_raw_state(r)) {
            s_a = o.stabilizer.size();
            break;
          }
        }

        // Instrument hopping contributions with current formula.
        for (const auto& hop : hops) {
          const int from = hop.from, to = hop.to, spin = hop.spin;
          std::uint16_t upp = r.up, dnp = r.dn;
          if (spin == 0) {
            if (((r.up >> from) & 1u) == 0 || ((r.up >> to) & 1u)) continue;
            upp = static_cast<std::uint16_t>(r.up ^ (1u << from) ^ (1u << to));
          } else {
            if (((r.dn >> from) & 1u) == 0 || ((r.dn >> to) & 1u)) continue;
            dnp = static_cast<std::uint16_t>(r.dn ^ (1u << from) ^ (1u << to));
          }
          const int ma = 2 * from + spin, mb = 2 * to + spin;
          const double sign = (fermions_between_modes16(ma, mb, r.up, r.dn) & 1) ? -1.0 : 1.0;
          const std::complex<double> bare = sign * hop.coeff;
          const auto s = ftlm::symmetry::RawState{upp, dnp};
          const auto rp = ftlm::symmetry::canonical_raw_state_rect(s, Lx, Ly);
          int dx = 0, dy = 0;
          if (!translation_map_rect(Lx, Ly, rp, s, &dx, &dy)) continue;
          const auto ib = kb.index_of(rp);
          if (ib == ftlm::symmetry::KBasis::npos) continue;
          std::size_t s_b = 1;
          for (const auto& o : map.orbits) {
            if (ftlm::symmetry::pack_raw_state(o.representative) == ftlm::symmetry::pack_raw_state(rp)) {
              s_b = o.stabilizer.size();
              break;
            }
          }
          const double ratio = std::sqrt(static_cast<double>(s_a) / static_cast<double>(s_b));
          const std::complex<double> phase_used = std::conj(ftlm::symmetry::translation_bloch_phase(K, dx, dy));
          const std::complex<double> expected_phase = std::exp(std::complex<double>(
              0.0, (2.0 * M_PI) * (static_cast<double>(kx * dx) / static_cast<double>(Lx) +
                                    static_cast<double>(ky * dy) / static_cast<double>(Ly))));
          const std::complex<double> term_used = bare * phase_used * ratio;
          const std::complex<double> term_expected = bare * expected_phase;
          std::cout << "hop j=" << j << " -> i=" << ib << " r=" << rs(r) << " s=" << rs(s) << " r'=" << rs(rp)
                    << " R=(" << dx << "," << dy << ") K=(" << kx << "," << ky << ") bare=" << bare
                    << " phase_used=" << phase_used << " phase_exp(iK.R)=" << expected_phase
                    << " term_used=" << term_used << " bare*exp(iK.R)=" << term_expected << "\n";
        }
      }

      // Build projector P_k into full basis and add P Hk P^dagger (orbit–Bloch columns; only valid when dk==KBasis.dim).
      if (static_cast<size_t>(dk) == kb.dim()) {
        std::vector<std::vector<std::complex<double>>> P(
            static_cast<size_t>(dim_full), std::vector<std::complex<double>>(static_cast<size_t>(dk), 0.0));
        for (int j = 0; j < dk; ++j) {
          const auto rep = kb.representatives[static_cast<size_t>(j)];
          std::size_t s_sz = 1;
          for (const auto& o : map.orbits) {
            if (ftlm::symmetry::pack_raw_state(o.representative) == ftlm::symmetry::pack_raw_state(rep)) {
              s_sz = o.stabilizer.size();
              break;
            }
          }
          const double c = std::sqrt(1.0 / (static_cast<double>(Lx * Ly) * static_cast<double>(s_sz)));
          for (int dy = 0; dy < Ly; ++dy) {
            for (int dx = 0; dx < Lx; ++dx) {
              const auto st = ftlm::symmetry::translate_raw_state_rect(rep, Lx, Ly, dx, dy);
              const int i_full = fb.index_of(st.up, st.dn);
              if (i_full >= 0) {
                P[static_cast<size_t>(i_full)][static_cast<size_t>(j)] +=
                    c * std::conj(ftlm::symmetry::translation_bloch_phase(K, dx, dy));
              }
            }
          }
        }
        for (int a = 0; a < dim_full; ++a) {
          for (int b = 0; b < dim_full; ++b) {
            std::complex<double> z(0.0, 0.0);
            for (int i = 0; i < dk; ++i) {
              for (int j = 0; j < dk; ++j) {
                z += P[static_cast<size_t>(a)][static_cast<size_t>(i)] *
                     Hk[static_cast<size_t>(i)][static_cast<size_t>(j)] *
                     std::conj(P[static_cast<size_t>(b)][static_cast<size_t>(j)]);
              }
            }
            H_recon[static_cast<size_t>(a)][static_cast<size_t>(b)] += z;
          }
        }
      } else {
        std::cout << "  [skip orbit P reconstruction: dk=" << dk << " != KBasis.dim=" << kb.dim() << "]\n";
      }
    }
  }

  print_matrix("\nH_reconstructed (from momentum blocks)", H_recon);
  std::cout << "\nElement-wise diff: H_reconstructed - H_full\n";
  double mx = 0.0;
  for (int i = 0; i < dim_full; ++i) {
    for (int j = 0; j < dim_full; ++j) {
      const auto d = H_recon[static_cast<size_t>(i)][static_cast<size_t>(j)] -
                     H_full[static_cast<size_t>(i)][static_cast<size_t>(j)];
      mx = std::max(mx, std::abs(d));
      std::cout << "(" << d.real() << (d.imag() >= 0 ? "+" : "") << d.imag() << "i)";
      if (j + 1 < dim_full) std::cout << " ";
    }
    std::cout << "\n";
  }
  std::cout << "max_abs_diff(H_reconstructed, H_full)=" << mx << "\n";
  return 0;
}
