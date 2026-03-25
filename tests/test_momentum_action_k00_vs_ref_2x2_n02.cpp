// Strict side-by-side: sector (2,2), 2x2, K=(0,0) (Γ) only.
// H_ref = Phi^H H Phi with Phi from dense P_k = rank(P_k) spectral basis (same as HubbardMomentumAction).
// H_cur = columns from HubbardMomentumAction::apply.

#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <limits>
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
#include "ftlm/symmetry/orbit_bloch_phi.hpp"
#include "ftlm/symmetry/packed_raw_state.hpp"
#include "ftlm/symmetry/raw_state.hpp"

namespace {

constexpr int Lx = 2;
constexpr int Ly = 2;
constexpr int Ns = 4;
constexpr int Nup = 2;
constexpr int Ndn = 2;

std::string rs(const ftlm::symmetry::RawState& s) {
  std::ostringstream oss;
  oss << "(up=0x" << std::hex << static_cast<int>(s.up) << ",dn=0x" << static_cast<int>(s.dn) << std::dec << ")";
  return oss.str();
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

void print_cmatrix(const std::string& name, const std::vector<std::vector<std::complex<double>>>& M) {
  std::cout << name << "\n" << std::setprecision(12);
  for (size_t i = 0; i < M.size(); ++i) {
    for (size_t j = 0; j < M[i].size(); ++j) {
      const auto z = M[i][j];
      std::cout << "(" << z.real() << (z.imag() >= 0 ? "+" : "") << z.imag() << "i)";
      if (j + 1 < M[i].size()) std::cout << " ";
    }
    std::cout << "\n";
  }
}

double cabs(std::complex<double> z) { return std::abs(z); }

/// C = A^H B, A dim n×ma, B n×mb
std::vector<std::vector<std::complex<double>>> matmul_AhB(const std::vector<std::vector<std::complex<double>>>& A,
                                                          const std::vector<std::vector<std::complex<double>>>& B) {
  const int n = static_cast<int>(A.size());
  const int ma = static_cast<int>(A[0].size());
  const int mb = static_cast<int>(B[0].size());
  std::vector<std::vector<std::complex<double>>> C(static_cast<size_t>(ma),
                                                   std::vector<std::complex<double>>(static_cast<size_t>(mb), 0.0));
  for (int i = 0; i < ma; ++i) {
    for (int j = 0; j < mb; ++j) {
      std::complex<double> s(0.0, 0.0);
      for (int k = 0; k < n; ++k) {
        s += std::conj(A[static_cast<size_t>(k)][static_cast<size_t>(i)]) * B[static_cast<size_t>(k)][static_cast<size_t>(j)];
      }
      C[static_cast<size_t>(i)][static_cast<size_t>(j)] = s;
    }
  }
  return C;
}

std::vector<std::vector<std::complex<double>>> matmul_AB(const std::vector<std::vector<std::complex<double>>>& A,
                                                           const std::vector<std::vector<std::complex<double>>>& B) {
  const int ma = static_cast<int>(A.size());
  const int na = static_cast<int>(A[0].size());
  const int mb = static_cast<int>(B[0].size());
  std::vector<std::vector<std::complex<double>>> C(static_cast<size_t>(ma),
                                                   std::vector<std::complex<double>>(static_cast<size_t>(mb), 0.0));
  for (int i = 0; i < ma; ++i) {
    for (int j = 0; j < mb; ++j) {
      std::complex<double> s(0.0, 0.0);
      for (int t = 0; t < na; ++t) {
        s += A[static_cast<size_t>(i)][static_cast<size_t>(t)] * B[static_cast<size_t>(t)][static_cast<size_t>(j)];
      }
      C[static_cast<size_t>(i)][static_cast<size_t>(j)] = s;
    }
  }
  return C;
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

}  // namespace

int main() {
  constexpr double tol = 1e-9;

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
  for (int i = 0; i < dim_full; ++i) {
    universe.push_back(ftlm::symmetry::RawState{static_cast<std::uint16_t>(fb.up_mask(i)),
                                                static_cast<std::uint16_t>(fb.down_mask(i))});
  }
  const auto map = ftlm::symmetry::build_momentum_sector_map_rect(std::move(universe), Lx, Ly);
  const ftlm::symmetry::MomentumSector K{0, 0, Lx, Ly};
  const auto kb = ftlm::symmetry::KBasis::build(map, K);

  ftlm::symmetry::HubbardMomentumAction hub(p);
  const int dk = static_cast<int>(hub.momentum_block_dim(map, K, Nup, Ndn));
  if (dk <= 0) {
    std::cerr << "Expected positive momentum block dim for K=(0,0), got " << dk << "\n";
    return 2;
  }

  const auto H_full = build_full_H(p, fb, hops, pairs);

  // Reference Phi: same construction as HubbardMomentumBlock (orbit Bloch columns + Gram–Schmidt).
  std::vector<std::complex<double>> phi_cm;
  std::size_t dk_ref = 0;
  ftlm::symmetry::build_momentum_phi_orbit_orthonormal(map, K, Lx, Ly, fb, &phi_cm, &dk_ref);
  if (static_cast<int>(dk_ref) != dk) {
    std::cerr << "dk mismatch: orbit phi dk=" << dk_ref << " vs momentum_block_dim=" << dk << "\n";
    return 2;
  }
  std::vector<std::vector<std::complex<double>>> Phi(static_cast<size_t>(dim_full),
                                                     std::vector<std::complex<double>>(static_cast<size_t>(dk), 0.0));
  for (int p = 0; p < dim_full; ++p) {
    for (int j = 0; j < dk; ++j) {
      Phi[static_cast<size_t>(p)][static_cast<size_t>(j)] =
          phi_cm[static_cast<size_t>(p) + static_cast<size_t>(j) * static_cast<size_t>(dim_full)];
    }
  }

  const auto H_Phi = matmul_AB(H_full, Phi);
  const auto H_ref = matmul_AhB(Phi, H_Phi);

  // Cross-check: (Phi^H H Phi)_ij = conj(Phi[:,i])^T (H (Phi[:,j])).
  std::vector<std::complex<double>> v(static_cast<size_t>(dim_full), 0.0), w(static_cast<size_t>(dim_full), 0.0);
  for (int p = 0; p < dim_full; ++p) v[static_cast<size_t>(p)] = Phi[static_cast<size_t>(p)][2];
  ftlm::apply_extended_hubbard(p, fb, hops, pairs, v.data(), w.data());
  std::complex<double> lift_proj_12(0.0, 0.0);
  for (int p = 0; p < dim_full; ++p) {
    lift_proj_12 += std::conj(Phi[static_cast<size_t>(p)][1]) * w[static_cast<size_t>(p)];
  }
  std::cout << "Cross-check lift-project (i,j)=(1,2): " << lift_proj_12 << " vs H_ref(1,2)=" << H_ref[1][2] << "\n";
  std::cout << "Total SpinfulHopping count=" << hops.size() << "\n";

  std::vector<std::vector<std::complex<double>>> H_cur(
      static_cast<size_t>(dk), std::vector<std::complex<double>>(static_cast<size_t>(dk), 0.0));
  std::vector<std::complex<double>> x(static_cast<size_t>(dk), 0.0), y(static_cast<size_t>(dk), 0.0);
  for (int j = 0; j < dk; ++j) {
    std::fill(x.begin(), x.end(), std::complex<double>(0.0, 0.0));
    x[static_cast<size_t>(j)] = {1.0, 0.0};
    hub.apply(map, K, Nup, Ndn, x.data(), y.data());
    for (int i = 0; i < dk; ++i) H_cur[static_cast<size_t>(i)][static_cast<size_t>(j)] = y[static_cast<size_t>(i)];
  }

  std::vector<std::vector<std::complex<double>>> Delta(static_cast<size_t>(dk),
                                                       std::vector<std::complex<double>>(static_cast<size_t>(dk), 0.0));
  double max_d = 0.0;
  int fi = -1, fj = -1;
  for (int i = 0; i < dk; ++i) {
    for (int j = 0; j < dk; ++j) {
      Delta[static_cast<size_t>(i)][static_cast<size_t>(j)] =
          H_cur[static_cast<size_t>(i)][static_cast<size_t>(j)] - H_ref[static_cast<size_t>(i)][static_cast<size_t>(j)];
      const double ad = cabs(Delta[static_cast<size_t>(i)][static_cast<size_t>(j)]);
      if (ad > max_d) max_d = ad;
      if (fi < 0 && ad > tol) {
        fi = i;
        fj = j;
      }
    }
  }

  std::cout << "=== Scope: 2x2, (Nup,Ndn)=(2,2), K=(0,0), dk=" << dk << " (KBasis.dim=" << kb.dim() << ") ===\n";
  std::cout << "K-basis reps (column order j=0.." << (dk - 1) << "):\n";
  for (int j = 0; j < dk; ++j) {
    std::cout << "  j=" << j << " " << rs(kb.representatives[static_cast<size_t>(j)]) << "\n";
  }

  std::cout << "\n--- Convention audit (reference path) ---\n";
  std::cout << "  Phi columns = orbit Bloch sums per KBasis rep, then orthonormalized (modified Gram–Schmidt).\n";
  std::cout << "  H_ref = Phi^H H_full Phi.\n";

  std::cout << "\n--- Convention audit (HubbardMomentumAction::apply) ---\n";
  std::cout << "  Same Phi as reference; y = Phi^H (H_full (Phi x)).\n";

  print_cmatrix("H_ref", H_ref);
  print_cmatrix("H_cur", H_cur);
  print_cmatrix("Delta = H_cur - H_ref", Delta);
  std::cout << "max_abs(Delta) = " << max_d << "\n";
  if (fi >= 0) {
    std::cout << "First |Delta_ij| > tol at (i,j)=(" << fi << "," << fj << ") Delta=" << Delta[static_cast<size_t>(fi)][static_cast<size_t>(fj)]
              << "\n";
  } else {
    std::cout << "All elements within tol=" << tol << "\n";
  }

  if (fi < 0) {
    std::cout << "\nPASS: H_cur matches H_ref within tol; production path matches Phi^H H Phi for this case.\n";
    return 0;
  }

  const int ti = fi;
  const int tj = fj;
  std::cout << "\n========== Provenance for H_ref(" << ti << "," << tj << ") ==========\n";
  std::cout << "H_ref_ij = sum_{p,q} conj(Phi_pi) * H_full[p,q] * Phi_qj\n";
  for (int p = 0; p < dim_full; ++p) {
    for (int q = 0; q < dim_full; ++q) {
      const auto hpq = H_full[static_cast<size_t>(p)][static_cast<size_t>(q)];
      if (cabs(hpq) < 1e-15) continue;
      const auto pip = Phi[static_cast<size_t>(p)][static_cast<size_t>(ti)];
      const auto phiq = Phi[static_cast<size_t>(q)][static_cast<size_t>(tj)];
      const auto term = std::conj(pip) * hpq * phiq;
      if (cabs(term) < 1e-15) continue;
      std::cout << "  p=" << p << " q=" << q << " H_pq=" << hpq << " Phi_p,i=" << pip << " Phi_q,j=" << phiq
                << " term=" << term << "\n";
    }
  }
  std::complex<double> sum_ref(0.0, 0.0);
  for (int p = 0; p < dim_full; ++p) {
    for (int q = 0; q < dim_full; ++q) {
      sum_ref += std::conj(Phi[static_cast<size_t>(p)][static_cast<size_t>(ti)]) *
                 H_full[static_cast<size_t>(p)][static_cast<size_t>(q)] *
                 Phi[static_cast<size_t>(q)][static_cast<size_t>(tj)];
    }
  }
  std::cout << "  recomputed sum = " << sum_ref << " (should match H_ref(" << ti << "," << tj << "))\n";

  std::cout << "\n========== Provenance for H_cur(" << ti << "," << tj << ") from apply (column j=" << tj << ") ==========\n";
  std::cout << "Tracing contributions to y[" << ti << "] with x[" << tj << "]=1.\n";

  const auto r_col = kb.representatives[static_cast<size_t>(tj)];
  std::size_t s_col = 1;
  for (const auto& o : map.orbits) {
    if (ftlm::symmetry::pack_raw_state(o.representative) == ftlm::symmetry::pack_raw_state(r_col)) {
      s_col = o.stabilizer.size();
      break;
    }
  }
  const double c_a = ftlm::symmetry::momentum_orbit_normalization_factor_rect(s_col, Lx, Ly);
  std::cout << "  Source column j=" << tj << " rep " << rs(r_col) << " stabilizer=" << s_col << " c_a=" << c_a << "\n";

  for (int gy = 0; gy < Ly; ++gy) {
    for (int gx = 0; gx < Lx; ++gx) {
      const auto rg = ftlm::symmetry::translate_raw_state_rect(r_col, Lx, Ly, gx, gy);
      const auto src_phase = ftlm::symmetry::translation_bloch_phase(K, gx, gy);
      // duplicate fermionic sign from apply (file-local copy of logic for print only)
      auto tsite = [&](int site) {
        const int x = site % Lx;
        const int y = site / Lx;
        return (x + gx) % Lx + Lx * ((y + gy) % Ly);
      };
      std::vector<int> modes;
      for (int site = 0; site < Lx * Ly; ++site) {
        if ((r_col.up >> site) & 1u) modes.push_back(2 * tsite(site) + 0);
        if ((r_col.dn >> site) & 1u) modes.push_back(2 * tsite(site) + 1);
      }
      int inv = 0;
      for (size_t a = 0; a < modes.size(); ++a)
        for (size_t b = a + 1; b < modes.size(); ++b)
          if (modes[a] > modes[b]) ++inv;
      const double src_sign = (inv & 1) ? -1.0 : 1.0;
      const std::complex<double> src_coeff = c_a * src_phase * src_sign;

      for (const auto& hop : hops) {
        const int from = hop.from;
        const int to = hop.to;
        const int spin = hop.spin;
        std::uint16_t upp = rg.up;
        std::uint16_t dnp = rg.dn;
        if (spin == 0) {
          if (((rg.up >> from) & 1u) == 0 || ((rg.up >> to) & 1u)) continue;
          upp = static_cast<std::uint16_t>(rg.up ^ (1u << from) ^ (1u << to));
        } else {
          if (((rg.dn >> from) & 1u) == 0 || ((rg.dn >> to) & 1u)) continue;
          dnp = static_cast<std::uint16_t>(rg.dn ^ (1u << from) ^ (1u << to));
        }
        const int ma = 2 * from + spin;
        const int mb = 2 * to + spin;
        const double jw = (fermions_between_modes16(ma, mb, rg.up, rg.dn) & 1) ? -1.0 : 1.0;
        const std::complex<double> bare = jw * hop.coeff;
        const ftlm::symmetry::RawState jumped{upp, dnp};
        const auto rep_b = ftlm::symmetry::canonical_raw_state_rect(jumped, Lx, Ly);
        const std::size_t ib = kb.index_of(rep_b);
        if (ib == ftlm::symmetry::KBasis::npos) continue;
        if (static_cast<int>(ib) != ti) continue;

        int dx_b = 0, dy_b = 0;
        if (!translation_map_rect(Lx, Ly, rep_b, jumped, &dx_b, &dy_b)) continue;
        std::size_t s_b = 1;
        for (const auto& o : map.orbits) {
          if (ftlm::symmetry::pack_raw_state(o.representative) == ftlm::symmetry::pack_raw_state(rep_b)) {
            s_b = o.stabilizer.size();
            break;
          }
        }
        const double c_b = ftlm::symmetry::momentum_orbit_normalization_factor_rect(s_b, Lx, Ly);
        auto tsite_b = [&](int site) {
          const int x = site % Lx;
          const int y = site / Lx;
          return (x + dx_b) % Lx + Lx * ((y + dy_b) % Ly);
        };
        std::vector<int> modes_b;
        for (int site = 0; site < Lx * Ly; ++site) {
          if ((rep_b.up >> site) & 1u) modes_b.push_back(2 * tsite_b(site) + 0);
          if ((rep_b.dn >> site) & 1u) modes_b.push_back(2 * tsite_b(site) + 1);
        }
        inv = 0;
        for (size_t a = 0; a < modes_b.size(); ++a)
          for (size_t b = a + 1; b < modes_b.size(); ++b)
            if (modes_b[a] > modes_b[b]) ++inv;
        const double bra_sign = (inv & 1) ? -1.0 : 1.0;
        const auto bra_phase = std::conj(ftlm::symmetry::translation_bloch_phase(K, dx_b, dy_b));
        const std::complex<double> contrib = src_coeff * bare * (c_b * bra_phase * bra_sign);

        std::cout << "  path: R_src=(" << gx << "," << gy << ") rg=" << rs(rg) << " hop spin=" << spin << " " << from
                  << "->" << to << " bare=" << bare << "\n";
        std::cout << "        jumped=" << rs(jumped) << " rep_b=" << rs(rep_b) << " ib=" << ib << " R_map=(" << dx_b
                  << "," << dy_b << ")\n";
        std::cout << "        src_phase=" << src_phase << " src_sign=" << src_sign << " c_b=" << c_b
                  << " bra_phase=" << bra_phase << " bra_sign=" << bra_sign << " contrib=" << contrib << "\n";
      }
    }
  }

  std::cout << "\n========== Diagnosis (if this prints, apply still disagrees with reference) ==========\n";
  std::cout << "Reference Phi uses bit translation only (no JW sign on T_R).\n";
  std::cout << "If H_cur != H_ref: typical bug was extra fermionic_translation_sign_rect on src/bra in apply().\n";
  std::cout << "Classification: (c) wrong fermionic sign on **translation** (JW on hop is in `bare` only).\n";
  std::cout << "Expected fix location: src/hubbard_momentum_action.cpp HubbardMomentumAction::apply\n";

  return (max_d < tol) ? 0 : 1;
}
