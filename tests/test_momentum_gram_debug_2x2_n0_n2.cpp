// Debug-only: Gram-matrix reference path for 2x2, sector (Nup,Ndn)=(0,2).
// Do not use in production. See user Steps 1–8 in momentum refactor debugging plan.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
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
#include "ftlm/symmetry/translation_projector_dense.hpp"

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

int translated_site_rect(int site, int lx, int ly, int dx, int dy) {
  const int x = site % lx;
  const int y = site / lx;
  const int tx = (x + dx) % lx;
  const int ty = (y + dy) % ly;
  return tx + lx * ty;
}

// Fermion sign for literal diagnostic (not used in production Phi).
double fermionic_translation_sign_rect(ftlm::symmetry::RawState s, int lx, int ly, int dx, int dy) {
  std::vector<int> transformed_modes;
  transformed_modes.reserve(2 * static_cast<std::size_t>(lx * ly));
  for (int site = 0; site < lx * ly; ++site) {
    if ((s.up >> site) & 1u) {
      const int tsite = translated_site_rect(site, lx, ly, dx, dy);
      transformed_modes.push_back(2 * tsite + 0);
    }
    if ((s.dn >> site) & 1u) {
      const int tsite = translated_site_rect(site, lx, ly, dx, dy);
      transformed_modes.push_back(2 * tsite + 1);
    }
  }
  int inv = 0;
  for (std::size_t i = 0; i < transformed_modes.size(); ++i) {
    for (std::size_t j = i + 1; j < transformed_modes.size(); ++j) {
      if (transformed_modes[i] > transformed_modes[j]) {
        ++inv;
      }
    }
  }
  return (inv & 1) ? -1.0 : 1.0;
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
  if (M.empty()) {
    std::cout << name << " (0x0)\n";
    return;
  }
  std::cout << name << " (" << M.size() << "x" << M[0].size() << ")\n" << std::setprecision(10);
  for (size_t i = 0; i < M.size(); ++i) {
    for (size_t j = 0; j < M[i].size(); ++j) {
      const auto z = M[i][j];
      std::cout << "(" << z.real() << (z.imag() >= 0 ? "+" : "") << z.imag() << "i)";
      if (j + 1 < M[i].size()) std::cout << " ";
    }
    std::cout << "\n";
  }
}

double cnorm(std::complex<double> z) { return std::sqrt(std::norm(z)); }

void jacobi_symmetric(std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& V, int max_sweeps,
                      double tol_offdiag) {
  const int n = static_cast<int>(A.size());
  V.assign(static_cast<size_t>(n), std::vector<double>(static_cast<size_t>(n), 0.0));
  for (int i = 0; i < n; ++i) V[static_cast<size_t>(i)][static_cast<size_t>(i)] = 1.0;

  for (int sweep = 0; sweep < max_sweeps; ++sweep) {
    int p = 0;
    int q = 1;
    double max_abs = 0.0;
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        const double v = std::abs(A[static_cast<size_t>(i)][static_cast<size_t>(j)]);
        if (v > max_abs) {
          max_abs = v;
          p = i;
          q = j;
        }
      }
    }
    if (max_abs < tol_offdiag) {
      break;
    }
    const double app = A[static_cast<size_t>(p)][static_cast<size_t>(p)];
    const double aqq = A[static_cast<size_t>(q)][static_cast<size_t>(q)];
    const double apq = A[static_cast<size_t>(p)][static_cast<size_t>(q)];
    const double phi = 0.5 * std::atan2(2.0 * apq, aqq - app);
    const double c = std::cos(phi);
    const double s = std::sin(phi);

    for (int k = 0; k < n; ++k) {
      if (k == p || k == q) {
        continue;
      }
      const double apk = A[static_cast<size_t>(p)][static_cast<size_t>(k)];
      const double aqk = A[static_cast<size_t>(q)][static_cast<size_t>(k)];
      const double rpk = c * apk - s * aqk;
      const double rqk = c * aqk + s * apk;
      A[static_cast<size_t>(p)][static_cast<size_t>(k)] = rpk;
      A[static_cast<size_t>(k)][static_cast<size_t>(p)] = rpk;
      A[static_cast<size_t>(q)][static_cast<size_t>(k)] = rqk;
      A[static_cast<size_t>(k)][static_cast<size_t>(q)] = rqk;
    }
    const double new_pp = c * c * app - 2.0 * s * c * apq + s * s * aqq;
    const double new_qq = s * s * app + 2.0 * s * c * apq + c * c * aqq;
    A[static_cast<size_t>(p)][static_cast<size_t>(q)] = 0.0;
    A[static_cast<size_t>(q)][static_cast<size_t>(p)] = 0.0;
    A[static_cast<size_t>(p)][static_cast<size_t>(p)] = new_pp;
    A[static_cast<size_t>(q)][static_cast<size_t>(q)] = new_qq;

    for (int k = 0; k < n; ++k) {
      const double vpk = V[static_cast<size_t>(k)][static_cast<size_t>(p)];
      const double vqk = V[static_cast<size_t>(k)][static_cast<size_t>(q)];
      V[static_cast<size_t>(k)][static_cast<size_t>(p)] = c * vpk - s * vqk;
      V[static_cast<size_t>(k)][static_cast<size_t>(q)] = c * vqk + s * vpk;
    }
  }
}

/// Real 2n×2n embedding of Hermitian H; eigenvalues appear in pairs.
void hermitian_to_real2(const std::vector<std::vector<std::complex<double>>>& H,
                        std::vector<std::vector<double>>& R) {
  const int n = static_cast<int>(H.size());
  const int m = 2 * n;
  R.assign(static_cast<size_t>(m), std::vector<double>(static_cast<size_t>(m), 0.0));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      const auto z = H[static_cast<size_t>(i)][static_cast<size_t>(j)];
      const double re = z.real();
      const double im = z.imag();
      R[static_cast<size_t>(i)][static_cast<size_t>(j)] = re;
      R[static_cast<size_t>(i)][static_cast<size_t>(j + n)] = -im;
      R[static_cast<size_t>(i + n)][static_cast<size_t>(j)] = im;
      R[static_cast<size_t>(i + n)][static_cast<size_t>(j + n)] = re;
    }
  }
}

/// Returns n eigenvalues (each real pair averaged) and n orthonormal complex eigenvectors (columns of U).
void eigh_hermitian_from_real2(
    const std::vector<std::vector<std::complex<double>>>& H,
    std::vector<double>& evals,
    std::vector<std::vector<std::complex<double>>>& evec_columns,
    double tol_pair = 1e-8) {
  const int n = static_cast<int>(H.size());
  std::vector<std::vector<double>> R;
  hermitian_to_real2(H, R);
  std::vector<std::vector<double>> V;
  jacobi_symmetric(R, V, std::max(8000, 200 * n * n), 1e-14);

  const int m = 2 * n;
  std::vector<double> d(static_cast<size_t>(m));
  for (int i = 0; i < m; ++i) {
    d[static_cast<size_t>(i)] = R[static_cast<size_t>(i)][static_cast<size_t>(i)];
  }
  std::vector<int> ord(static_cast<size_t>(m));
  std::iota(ord.begin(), ord.end(), 0);
  std::sort(ord.begin(), ord.end(),
            [&](int a, int b) { return d[static_cast<size_t>(a)] < d[static_cast<size_t>(b)]; });

  evals.clear();
  evec_columns.assign(static_cast<size_t>(n), std::vector<std::complex<double>>(static_cast<size_t>(n), 0.0));

  for (int k = 0; k < n; ++k) {
    const int ia = ord[static_cast<size_t>(2 * k)];
    const int ib = ord[static_cast<size_t>(2 * k + 1)];
    const double la = d[static_cast<size_t>(ia)];
    const double lb = d[static_cast<size_t>(ib)];
    if (std::abs(la - lb) > tol_pair * (1.0 + std::max(std::abs(la), std::abs(lb)))) {
      std::cerr << "[Gram debug] WARNING: paired eigenvalues differ: " << la << " vs " << lb << "\n";
    }
    evals.push_back(0.5 * (la + lb));

    std::vector<std::complex<double>> z(static_cast<size_t>(n), 0.0);
    double norm2 = 0.0;
    for (int i = 0; i < n; ++i) {
      const double ur = V[static_cast<size_t>(i)][static_cast<size_t>(ia)];
      const double ui = V[static_cast<size_t>(i + n)][static_cast<size_t>(ia)];
      z[static_cast<size_t>(i)] = {ur, ui};
      norm2 += ur * ur + ui * ui;
    }
    const double invn = 1.0 / std::sqrt(std::max(norm2, 1e-300));
    for (int i = 0; i < n; ++i) {
      z[static_cast<size_t>(i)] *= invn;
    }
    for (int i = 0; i < n; ++i) {
      evec_columns[static_cast<size_t>(i)][static_cast<size_t>(k)] = z[static_cast<size_t>(i)];
    }
  }
}

std::vector<std::vector<std::complex<double>>> matmul_c(
    const std::vector<std::vector<std::complex<double>>>& A,
    const std::vector<std::vector<std::complex<double>>>& B) {
  const int n = static_cast<int>(A.size());
  const int k = static_cast<int>(B.size());
  const int p = static_cast<int>(B[0].size());
  std::vector<std::vector<std::complex<double>>> C(static_cast<size_t>(n),
                                                   std::vector<std::complex<double>>(static_cast<size_t>(p), 0.0));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < p; ++j) {
      std::complex<double> sum(0.0, 0.0);
      for (int t = 0; t < k; ++t) {
        sum += A[static_cast<size_t>(i)][static_cast<size_t>(t)] * B[static_cast<size_t>(t)][static_cast<size_t>(j)];
      }
      C[static_cast<size_t>(i)][static_cast<size_t>(j)] = sum;
    }
  }
  return C;
}

/// C = A^H B with A: n×ma, B: n×mb -> C ma×mb, C[i][j] = sum_k conj(A[k][i]) B[k][j]
std::vector<std::vector<std::complex<double>>> matmul_AhB(const std::vector<std::vector<std::complex<double>>>& A,
                                                          const std::vector<std::vector<std::complex<double>>>& B) {
  const int n = static_cast<int>(A.size());
  const int ma = static_cast<int>(A[0].size());
  const int mb = static_cast<int>(B[0].size());
  std::vector<std::vector<std::complex<double>>> C(static_cast<size_t>(ma),
                                                   std::vector<std::complex<double>>(static_cast<size_t>(mb), 0.0));
  for (int i = 0; i < ma; ++i) {
    for (int j = 0; j < mb; ++j) {
      std::complex<double> sum(0.0, 0.0);
      for (int k = 0; k < n; ++k) {
        sum += std::conj(A[static_cast<size_t>(k)][static_cast<size_t>(i)]) * B[static_cast<size_t>(k)][static_cast<size_t>(j)];
      }
      C[static_cast<size_t>(i)][static_cast<size_t>(j)] = sum;
    }
  }
  return C;
}

/// C = A B with A ma×na, B na×mb
std::vector<std::vector<std::complex<double>>> matmul_AB(const std::vector<std::vector<std::complex<double>>>& A,
                                                         const std::vector<std::vector<std::complex<double>>>& B) {
  const int ma = static_cast<int>(A.size());
  const int na = static_cast<int>(A[0].size());
  const int mb = static_cast<int>(B[0].size());
  std::vector<std::vector<std::complex<double>>> C(static_cast<size_t>(ma),
                                                   std::vector<std::complex<double>>(static_cast<size_t>(mb), 0.0));
  for (int i = 0; i < ma; ++i) {
    for (int j = 0; j < mb; ++j) {
      std::complex<double> sum(0.0, 0.0);
      for (int t = 0; t < na; ++t) {
        sum += A[static_cast<size_t>(i)][static_cast<size_t>(t)] * B[static_cast<size_t>(t)][static_cast<size_t>(j)];
      }
      C[static_cast<size_t>(i)][static_cast<size_t>(j)] = sum;
    }
  }
  return C;
}

double max_abs_Hermitian_offdiag(const std::vector<std::vector<std::complex<double>>>& M) {
  const int n = static_cast<int>(M.size());
  double m = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (i == j) {
        continue;
      }
      m = std::max(m, cnorm(M[static_cast<size_t>(i)][static_cast<size_t>(j)]));
    }
  }
  return m;
}

double max_deviation_from_identity(const std::vector<std::vector<std::complex<double>>>& M) {
  const int n = static_cast<int>(M.size());
  double m = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      const std::complex<double> want = (i == j) ? std::complex<double>(1.0, 0.0) : std::complex<double>(0.0, 0.0);
      m = std::max(m, cnorm(M[static_cast<size_t>(i)][static_cast<size_t>(j)] - want));
    }
  }
  return m;
}

std::vector<std::vector<std::complex<double>>> make_identity_c(int n) {
  std::vector<std::vector<std::complex<double>>> I(
      static_cast<size_t>(n), std::vector<std::complex<double>>(static_cast<size_t>(n), 0.0));
  for (int i = 0; i < n; ++i) {
    I[static_cast<size_t>(i)][static_cast<size_t>(i)] = std::complex<double>(1.0, 0.0);
  }
  return I;
}

/// Hermitian PSD inverse square root; uses identity when G ≈ I (EVD pairing fails for degenerate 2n embedding).
std::vector<std::vector<std::complex<double>>> gram_inverse_sqrt(
    const std::vector<std::vector<std::complex<double>>>& G, double tol_identity = 1e-9) {
  const int n = static_cast<int>(G.size());
  if (max_deviation_from_identity(G) < tol_identity) {
    return make_identity_c(n);
  }
  std::vector<double> g_evals;
  std::vector<std::vector<std::complex<double>>> g_evecs;
  eigh_hermitian_from_real2(G, g_evals, g_evecs, 1e-8);
  const double ge_max = g_evals.empty() ? 0.0 : *std::max_element(g_evals.begin(), g_evals.end());
  const double tol_null = 1e-10 * std::max(1.0, ge_max);
  std::vector<std::vector<std::complex<double>>> Ginvh(static_cast<size_t>(n),
                                                       std::vector<std::complex<double>>(static_cast<size_t>(n), 0.0));
  for (size_t t = 0; t < g_evals.size(); ++t) {
    if (g_evals[t] <= tol_null) {
      continue;
    }
    const double inv_sqrt = 1.0 / std::sqrt(g_evals[t]);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        Ginvh[static_cast<size_t>(i)][static_cast<size_t>(j)] +=
            inv_sqrt * g_evecs[static_cast<size_t>(i)][static_cast<size_t>(t)] *
            std::conj(g_evecs[static_cast<size_t>(j)][static_cast<size_t>(t)]);
      }
    }
  }
  return Ginvh;
}

double max_hermitian_symmetry_error(const std::vector<std::vector<std::complex<double>>>& M) {
  const int n = static_cast<int>(M.size());
  double m = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      const auto d = M[static_cast<size_t>(i)][static_cast<size_t>(j)] -
                     std::conj(M[static_cast<size_t>(j)][static_cast<size_t>(i)]);
      m = std::max(m, cnorm(d));
    }
  }
  return m;
}

// Eigenvalues of Hermitian H (n×n) via real embedding + Jacobi (duplicated evals).
std::vector<double> eigval_hermitian(const std::vector<std::vector<std::complex<double>>>& H) {
  std::vector<double> ev;
  std::vector<std::vector<std::complex<double>>> vecs;
  eigh_hermitian_from_real2(H, ev, vecs, 1e-7);
  std::sort(ev.begin(), ev.end());
  return ev;
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
  const std::vector<double> eig_full = eigval_hermitian(H_full);

  std::cout << "=== Gram-matrix debug: 2x2 sector (Nup,Ndn)=(" << Nup << "," << Ndn << "), dim_full=" << dim_full
            << " ===\n";
  std::cout << "\n[Step 1] Full-sector eigenvalues (reference ED, sorted):\n";
  for (size_t i = 0; i < eig_full.size(); ++i) {
    std::cout << "  lam[" << i << "]=" << eig_full[i] << "\n";
  }

  std::vector<double> concat_ref;
  double first_mismatch_cur_vs_gram = -1.0;
  int first_mismatch_kx = -1;
  int first_mismatch_ky = -1;
  double max_global_spec_diff = std::numeric_limits<double>::quiet_NaN();

  auto stabilizer_for_rep = [&](ftlm::symmetry::RawState r) -> std::size_t {
    for (const auto& o : map.orbits) {
      if (ftlm::symmetry::pack_raw_state(o.representative) == ftlm::symmetry::pack_raw_state(r)) {
        return o.stabilizer.size();
      }
    }
    return 1;
  };

  for (int ky = 0; ky < Ly; ++ky) {
    for (int kx = 0; kx < Lx; ++kx) {
      const ftlm::symmetry::MomentumSector K{kx, ky, Lx, Ly};
      const auto kb = ftlm::symmetry::KBasis::build(map, K);
      const int dk = static_cast<int>(hub.momentum_block_dim(map, K, Nup, Ndn));
      if (dk <= 0) {
        continue;
      }

      std::cout << "\n========== K=(" << kx << "," << ky << "), dense P_k rank dk=" << dk
                << " (KBasis.dim=" << kb.dim() << ") ==========\n";

      std::vector<std::complex<double>> phi_cm;
      std::size_t dk_phi = 0;
      ftlm::symmetry::detail::build_phi_from_translation_projector_dense(fb, Lx, Ly, K, &phi_cm, &dk_phi);
      if (static_cast<int>(dk_phi) != dk) {
        std::cerr << "Gram debug: dk mismatch build_phi vs momentum_block_dim\n";
        return 2;
      }
      std::vector<std::vector<std::complex<double>>> Phi(
          static_cast<size_t>(dim_full), std::vector<std::complex<double>>(static_cast<size_t>(dk), 0.0));
      for (int j = 0; j < dk; ++j) {
        for (int i = 0; i < dim_full; ++i) {
          Phi[static_cast<size_t>(i)][static_cast<size_t>(j)] =
              phi_cm[static_cast<size_t>(i) + static_cast<size_t>(j) * static_cast<size_t>(dim_full)];
        }
      }
      std::cout << "[Step 2–3] Phi columns = orthonormal spectral basis of Hermitian P_k (production path).\n";

      // Step 3: Gram matrix G = Phi^H Phi
      std::vector<std::vector<std::complex<double>>> G(static_cast<size_t>(dk),
                                                       std::vector<std::complex<double>>(static_cast<size_t>(dk), 0.0));
      for (int a = 0; a < dk; ++a) {
        for (int b = 0; b < dk; ++b) {
          std::complex<double> s(0.0, 0.0);
          for (int i = 0; i < dim_full; ++i) {
            s += std::conj(Phi[static_cast<size_t>(i)][static_cast<size_t>(a)]) *
                 Phi[static_cast<size_t>(i)][static_cast<size_t>(b)];
          }
          G[static_cast<size_t>(a)][static_cast<size_t>(b)] = s;
        }
      }
      std::cout << "\n[Step 3] Gram matrix G = Phi^H Phi\n";
      print_cmatrix("G", G);
      std::cout << "  max Hermitian symmetry error: " << max_hermitian_symmetry_error(G) << "\n";
      std::cout << "  max off-diagonal |G_ij| (all i!=j): " << max_abs_Hermitian_offdiag(G) << "\n";

      std::vector<double> g_evals;
      std::vector<std::vector<std::complex<double>>> g_evecs;
      eigh_hermitian_from_real2(G, g_evals, g_evecs, 1e-8);
      std::cout << "  Eigenvalues of G (ascending):\n";
      int rank = 0;
      const double ge_max = g_evals.empty() ? 0.0 : *std::max_element(g_evals.begin(), g_evals.end());
      const double tol_null = 1e-10 * std::max(1.0, ge_max);
      for (size_t t = 0; t < g_evals.size(); ++t) {
        std::cout << "    lambda_G[" << t << "]=" << g_evals[t] << "\n";
        if (g_evals[t] > tol_null) {
          ++rank;
        }
      }
      std::cout << "  numeric rank (lambda > " << tol_null << "): " << rank << "/" << dk << "\n";

      // Step 4: G^{-1/2} (identity fast path when G ≈ I avoids broken degenerate EVD pairing)
      const std::vector<std::vector<std::complex<double>>> Ginvh = gram_inverse_sqrt(G);
      if (max_deviation_from_identity(G) < 1e-9) {
        std::cout << "  G ≈ I: using G^{-1/2} = I (skip degenerate 2n-embedding EVD)\n";
      }

      auto Psi = matmul_c(Phi, Ginvh);
      // Drop columns that are ~0 (null of G)
      std::vector<int> active_cols;
      for (int c = 0; c < dk; ++c) {
        double coln = 0.0;
        for (int i = 0; i < dim_full; ++i) {
          coln += std::norm(Psi[static_cast<size_t>(i)][static_cast<size_t>(c)]);
        }
        if (coln > tol_null * tol_null) {
          active_cols.push_back(c);
        }
      }
      int rdim = static_cast<int>(active_cols.size());
      std::cout << "\n[Step 4] After G^{-1/2}: active orthonormal columns rdim=" << rdim << " (from dk=" << dk << ")\n";
      std::vector<std::vector<std::complex<double>>> Psi_red(
          static_cast<size_t>(dim_full), std::vector<std::complex<double>>(static_cast<size_t>(rdim), 0.0));
      for (int c = 0; c < rdim; ++c) {
        const int oc = active_cols[static_cast<size_t>(c)];
        for (int i = 0; i < dim_full; ++i) {
          Psi_red[static_cast<size_t>(i)][static_cast<size_t>(c)] =
              Psi[static_cast<size_t>(i)][static_cast<size_t>(oc)];
        }
      }
      // Normalize columns (numerical cleanup)
      for (int c = 0; c < rdim; ++c) {
        double nn = 0.0;
        for (int i = 0; i < dim_full; ++i) {
          nn += std::norm(Psi_red[static_cast<size_t>(i)][static_cast<size_t>(c)]);
        }
        const double scale = 1.0 / std::sqrt(nn);
        for (int i = 0; i < dim_full; ++i) {
          Psi_red[static_cast<size_t>(i)][static_cast<size_t>(c)] *= scale;
        }
      }
      {
        std::vector<std::vector<std::complex<double>>> overlap =
            matmul_AhB(Psi_red, Psi_red);
        double max_id = 0.0;
        for (int i = 0; i < rdim; ++i) {
          for (int j = 0; j < rdim; ++j) {
            const std::complex<double> want =
                (i == j) ? std::complex<double>(1.0, 0.0) : std::complex<double>(0.0, 0.0);
            max_id = std::max(max_id, cnorm(overlap[static_cast<size_t>(i)][static_cast<size_t>(j)] - want));
          }
        }
        std::cout << "  max |(Psi_red^H Psi_red) - I|: " << max_id << "\n";
      }

      // Step 5: H_proj = Psi_red^H H_full Psi_red
      auto H_Psi = matmul_c(H_full, Psi_red);
      auto H_proj = matmul_AhB(Psi_red, H_Psi);
      std::cout << "\n[Step 5] H_proj = Psi^H H_full Psi (Gram-orthonormal basis), size " << rdim << "x" << rdim << "\n";
      print_cmatrix("H_proj_gram", H_proj);
      const std::vector<double> eig_k_gram = eigval_hermitian(H_proj);
      std::cout << "  eigenvalues (Gram block):\n";
      for (size_t i = 0; i < eig_k_gram.size(); ++i) {
        std::cout << "    " << eig_k_gram[i] << "\n";
      }
      for (double lam : eig_k_gram) {
        concat_ref.push_back(lam);
      }

      // Current momentum implementation H_cur (dk×dk)
      std::vector<std::vector<std::complex<double>>> H_cur(
          static_cast<size_t>(dk), std::vector<std::complex<double>>(static_cast<size_t>(dk), 0.0));
      std::vector<std::complex<double>> x(static_cast<size_t>(dk), 0.0), y(static_cast<size_t>(dk), 0.0);
      for (int j = 0; j < dk; ++j) {
        std::fill(x.begin(), x.end(), std::complex<double>(0.0, 0.0));
        x[static_cast<size_t>(j)] = {1.0, 0.0};
        hub.apply(map, K, Nup, Ndn, x.data(), y.data());
        for (int i = 0; i < dk; ++i) {
          H_cur[static_cast<size_t>(i)][static_cast<size_t>(j)] = y[static_cast<size_t>(i)];
        }
      }
      std::cout << "\n[Step 6] Current HubbardMomentumAction block H_cur (" << dk << "x" << dk << ")\n";
      print_cmatrix("H_cur", H_cur);

      // Raw matrix in non-orthogonal phi basis: H_tilde = Phi^H H Phi.
      // Orthonormalized basis: H_orth = G^{-1/2} H_tilde G^{-1/2} must match H_cur for a correct implementation.
      auto H_tilde = matmul_AhB(Phi, matmul_c(H_full, Phi));
      double max_diff_vs_orth = 0.0;
      if (rank == dk) {
        auto H_orth_from_tilde = matmul_AB(matmul_AB(Ginvh, H_tilde), Ginvh);
        for (int i = 0; i < dk; ++i) {
          for (int j = 0; j < dk; ++j) {
            max_diff_vs_orth = std::max(
                max_diff_vs_orth,
                cnorm(H_orth_from_tilde[static_cast<size_t>(i)][static_cast<size_t>(j)] -
                      H_cur[static_cast<size_t>(i)][static_cast<size_t>(j)]));
          }
        }
        std::cout << "\n  [Step 6] max |H_cur - G^{-1/2} Phi^H H Phi G^{-1/2}|: " << max_diff_vs_orth << "\n";
        if (first_mismatch_cur_vs_gram < 0 && max_diff_vs_orth > 1e-8) {
          first_mismatch_cur_vs_gram = max_diff_vs_orth;
          first_mismatch_kx = kx;
          first_mismatch_ky = ky;
        }
      } else {
        std::cout << "\n  G is rank-deficient; skip H_cur vs G^{-1/2} H_tilde G^{-1/2} elementwise comparison.\n";
      }

      // Step 7: phase convention sample (first rep, first hopping that connects)
      std::cout << "\n[Step 7] Phase convention sample (representative j=0): ";
      if (dk > 0) {
        const auto r0 = kb.representatives[0];
        std::cout << rs(r0) << "\n";
        for (const auto& hop : hops) {
          const int from = hop.from;
          const int to = hop.to;
          const int spin = hop.spin;
          std::uint16_t upp = r0.up, dnp = r0.dn;
          if (spin == 0) {
            if (((r0.up >> from) & 1u) == 0 || ((r0.up >> to) & 1u)) {
              continue;
            }
            upp = static_cast<std::uint16_t>(r0.up ^ (1u << from) ^ (1u << to));
          } else {
            if (((r0.dn >> from) & 1u) == 0 || ((r0.dn >> to) & 1u)) {
              continue;
            }
            dnp = static_cast<std::uint16_t>(r0.dn ^ (1u << from) ^ (1u << to));
          }
          const auto s = ftlm::symmetry::RawState{upp, dnp};
          const auto rp = ftlm::symmetry::canonical_raw_state_rect(s, Lx, Ly);
          int dx = 0, dy = 0;
          bool found = false;
          for (int ey = 0; ey < Ly && !found; ++ey) {
            for (int ex = 0; ex < Lx && !found; ++ex) {
              if (ftlm::symmetry::translate_raw_state_rect(rp, Lx, Ly, ex, ey) == s) {
                dx = ex;
                dy = ey;
                found = true;
              }
            }
          }
          const auto phase_code = std::conj(ftlm::symmetry::translation_bloch_phase(K, dx, dy));
          // reference Bloch sum phase for |r0,K> -> |r',K>: coefficient ratio involves
          // conj(phase(rep_b)) * phase(rep_a) from bra/ket projection — print bare phases
          std::cout << "  hop spin=" << spin << " " << from << "->" << to << " coeff=" << hop.coeff << "\n";
          std::cout << "    r=" << rs(r0) << " s(hopped)=" << rs(s) << " r'(canon)=" << rs(rp) << " R(map r'->s)=(" << dx
                    << "," << dy << ")\n";
          std::cout << "    translation_bloch_phase(K,R)=exp(-iK.R)=" << ftlm::symmetry::translation_bloch_phase(K, dx, dy)
                    << "\n";
          std::cout << "    current hub uses conj(phase) on rep map = " << phase_code << "\n";
          break;
        }
      }
    }
  }

  // --- Global: stack all code-matched columns U_all (dim x dim); Gram and full projected H ---
  {
    std::vector<std::vector<std::complex<double>>> U_all(
        static_cast<size_t>(dim_full), std::vector<std::complex<double>>(static_cast<size_t>(dim_full), 0.0));
    int col_g = 0;
    for (int ky = 0; ky < Ly; ++ky) {
      for (int kx = 0; kx < Lx; ++kx) {
        const ftlm::symmetry::MomentumSector Kg{kx, ky, Lx, Ly};
        const auto kb_g = ftlm::symmetry::KBasis::build(map, Kg);
        const int dkg = static_cast<int>(kb_g.dim());
        for (int j = 0; j < dkg; ++j) {
          const auto r = kb_g.representatives[static_cast<size_t>(j)];
          const std::size_t stab = stabilizer_for_rep(r);
          const double c_g =
              std::sqrt(1.0 / (static_cast<double>(Lx * Ly) * static_cast<double>(stab)));
          for (int ey = 0; ey < Ly; ++ey) {
            for (int ex = 0; ex < Lx; ++ex) {
              const auto st = ftlm::symmetry::translate_raw_state_rect(r, Lx, Ly, ex, ey);
              const int idx = fb.index_of(st.up, st.dn);
              if (idx < 0) {
                continue;
              }
              const auto ph = ftlm::symmetry::translation_bloch_phase(Kg, ex, ey);
              U_all[static_cast<size_t>(idx)][static_cast<size_t>(col_g)] += c_g * std::conj(ph);
            }
          }
          ++col_g;
        }
      }
    }
    std::cout << "\n========== [Global stack] code-matched Bloch matrix U (" << dim_full << "x" << col_g << ") ==========\n";
    if (col_g != dim_full) {
      std::cout << "ERROR: column count " << col_g << " != dim_full " << dim_full << "\n";
    }
    auto G_all = matmul_AhB(U_all, U_all);
    std::cout << "  G_all = U^H U: max offdiag |(U^H U)_ij|, i!=j: " << max_abs_Hermitian_offdiag(G_all) << "\n";
    std::cout << "  Hermitian symmetry error: " << max_hermitian_symmetry_error(G_all) << "\n";

    const std::vector<std::vector<std::complex<double>>> G_all_invh = gram_inverse_sqrt(G_all);
    if (max_deviation_from_identity(G_all) < 1e-9) {
      std::cout << "  G_all ≈ I: G_all^{-1/2} = I\n";
    }
    auto H_t_all_base = matmul_AhB(U_all, matmul_c(H_full, U_all));
    const std::vector<int> part = {3, 1, 1, 1};
    std::vector<int> offs(static_cast<size_t>(part.size() + 1), 0);
    for (size_t t = 0; t < part.size(); ++t) {
      offs[t + 1] = offs[t] + part[t];
    }
    double max_offblock = 0.0;
    for (size_t bi = 0; bi < part.size(); ++bi) {
      for (size_t bj = 0; bj < part.size(); ++bj) {
        if (bi == bj) {
          continue;
        }
        for (int p = offs[bi]; p < offs[bi + 1]; ++p) {
          for (int q = offs[bj]; q < offs[bj + 1]; ++q) {
            max_offblock = std::max(
                max_offblock, cnorm(H_t_all_base[static_cast<size_t>(p)][static_cast<size_t>(q)]));
          }
        }
      }
    }
    std::cout << "  max |(U^H H U)_pq| coupling different K-blocks (3+1+1+1 partition): " << max_offblock << "\n";

    auto H_glob = matmul_AB(matmul_AB(G_all_invh, H_t_all_base), G_all_invh);
    const std::vector<double> eig_glob = eigval_hermitian(H_glob);
    std::cout << "  Eigenvalues of G_all^{-1/2} U^H H U G_all^{-1/2} (global orthonormalized stack), sorted:\n";
    for (double lam : eig_glob) {
      std::cout << "    " << lam << "\n";
    }
    double mxg = 0.0;
    if (eig_glob.size() == eig_full.size()) {
      for (size_t i = 0; i < eig_full.size(); ++i) {
        mxg = std::max(mxg, std::abs(eig_glob[i] - eig_full[i]));
      }
    }
    std::cout << "  max abs diff vs full-sector ED: " << mxg << "\n";
    if (mxg < 1e-6) {
      std::cout << "  => Per-K eigenvalue concatenation can still differ; use this global spectrum as Gram reference.\n";
    }
    max_global_spec_diff = mxg;
  }

  std::sort(concat_ref.begin(), concat_ref.end());
  std::cout << "\n========== [Step 5 cont.] Concatenated Gram-block spectra (all K), sorted ==========\n";
  for (size_t i = 0; i < concat_ref.size(); ++i) {
    std::cout << "  " << concat_ref[i] << "\n";
  }
  double max_eig_diff = 0.0;
  if (concat_ref.size() == eig_full.size()) {
    for (size_t i = 0; i < eig_full.size(); ++i) {
      max_eig_diff = std::max(max_eig_diff, std::abs(concat_ref[i] - eig_full[i]));
    }
  }
  std::cout << "\n[Step 5 / report] max abs eigenvalue diff |eig_sorted_concat_gram - eig_full|: " << max_eig_diff << "\n";

  std::cout << "\n========== Step 8 summary ==========\n";
  std::cout << "- Per-K: literal fermionic sum_R e^{-iK·R} T_R|r> is ~0 for every rep here (JW signs cancel);\n";
  std::cout << "  code-matched lift (bit translate + c*conj(phase)) gives G=I within K and U^H U=I globally.\n";
  std::cout << "- Even so, |(U^H H U)| between distinct K-blocks can be O(1) (printed above): columns are not\n";
  std::cout << "  simultaneous momentum eigenstates of the same H; do not concatenate per-K ED and expect full spectrum.\n";
  std::cout << "- Concatenating per-K block eigenvalues matches full ED: "
            << (max_eig_diff < 1e-6 ? "YES (within 1e-6)" : "NO") << "\n";
  std::cout << "- Global stacked U (code-matched) + G^{-1/2} U^H H U G^{-1/2} vs ED: "
            << (std::isfinite(max_global_spec_diff) && max_global_spec_diff < 1e-6 ? "YES (within 1e-6)" : "NO")
            << "  max_diff=" << max_global_spec_diff << "\n";
  if (first_mismatch_cur_vs_gram >= 0) {
    std::cout << "- First notable |H_cur - G^{-1/2} Phi^H H Phi G^{-1/2}| at K=(" << first_mismatch_kx << ","
              << first_mismatch_ky << "): max_abs=" << first_mismatch_cur_vs_gram << "\n";
  }
  std::cout << "- Production fix target: src/hubbard_momentum_action.cpp HubbardMomentumAction::apply\n";
  std::cout << "  until H_cur matches G^{-1/2} (Phi^H H Phi) G^{-1/2} with raw phi columns in Phi.\n";

  const bool ok =
      std::isfinite(max_global_spec_diff) && max_global_spec_diff < 1e-6;
  return ok ? 0 : 1;
}
