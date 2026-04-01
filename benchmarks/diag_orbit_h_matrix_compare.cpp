// Dense matrix comparison for one K block: exact orbit-basis H vs apply_orbit_k_block per-hop formula.
// Block: 3x2, (nu,nd)=(2,3), K=(1,0), dim_full=300, dk=50.
//
// H_prod = Phi^dagger H Phi (Gram) from HubbardMomentumBlock::apply on each unit vector.
// T = Phi^dagger O with O columns = fill_phi_orbit_bloch_from_seed per orbit rep.
// H_orbit_exact = T^{-1} H_prod T  (orbit coords x map to Gram coords T x).
// H_orbit_formula[j] = apply_orbit_k_block(e_j) column-wise.

#include "ftlm/fock_basis.hpp"
#include "ftlm/hubbard_params.hpp"
#include "ftlm/symmetry/hubbard_momentum_action.hpp"
#include "ftlm/symmetry/momentum_sector.hpp"
#include "ftlm/symmetry/orbit_bloch_phi.hpp"
#include "ftlm/symmetry/orbit_k_block_basis.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using C = std::complex<double>;

// Column-major nrows x ncols: index(i,j) = i + j * nrows
void swap_rows_cm(int nrows, int ncols, C* a, int r1, int r2) {
  if (r1 == r2) {
    return;
  }
  for (int j = 0; j < ncols; ++j) {
    std::swap(a[r1 + j * nrows], a[r2 + j * nrows]);
  }
}

// In-place inverse of n x n column-major matrix A (overwritten by A^{-1}).
bool invert_colmajor_gauss_jordan(int n, std::vector<C>* a) {
  const int ncol = 2 * n;
  std::vector<C> aug(static_cast<std::size_t>(n * ncol), C(0.0, 0.0));
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      aug[static_cast<std::size_t>(i + j * n)] = (*a)[static_cast<std::size_t>(i + j * n)];
    }
  }
  for (int j = 0; j < n; ++j) {
    aug[static_cast<std::size_t>(j + (n + j) * n)] = C(1.0, 0.0);
  }
  // Column-major: aug[row + col * nrows] with nrows = n.
  for (int col = 0; col < n; ++col) {
    int piv = col;
    double best = std::abs(aug[static_cast<std::size_t>(col + col * n)]);
    for (int r = col + 1; r < n; ++r) {
      const double v = std::abs(aug[static_cast<std::size_t>(r + col * n)]);
      if (v > best) {
        best = v;
        piv = r;
      }
    }
    if (best < 1e-18) {
      return false;
    }
    swap_rows_cm(n, ncol, aug.data(), col, piv);
    const C invd = C(1.0, 0.0) / aug[static_cast<std::size_t>(col + col * n)];
    for (int j = 0; j < ncol; ++j) {
      aug[static_cast<std::size_t>(col + j * n)] *= invd;
    }
    for (int r = 0; r < n; ++r) {
      if (r == col) {
        continue;
      }
      const C f = aug[static_cast<std::size_t>(r + col * n)];
      if (std::abs(f) < 1e-28) {
        continue;
      }
      for (int j = 0; j < ncol; ++j) {
        aug[static_cast<std::size_t>(r + j * n)] -= f * aug[static_cast<std::size_t>(col + j * n)];
      }
    }
  }
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      (*a)[static_cast<std::size_t>(i + j * n)] = aug[static_cast<std::size_t>(i + (n + j) * n)];
    }
  }
  return true;
}

// out = A * B, all n x n column-major
void matmul_nn(int n, const std::vector<C>& A, const std::vector<C>& B, std::vector<C>* out) {
  out->assign(static_cast<std::size_t>(n * n), C(0.0, 0.0));
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      C acc(0.0, 0.0);
      for (int k = 0; k < n; ++k) {
        acc += A[static_cast<std::size_t>(i + k * n)] * B[static_cast<std::size_t>(k + j * n)];
      }
      (*out)[static_cast<std::size_t>(i + j * n)] = acc;
    }
  }
}

std::vector<C> build_O(const ftlm::FockBasis& fb, const ftlm::symmetry::MomentumSectorMap& map,
                       const ftlm::symmetry::OrbitKBlockBasis& basis) {
  const int d = fb.dim();
  const int dk = basis.dim();
  std::vector<C> O(static_cast<std::size_t>(d * dk), C(0.0, 0.0));
  for (int j = 0; j < dk; ++j) {
    const auto& e = basis.entries()[static_cast<std::size_t>(j)];
    ftlm::symmetry::detail::fill_phi_orbit_bloch_from_seed(map, basis.K, basis.lx, basis.ly, fb, e.rep,
                                                          O.data() + static_cast<std::size_t>(j * d));
  }
  return O;
}

}  // namespace

int main() {
  ftlm::HubbardParams p{};
  p.Lx = 3;
  p.Ly = 2;
  p.t = 1.0;
  p.tp = -0.35;
  p.U = 5.75;
  p.V = 0.9;
  p.phi_x = 0.0;
  p.phi_y = 0.0;

  const int n_up = 2;
  const int n_dn = 3;
  ftlm::symmetry::MomentumSector K{1, 0, p.Lx, p.Ly};

  ftlm::symmetry::HubbardMomentumAction hub(p);
  ftlm::FockBasis fb(p.Lx * p.Ly, n_up, n_dn);
  std::vector<ftlm::symmetry::RawState> universe;
  universe.reserve(static_cast<std::size_t>(fb.dim()));
  for (int i = 0; i < fb.dim(); ++i) {
    universe.push_back(ftlm::symmetry::RawState{static_cast<std::uint16_t>(fb.up_mask(i)),
                                                static_cast<std::uint16_t>(fb.down_mask(i))});
  }
  const auto map = ftlm::symmetry::build_momentum_sector_map_rect(std::move(universe), p.Lx, p.Ly);

  ftlm::symmetry::OrbitKBlockBasis orbit_basis;
  orbit_basis.build(fb, K);
  const int dk = orbit_basis.dim();
  if (dk <= 0) {
    std::cerr << "empty orbit basis\n";
    return 2;
  }

  ftlm::symmetry::MomentumBlockScratch scratch;
  ftlm::symmetry::HubbardMomentumBlock prod_block(hub, map, K, n_up, n_dn, &scratch);
  const int dk_prod = static_cast<int>(prod_block.dim());
  if (dk_prod != dk) {
    std::cerr << "dk mismatch orbit=" << dk << " prod=" << dk_prod << "\n";
    return 2;
  }

  const int d = fb.dim();
  const int n = dk;

  std::vector<C> O = build_O(fb, map, orbit_basis);

  // T[i,j] = (Phi^dagger O(:,j))_i = project_full_to_block(O column j)
  std::vector<C> T(static_cast<std::size_t>(n * n), C(0.0, 0.0));
  std::vector<C> col_d(static_cast<std::size_t>(d));
  for (int j = 0; j < n; ++j) {
    for (int p = 0; p < d; ++p) {
      col_d[static_cast<std::size_t>(p)] = O[static_cast<std::size_t>(p + j * d)];
    }
    std::vector<C> tcol(static_cast<std::size_t>(n));
    prod_block.project_full_to_block(col_d.data(), tcol.data());
    for (int i = 0; i < n; ++i) {
      T[static_cast<std::size_t>(i + j * n)] = tcol[static_cast<std::size_t>(i)];
    }
  }

  // H_prod columns from block.apply
  std::vector<C> H_prod(static_cast<std::size_t>(n * n), C(0.0, 0.0));
  std::vector<C> ej(static_cast<std::size_t>(n), C(0.0, 0.0));
  std::vector<C> yj(static_cast<std::size_t>(n));
  for (int j = 0; j < n; ++j) {
    std::fill(ej.begin(), ej.end(), C(0.0, 0.0));
    ej[static_cast<std::size_t>(j)] = C(1.0, 0.0);
    prod_block.apply(ej.data(), yj.data());
    for (int i = 0; i < n; ++i) {
      H_prod[static_cast<std::size_t>(i + j * n)] = yj[static_cast<std::size_t>(i)];
    }
  }

  std::vector<C> T_inv = T;
  if (!invert_colmajor_gauss_jordan(n, &T_inv)) {
    std::cerr << "T is singular or nearly singular\n";
    return 2;
  }

  std::vector<C> tmp;
  matmul_nn(n, H_prod, T, &tmp);
  std::vector<C> H_exact;
  matmul_nn(n, T_inv, tmp, &H_exact);

  std::vector<C> H_form(static_cast<std::size_t>(n * n), C(0.0, 0.0));
  for (int j = 0; j < n; ++j) {
    std::fill(ej.begin(), ej.end(), C(0.0, 0.0));
    ej[static_cast<std::size_t>(j)] = C(1.0, 0.0);
    ftlm::symmetry::apply_orbit_k_block(p, fb, orbit_basis, hub.hoppings, hub.nn_pairs, ej.data(), yj.data());
    for (int i = 0; i < n; ++i) {
      H_form[static_cast<std::size_t>(i + j * n)] = yj[static_cast<std::size_t>(i)];
    }
  }

  double max_diff = 0.0;
  double frob_sq = 0.0;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const C d = H_exact[static_cast<std::size_t>(i + j * n)] - H_form[static_cast<std::size_t>(i + j * n)];
      const double ad = std::abs(d);
      max_diff = std::max(max_diff, ad);
      frob_sq += static_cast<double>(std::norm(d));
    }
  }

  std::cout << "=== diag_orbit_h_matrix_compare ===\n";
  std::cout << "3x2 (nu,nd)=(2,3) K=(1,0) dim_full=" << d << " dk=" << n << "\n";
  std::cout << "max_ij |H_exact(i,j) - H_formula(i,j)| = " << max_diff << "\n";
  std::cout << "Frobenius ||H_exact - H_formula||_F = " << std::sqrt(frob_sq) << "\n";

  struct Entry {
    double err{};
    int i{};
    int j{};
  };
  std::vector<Entry> entries;
  entries.reserve(static_cast<std::size_t>(n * n));
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const C d = H_exact[static_cast<std::size_t>(i + j * n)] - H_form[static_cast<std::size_t>(i + j * n)];
      entries.push_back({std::abs(d), i, j});
    }
  }
  std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) { return a.err > b.err; });

  const int nshow = std::min(12, static_cast<int>(entries.size()));
  std::cout << "\nTop " << nshow << " |delta| entries (i,j are orbit-basis indices):\n";
  for (int k = 0; k < nshow; ++k) {
    const int i = entries[static_cast<std::size_t>(k)].i;
    const int j = entries[static_cast<std::size_t>(k)].j;
    const C he = H_exact[static_cast<std::size_t>(i + j * n)];
    const C hf = H_form[static_cast<std::size_t>(i + j * n)];
    std::cout << "  (i,j)=(" << i << "," << j << ")  |diff|=" << entries[static_cast<std::size_t>(k)].err << "  H_exact=" << he.real()
              << (he.imag() >= 0 ? "+" : "") << he.imag() << "i"
              << "  H_form=" << hf.real() << (hf.imag() >= 0 ? "+" : "") << hf.imag() << "i\n";
  }

  // Hermiticity residuals
  double max_herm_Hex = 0.0;
  double max_herm_Hf = 0.0;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < j; ++i) {
      const C dex = H_exact[static_cast<std::size_t>(i + j * n)] - std::conj(H_exact[static_cast<std::size_t>(j + i * n)]);
      const C df = H_form[static_cast<std::size_t>(i + j * n)] - std::conj(H_form[static_cast<std::size_t>(j + i * n)]);
      max_herm_Hex = std::max(max_herm_Hex, std::abs(dex));
      max_herm_Hf = std::max(max_herm_Hf, std::abs(df));
    }
  }
  std::cout << "\nHermiticity: max|H_exact - H_exact^dagger| offdiag = " << max_herm_Hex << "\n";
  std::cout << "Hermiticity: max|H_form - H_form^dagger| offdiag = " << max_herm_Hf << "\n";

  std::cout << "\nInterpretation:\n"
            << "  H_exact = T^{-1} H_prod T with T = Phi^dagger O (project_full_to_block on each O column).\n"
            << "  If apply_orbit_k_block matched production, H_form would equal H_exact.\n";

  return 0;
}
