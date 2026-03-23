#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <vector>

#include "ftlm/fock_basis.hpp"
#include "ftlm/hubbard_hamiltonian.hpp"
#include "ftlm/hubbard_params.hpp"
#include "ftlm/lanczos.hpp"
#include "ftlm/lattice.hpp"

namespace {

using ftlm::FockBasis;
using ftlm::HubbardParams;
using ftlm::LanczosExtrema;
using ftlm::NearestPair;
using ftlm::RectLattice;
using ftlm::SpinfulHopping;
using ftlm::apply_extended_hubbard;
using ftlm::build_hubbard_geometry;
using ftlm::lanczos_extrema;

/// Classic Jacobi diagonalization (real symmetric) — test-only duplicate of the core used in Lanczos.
std::pair<double, double> jacobi_min_max(std::vector<std::vector<double>> A) {
  const int n = static_cast<int>(A.size());
  if (n == 0) {
    return {0.0, 0.0};
  }
  if (n == 1) {
    return {A[0][0], A[0][0]};
  }
  for (int sweep = 0; sweep < 120; ++sweep) {
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
    if (max_abs < 1e-15) {
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
  }
  double lo = std::numeric_limits<double>::infinity();
  double hi = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < n; ++i) {
    const double d = A[static_cast<size_t>(i)][static_cast<size_t>(i)];
    lo = std::min(lo, d);
    hi = std::max(hi, d);
  }
  return {lo, hi};
}

void build_dense(const HubbardParams& p, const RectLattice& lat, const FockBasis& basis,
                 const std::vector<SpinfulHopping>& hops, const std::vector<NearestPair>& pairs,
                 std::vector<std::vector<std::complex<double>>>* H) {
  const int dim = basis.dim();
  H->assign(static_cast<size_t>(dim), std::vector<std::complex<double>>(static_cast<size_t>(dim), 0.0));
  std::vector<std::complex<double>> col(static_cast<size_t>(dim)), out(static_cast<size_t>(dim));
  for (int j = 0; j < dim; ++j) {
    std::fill(col.begin(), col.end(), std::complex<double>(0.0, 0.0));
    col[static_cast<size_t>(j)] = {1.0, 0.0};
    apply_extended_hubbard(p, basis, hops, pairs, col.data(), out.data());
    for (int i = 0; i < dim; ++i) {
      (*H)[static_cast<size_t>(i)][static_cast<size_t>(j)] = out[static_cast<size_t>(i)];
    }
  }
}

double max_hermitian_residual(const std::vector<std::vector<std::complex<double>>>& H) {
  const int n = static_cast<int>(H.size());
  double mx = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      const std::complex<double> d =
          H[static_cast<size_t>(i)][static_cast<size_t>(j)] -
          std::conj(H[static_cast<size_t>(j)][static_cast<size_t>(i)]);
      mx = std::max(mx, std::abs(d));
    }
  }
  return mx;
}

}  // namespace

int main() {
  // --- 1) Fock basis dimension: 2 sites, (1,1) -> 2 * 2 = 4
  {
    FockBasis b(2, 1, 1);
    assert(b.dim() == 4);
    assert(b.index_of((1ULL << 0), (1ULL << 1)) >= 0);
  }

  // --- 2–3) Real 2x1 Hubbard chain, PBC in x, kinetic only: Lanczos vs dense Jacobi
  {
    HubbardParams p;
    p.Lx = 2;
    p.Ly = 1;
    p.t = 1.0;
    p.tp = 0.0;
    p.U = 0.0;
    p.V = 0.0;
    p.phi_x = 0.0;
    p.phi_y = 0.0;
    RectLattice lat{p.Lx, p.Ly};
    std::vector<SpinfulHopping> hops;
    std::vector<NearestPair> pairs;
    build_hubbard_geometry(p, lat, &hops, &pairs);
    FockBasis basis(lat.n_sites(), 1, 1);

    std::vector<std::vector<std::complex<double>>> Hfull;
    build_dense(p, lat, basis, hops, pairs, &Hfull);
    assert(max_hermitian_residual(Hfull) < 1e-12);

    std::vector<std::vector<double>> Hreal(static_cast<size_t>(basis.dim()),
                                           std::vector<double>(static_cast<size_t>(basis.dim()), 0.0));
    for (int i = 0; i < basis.dim(); ++i) {
      for (int j = 0; j < basis.dim(); ++j) {
        assert(std::abs(Hfull[static_cast<size_t>(i)][static_cast<size_t>(j)].imag()) < 1e-12);
        Hreal[static_cast<size_t>(i)][static_cast<size_t>(j)] =
            Hfull[static_cast<size_t>(i)][static_cast<size_t>(j)].real();
      }
    }
    const auto jac = jacobi_min_max(Hreal);
    const double e0_dense = jac.first;

    auto apply = [&](const std::complex<double>* v, std::complex<double>* w) {
      apply_extended_hubbard(p, basis, hops, pairs, v, w);
    };
    const LanczosExtrema lz = lanczos_extrema(basis.dim(), apply, 48, 42);
    if (std::abs(lz.min_eval - e0_dense) > 5e-5) {
      std::cerr << "Lanczos min " << lz.min_eval << " dense " << e0_dense << "\n";
      return 1;
    }
  }

  // --- Twisted flux: Hermiticity smoke (matrix should be Hermitian)
  {
    HubbardParams p;
    p.Lx = 2;
    p.Ly = 2;
    p.t = 1.0;
    p.phi_x = 0.31;
    p.phi_y = -0.27;
    RectLattice lat{p.Lx, p.Ly};
    std::vector<SpinfulHopping> hops;
    std::vector<NearestPair> pairs;
    build_hubbard_geometry(p, lat, &hops, &pairs);
    FockBasis basis(lat.n_sites(), 2, 2);
    std::vector<std::vector<std::complex<double>>> Hfull;
    build_dense(p, lat, basis, hops, pairs, &Hfull);
    const double res = max_hermitian_residual(Hfull);
    if (res > 1e-10) {
      std::cerr << "Hermitian residual " << res << "\n";
      return 2;
    }
  }

  std::cout << "test_hubbard_pipeline ok\n";
  return 0;
}
