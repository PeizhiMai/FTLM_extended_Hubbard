// 2×2 cluster: grand-canonical density n = ⟨N⟩/N_sites vs μ at fixed β (exact diagonalization, all sectors).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "ftlm/fock_basis.hpp"
#include "ftlm/hubbard_hamiltonian.hpp"
#include "ftlm/hubbard_params.hpp"
#include "ftlm/lattice.hpp"

namespace {

using clock = std::chrono::high_resolution_clock;

double parse_double_arg(int argc, char** argv, const char* key, double default_value) {
  const std::string prefix = std::string("--") + key + "=";
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    if (std::strncmp(a, prefix.c_str(), prefix.size()) == 0) {
      return std::strtod(a + prefix.size(), nullptr);
    }
  }
  return default_value;
}

int parse_int_arg(int argc, char** argv, const char* key, int default_value) {
  const std::string prefix = std::string("--") + key + "=";
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    if (std::strncmp(a, prefix.c_str(), prefix.size()) == 0) {
      return std::atoi(a + prefix.size());
    }
  }
  return default_value;
}

const char* parse_string_arg(int argc, char** argv, const char* key) {
  const std::string prefix = std::string("--") + key + "=";
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    if (std::strncmp(a, prefix.c_str(), prefix.size()) == 0) {
      return a + prefix.size();
    }
  }
  return nullptr;
}

void build_dense_real(const ftlm::HubbardParams& p, const ftlm::FockBasis& basis,
                      const std::vector<ftlm::SpinfulHopping>& hops,
                      const std::vector<ftlm::NearestPair>& pairs,
                      std::vector<std::vector<double>>* Hreal) {
  const int dim = basis.dim();
  Hreal->assign(static_cast<size_t>(dim), std::vector<double>(static_cast<size_t>(dim), 0.0));
  std::vector<std::complex<double>> col(static_cast<size_t>(dim)), out(static_cast<size_t>(dim));
  for (int j = 0; j < dim; ++j) {
    std::fill(col.begin(), col.end(), std::complex<double>(0.0, 0.0));
    col[static_cast<size_t>(j)] = {1.0, 0.0};
    ftlm::apply_extended_hubbard(p, basis, hops, pairs, col.data(), out.data());
    for (int i = 0; i < dim; ++i) {
      (*Hreal)[static_cast<size_t>(i)][static_cast<size_t>(j)] = out[static_cast<size_t>(i)].real();
    }
  }
}

/// All eigenvalues of a real symmetric matrix (Jacobi sweeps).
std::vector<double> sorted_eigenvalues_jacobi(std::vector<std::vector<double>> A, int max_sweeps,
                                              double tol_offdiag) {
  const int n = static_cast<int>(A.size());
  if (n == 0) {
    return {};
  }
  if (n == 1) {
    return {A[0][0]};
  }
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
  }
  std::vector<double> d;
  d.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    d.push_back(A[static_cast<size_t>(i)][static_cast<size_t>(i)]);
  }
  std::sort(d.begin(), d.end());
  return d;
}

struct Level {
  double E = 0.0;
  int N = 0;  // total electron count
};

/// ⟨N⟩ and n = ⟨N⟩/n_sites at chemical potential μ (grand canonical), stable at large β.
void grand_canonical_expectation(double beta, double mu, const std::vector<Level>& levels, double* avg_N,
                                 double* avg_n, int n_sites) {
  double max_a = -std::numeric_limits<double>::infinity();
  for (const Level& lv : levels) {
    const double a = -beta * (lv.E - mu * static_cast<double>(lv.N));
    max_a = std::max(max_a, a);
  }
  double sumw = 0.0;
  double sum_Nw = 0.0;
  for (const Level& lv : levels) {
    const double a = -beta * (lv.E - mu * static_cast<double>(lv.N));
    const double w = std::exp(a - max_a);
    sumw += w;
    sum_Nw += static_cast<double>(lv.N) * w;
  }
  *avg_N = sum_Nw / sumw;
  *avg_n = (*avg_N) / static_cast<double>(n_sites);
}

}  // namespace

int main(int argc, char** argv) {
  const double beta = parse_double_arg(argc, argv, "beta", 20.0);
  const double mu_min = parse_double_arg(argc, argv, "mu-min", -5.0);
  const double mu_max = parse_double_arg(argc, argv, "mu-max", 30.0);
  const int n_mu = std::max(2, parse_int_arg(argc, argv, "n-mu", 221));
  const char* out_path = parse_string_arg(argc, argv, "out");

  ftlm::HubbardParams p;
  p.Lx = 2;
  p.Ly = 2;
  p.t = parse_double_arg(argc, argv, "t", 1.0);
  p.tp = parse_double_arg(argc, argv, "tp", -0.35);
  p.U = parse_double_arg(argc, argv, "U", 5.75);
  p.V = parse_double_arg(argc, argv, "V", 0.9);
  p.phi_x = 0.0;
  p.phi_y = 0.0;

  const int n_sites = p.Lx * p.Ly;
  ftlm::RectLattice lat{p.Lx, p.Ly};
  std::vector<ftlm::SpinfulHopping> hops;
  std::vector<ftlm::NearestPair> pairs;
  ftlm::build_hubbard_geometry(p, lat, &hops, &pairs);

  std::cout << "=== 2×2 grand canonical: n(μ) at β=" << beta << " ===\n";
  std::cout << "t=" << p.t << " t'=" << p.tp << " U=" << p.U << " V=" << p.V << "  N_sites=" << n_sites << "\n";

  std::vector<Level> levels;
  levels.reserve(256);

  const auto t0 = clock::now();
  for (int nu = 0; nu <= n_sites; ++nu) {
    for (int nd = 0; nd <= n_sites; ++nd) {
      ftlm::FockBasis basis(n_sites, nu, nd);
      const int dim = basis.dim();
      if (dim <= 0) {
        continue;
      }
      std::vector<std::vector<double>> H;
      build_dense_real(p, basis, hops, pairs, &H);
      std::vector<double> evals = sorted_eigenvalues_jacobi(std::move(H), 12000, 1e-14);
      const int N = nu + nd;
      for (double e : evals) {
        levels.push_back(Level{e, N});
      }
    }
  }
  const double t_ed = std::chrono::duration<double>(clock::now() - t0).count();

  std::cout << "exact diagonalization: " << levels.size() << " eigenlevels in " << (t_ed * 1e3) << " ms\n";

  std::vector<std::pair<double, double>> curve;
  curve.reserve(static_cast<size_t>(n_mu));
  for (int i = 0; i < n_mu; ++i) {
    const double tmu = static_cast<double>(i) / static_cast<double>(n_mu - 1);
    const double mu = mu_min + tmu * (mu_max - mu_min);
    double avg_N = 0.0;
    double avg_n = 0.0;
    grand_canonical_expectation(beta, mu, levels, &avg_N, &avg_n, n_sites);
    curve.push_back({mu, avg_n});
  }

  std::ostream* out = &std::cout;
  std::ofstream file;
  if (out_path) {
    file.open(out_path);
    if (!file) {
      std::cerr << "bench_2x2_nmu: cannot open " << out_path << "\n";
      return 2;
    }
    out = &file;
  }

  *out << "# 2x2 PBC grand canonical  mu  n=<N>/N_sites  beta=" << beta << "  t=" << p.t << " tp=" << p.tp
       << " U=" << p.U << " V=" << p.V << "\n";
  for (const auto& pr : curve) {
    *out << pr.first << "\t" << pr.second << "\n";
  }

  if (out_path) {
    std::cout << "wrote " << out_path << "\n";
  }

  return 0;
}
