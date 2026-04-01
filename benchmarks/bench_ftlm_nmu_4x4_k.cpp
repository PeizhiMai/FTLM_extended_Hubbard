// Grand-canonical n(mu) at fixed beta via FTLM using translation-symmetry momentum blocks.
//
// Solver selection (`ftlm_bench_ftlm_nmu_rect_k`):
//   --solver=k-gram     — production Gram / HubbardMomentumBlock (default when no --solver)
//   --solver=k-orbit    — orbit k-block (same as --kblock-prototype=orbit-direct)
//   --solver=nonmom     — particle-sector FTLM without momentum (same as ftlm_bench_ftlm_nmu_rect)
//   --solver=ed         — dense exact diagonalization per momentum block (large --ed-cutoff; use --ed-cutoff to cap)
//
// Legacy: `--kblock-prototype=...` still works when `--solver` is omitted.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <sys/resource.h>
#endif

#include "ftlm/fock_basis.hpp"
#include "ftlm/ftlm_thermo.hpp"
#include "ftlm/hubbard_hamiltonian.hpp"
#include "ftlm/hubbard_params.hpp"
#include "ftlm/lattice.hpp"
#include "ftlm/symmetry/hubbard_lanczos.hpp"
#include "ftlm/symmetry/hubbard_momentum_action.hpp"
#include "ftlm/symmetry/k_basis.hpp"
#include "ftlm/symmetry/k_block_im_pk_reduced.hpp"
#include "ftlm/symmetry/k_block_pk_tiny_dense.hpp"
#include "ftlm/symmetry/momentum_sector.hpp"
#include "ftlm/symmetry/orbit_bloch_phi.hpp"
#include "ftlm/symmetry/orbit_k_block_basis.hpp"
#include "ftlm/symmetry/raw_state.hpp"

namespace {

using clock = std::chrono::high_resolution_clock;
using C = std::complex<double>;

// Small dense helpers for orbit–Gram bridge (T = Phi^dagger O): FTLM/Lanczos uses Gram ONB coordinates.
void swap_rows_cm(int nrows, int ncols, C* a, int r1, int r2) {
  if (r1 == r2) {
    return;
  }
  for (int j = 0; j < ncols; ++j) {
    std::swap(a[r1 + j * nrows], a[r2 + j * nrows]);
  }
}

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

bool has_arg(int argc, char** argv, const char* key) {
  const std::string prefix = std::string("--") + key + "=";
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    if (std::strncmp(a, prefix.c_str(), prefix.size()) == 0) {
      return true;
    }
  }
  return false;
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

int sector_index(int nu, int nd, int n_sites) { return nu * (n_sites + 1) + nd; }

double peak_rss_bytes_self() {
#if defined(_WIN32)
  return 0.0;
#else
  struct rusage ru {};
  if (getrusage(RUSAGE_SELF, &ru) != 0) {
    return 0.0;
  }
#if defined(__APPLE__)
  return static_cast<double>(ru.ru_maxrss);
#else
  return static_cast<double>(ru.ru_maxrss) * 1024.0;
#endif
#endif
}

double logsumexp2(double a, double b) {
  if (!std::isfinite(a)) {
    return b;
  }
  if (!std::isfinite(b)) {
    return a;
  }
  const double m = (a > b) ? a : b;
  return m + std::log(std::exp(a - m) + std::exp(b - m));
}

void jacobi_symmetric_all(std::vector<std::vector<double>>& A, int max_sweeps, double tol_offdiag) {
  const int n = static_cast<int>(A.size());
  if (n <= 1) {
    return;
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
}

struct ExactEdMemStats {
  bool ran = false;
  int max_dim = 0;
  std::size_t max_H_dense_B = 0;
  std::size_t max_R_dense_B = 0;
  std::size_t max_aux_vecs_B = 0;
  std::size_t max_total_B = 0;
  bool reused_buffers = false;
};

struct ExactEdScratch {
  std::vector<std::vector<std::complex<double>>> H{};
  std::vector<std::vector<double>> R{};
  std::vector<std::complex<double>> e{};
  std::vector<std::complex<double>> y{};
};

double exact_log_partition_small_complex(
    int dim, const std::function<void(const std::complex<double>*, std::complex<double>*)>& apply_h, double beta,
    ExactEdMemStats* mem_stats = nullptr, ExactEdScratch* scratch = nullptr) {
  if (dim <= 0) {
    return -std::numeric_limits<double>::infinity();
  }
  if (dim == 1) {
    std::complex<double> x[1] = {std::complex<double>(1.0, 0.0)};
    std::complex<double> y[1] = {std::complex<double>(0.0, 0.0)};
    apply_h(x, y);
    return -beta * y[0].real();
  }
  if (mem_stats != nullptr) {
    mem_stats->ran = true;
    mem_stats->max_dim = std::max(mem_stats->max_dim, dim);
  }

  std::vector<std::vector<std::complex<double>>> local_H;
  std::vector<std::vector<double>> local_R;
  std::vector<std::complex<double>> local_e;
  std::vector<std::complex<double>> local_y;
  auto* H_ptr = scratch ? &scratch->H : &local_H;
  auto* R_ptr = scratch ? &scratch->R : &local_R;
  auto* e_ptr = scratch ? &scratch->e : &local_e;
  auto* y_ptr = scratch ? &scratch->y : &local_y;
  auto& H = *H_ptr;
  auto& R = *R_ptr;
  auto& e = *e_ptr;
  auto& y = *y_ptr;

  H.resize(static_cast<size_t>(dim));
  for (auto& row : H) {
    row.resize(static_cast<size_t>(dim));
    std::fill(row.begin(), row.end(), std::complex<double>(0.0, 0.0));
  }
  e.resize(static_cast<size_t>(dim));
  y.resize(static_cast<size_t>(dim));
  for (int j = 0; j < dim; ++j) {
    std::fill(e.begin(), e.end(), std::complex<double>(0.0, 0.0));
    e[static_cast<size_t>(j)] = {1.0, 0.0};
    apply_h(e.data(), y.data());
    for (int i = 0; i < dim; ++i) {
      H[static_cast<size_t>(i)][static_cast<size_t>(j)] = y[static_cast<size_t>(i)];
    }
  }

  const int m = 2 * dim;
  R.resize(static_cast<size_t>(m));
  for (auto& row : R) {
    row.resize(static_cast<size_t>(m));
    std::fill(row.begin(), row.end(), 0.0);
  }
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      const std::complex<double> hij = H[static_cast<size_t>(i)][static_cast<size_t>(j)];
      const double re = hij.real();
      const double im = hij.imag();
      R[static_cast<size_t>(i)][static_cast<size_t>(j)] = re;
      R[static_cast<size_t>(i)][static_cast<size_t>(j + dim)] = -im;
      R[static_cast<size_t>(i + dim)][static_cast<size_t>(j)] = im;
      R[static_cast<size_t>(i + dim)][static_cast<size_t>(j + dim)] = re;
    }
  }
  if (mem_stats != nullptr) {
    const std::size_t h_dense_b =
        static_cast<std::size_t>(dim) * static_cast<std::size_t>(dim) * sizeof(std::complex<double>);
    const std::size_t r_dense_b = static_cast<std::size_t>(m) * static_cast<std::size_t>(m) * sizeof(double);
    const std::size_t aux_vecs_b = static_cast<std::size_t>(2 * dim) * sizeof(std::complex<double>);
    mem_stats->max_H_dense_B = std::max(mem_stats->max_H_dense_B, h_dense_b);
    mem_stats->max_R_dense_B = std::max(mem_stats->max_R_dense_B, r_dense_b);
    mem_stats->max_aux_vecs_B = std::max(mem_stats->max_aux_vecs_B, aux_vecs_b);
    mem_stats->max_total_B = std::max(mem_stats->max_total_B, h_dense_b + r_dense_b + aux_vecs_b);
    mem_stats->reused_buffers = mem_stats->reused_buffers || (scratch != nullptr);
  }

  jacobi_symmetric_all(R, std::max(5000, 150 * m), 1e-13);
  double max_t = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < m; ++i) {
    const double t = -beta * R[static_cast<size_t>(i)][static_cast<size_t>(i)];
    max_t = std::max(max_t, t);
  }
  double sum = 0.0;
  for (int i = 0; i < m; ++i) {
    const double t = -beta * R[static_cast<size_t>(i)][static_cast<size_t>(i)];
    sum += std::exp(t - max_t);
  }
  return std::log(0.5) + max_t + std::log(sum);
}

enum class NmuSolver { KGram, KOrbit, NonMom, Ed };

bool parse_solver_name(const char* s, NmuSolver* out) {
  if (s == nullptr) {
    return false;
  }
  if (std::strcmp(s, "k-gram") == 0 || std::strcmp(s, "k_block_gram") == 0 || std::strcmp(s, "gram") == 0 ||
      std::strcmp(s, "production") == 0) {
    *out = NmuSolver::KGram;
    return true;
  }
  if (std::strcmp(s, "k-orbit") == 0 || std::strcmp(s, "orbit") == 0 || std::strcmp(s, "orbit-direct") == 0) {
    *out = NmuSolver::KOrbit;
    return true;
  }
  if (std::strcmp(s, "nonmom") == 0 || std::strcmp(s, "nomom") == 0 || std::strcmp(s, "non-momentum") == 0) {
    *out = NmuSolver::NonMom;
    return true;
  }
  if (std::strcmp(s, "ed") == 0 || std::strcmp(s, "exact") == 0) {
    *out = NmuSolver::Ed;
    return true;
  }
  return false;
}

const char* solver_label(NmuSolver s) {
  switch (s) {
    case NmuSolver::KGram:
      return "k-gram";
    case NmuSolver::KOrbit:
      return "k-orbit";
    case NmuSolver::NonMom:
      return "nonmom";
    case NmuSolver::Ed:
      return "ed";
  }
  return "?";
}

/// Same physics as `ftlm_bench_ftlm_nmu_rect`: particle-sector FTLM (no momentum blocks).
int run_nonmom_solver(int argc, char** argv) {
  const int Lx = parse_int_arg(argc, argv, "Lx", 4);
  const int Ly = parse_int_arg(argc, argv, "Ly", 2);
  if (Lx <= 0 || Ly <= 0 || Lx * Ly > 16) {
    std::cerr << "bench_ftlm_nmu_rect_k (--solver=nonmom): requires 1 <= Lx*Ly <= 16\n";
    return 2;
  }

  const double beta = parse_double_arg(argc, argv, "beta", 20.0);
  const double mu_min = parse_double_arg(argc, argv, "mu-min", -5.0);
  const double mu_max = parse_double_arg(argc, argv, "mu-max", 25.0);
  const int n_mu = std::max(2, parse_int_arg(argc, argv, "n-mu", 221));
  const int n_rand = parse_int_arg(argc, argv, "ftlm-random", 12);
  const int lz_steps = parse_int_arg(argc, argv, "lanczos-steps", 72);
  const unsigned fseed = static_cast<unsigned>(parse_int_arg(argc, argv, "seed", 7));
  const char* out_path = parse_string_arg(argc, argv, "out");
  const int no_monitor = parse_int_arg(argc, argv, "no-monitor", 0);

  const auto t_wall0 = clock::now();

  ftlm::HubbardParams p;
  p.Lx = Lx;
  p.Ly = Ly;
  p.t = parse_double_arg(argc, argv, "t", 1.0);
  p.tp = parse_double_arg(argc, argv, "tp", -0.35);
  p.U = parse_double_arg(argc, argv, "U", 5.75);
  p.V = parse_double_arg(argc, argv, "V", 0.9);
  p.phi_x = 0.0;
  p.phi_y = 0.0;

  const int n_sites = Lx * Ly;
  ftlm::RectLattice lat{p.Lx, p.Ly};
  std::vector<ftlm::SpinfulHopping> hops;
  std::vector<ftlm::NearestPair> pairs;
  ftlm::build_hubbard_geometry(p, lat, &hops, &pairs);

  ftlm::FtlmParams fpar;
  fpar.n_random = n_rand;
  fpar.lanczos_steps = lz_steps;
  fpar.seed = fseed;

  const int nsec = (n_sites + 1) * (n_sites + 1);
  std::vector<double> logZ(static_cast<size_t>(nsec), -std::numeric_limits<double>::infinity());
  std::vector<int> Nelec(static_cast<size_t>(nsec), 0);
  std::vector<int> dims(static_cast<size_t>(nsec), 0);

  std::cout << "=== FTLM grand-canonical n(mu)  " << Lx << "x" << Ly << "  beta=" << beta << "  solver=nonmom ===\n";
  std::cout << "t=" << p.t << " tp=" << p.tp << " U=" << p.U << " V=" << p.V << "\n";
  std::cout << "ftlm_random=" << n_rand << " lanczos_steps=" << lz_steps << "  sectors=" << nsec << "\n";
  if (!no_monitor) {
    std::cout << "[monitor] wall time + peak RSS reported at exit (disable with --no-monitor=1)\n";
  }

  const auto t0 = clock::now();
  std::vector<std::complex<double>> xc;
  std::vector<std::complex<double>> yc;

  for (int nu = 0; nu <= n_sites; ++nu) {
    for (int nd = 0; nd <= n_sites; ++nd) {
      const int idx = sector_index(nu, nd, n_sites);
      Nelec[static_cast<size_t>(idx)] = nu + nd;
      ftlm::FockBasis basis(n_sites, nu, nd);
      const int dim = basis.dim();
      dims[static_cast<size_t>(idx)] = dim;
      if (dim <= 0) {
        continue;
      }
      xc.assign(static_cast<size_t>(dim), {0.0, 0.0});
      yc.assign(static_cast<size_t>(dim), {0.0, 0.0});

      auto apply_real = [&](const double* x, double* y) {
        for (int i = 0; i < dim; ++i) {
          xc[static_cast<size_t>(i)] = {x[static_cast<size_t>(i)], 0.0};
        }
        ftlm::apply_extended_hubbard(p, basis, hops, pairs, xc.data(), yc.data());
        for (int i = 0; i < dim; ++i) {
          y[static_cast<size_t>(i)] = yc[static_cast<size_t>(i)].real();
        }
      };

      const double lz = ftlm::ftlm_log_partition_real(dim, apply_real, beta, fpar);
      logZ[static_cast<size_t>(idx)] = lz;
      std::cerr << "sector (" << nu << "," << nd << ") dim=" << dim << " logZ=" << lz << "\n";
    }
  }
  const double t_sectors = std::chrono::duration<double>(clock::now() - t0).count();
  std::cout << "[timing] all sectors logZ: " << t_sectors << " s\n";

  std::ostream* out = &std::cout;
  std::ofstream file;
  if (out_path) {
    file.open(out_path);
    if (!file) {
      std::cerr << "bench_ftlm_nmu_rect_k (--solver=nonmom): cannot open " << out_path << "\n";
      return 2;
    }
    out = &file;
  }

  *out << "# FTLM grand canonical  Lx=" << Lx << " Ly=" << Ly << "  beta=" << beta << "  t=" << p.t
       << " tp=" << p.tp << " U=" << p.U << " V=" << p.V << "  solver=nonmom\n";
  *out << "# ftlm_random=" << n_rand << " lanczos_steps=" << lz_steps << "  sector_time_s=" << t_sectors << "\n";

  for (int k = 0; k < n_mu; ++k) {
    const double tmu = static_cast<double>(k) / static_cast<double>(n_mu - 1);
    const double mu = mu_min + tmu * (mu_max - mu_min);

    double mx = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < nsec; ++i) {
      const double lz = logZ[static_cast<size_t>(i)];
      if (!std::isfinite(lz)) {
        continue;
      }
      const int N = Nelec[static_cast<size_t>(i)];
      const double ex = lz + beta * mu * static_cast<double>(N);
      mx = std::max(mx, ex);
    }
    if (!std::isfinite(mx)) {
      *out << mu << "\t0\n";
      continue;
    }
    double sum_w = 0.0;
    double sum_Nw = 0.0;
    for (int i = 0; i < nsec; ++i) {
      const double lz = logZ[static_cast<size_t>(i)];
      if (!std::isfinite(lz)) {
        continue;
      }
      const int N = Nelec[static_cast<size_t>(i)];
      const double ex = lz + beta * mu * static_cast<double>(N) - mx;
      const double w = std::exp(ex);
      sum_w += w;
      sum_Nw += static_cast<double>(N) * w;
    }
    const double n_avg = (sum_w > 0.0) ? (sum_Nw / sum_w) / static_cast<double>(n_sites) : 0.0;
    *out << mu << "\t" << n_avg << "\n";
  }

  if (out_path) {
    std::cout << "wrote " << out_path << "\n";
  }

  const double t_wall = std::chrono::duration<double>(clock::now() - t_wall0).count();
  const double peak_b = peak_rss_bytes_self();
  const double peak_mib = peak_b / (1024.0 * 1024.0);
  if (!no_monitor) {
    std::cout << "[monitor] wall_time_s=" << t_wall << "  wall_sector_logZ_s=" << t_sectors
              << "  peak_rss_mib=" << peak_mib << "  peak_rss_bytes=" << peak_b << "\n";
    if (out_path && file.is_open()) {
      *out << "# wall_time_s=" << t_wall << "  peak_rss_bytes=" << peak_b << "  peak_rss_mib=" << peak_mib
           << "\n";
      file.flush();
    }
  }

  std::cout << "METRIC kind=FTLM_nmu_rect solver=nonmom Lx=" << Lx << " Ly=" << Ly << " beta=" << beta
            << " wall_time_s=" << t_wall << " wall_sector_logZ_s=" << t_sectors << " peak_rss_bytes=" << peak_b
            << " peak_rss_mib=" << peak_mib << " n_mu=" << n_mu;
  if (no_monitor) {
    std::cout << " no_monitor=1";
  }
  std::cout << "\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const char* solver_arg = parse_string_arg(argc, argv, "solver");
  NmuSolver solver = NmuSolver::KGram;
  bool solver_set = false;
  if (solver_arg != nullptr) {
    if (!parse_solver_name(solver_arg, &solver)) {
      std::cerr << "bench_ftlm_nmu_rect_k: unknown --solver=" << solver_arg
                << " (supported: k-gram, k-orbit, nonmom, ed)\n";
      return 2;
    }
    solver_set = true;
  }
  if (solver_set && solver == NmuSolver::NonMom) {
    return run_nonmom_solver(argc, argv);
  }

  const int Lx = parse_int_arg(argc, argv, "Lx", 4);
  const int Ly = parse_int_arg(argc, argv, "Ly", 3);
  if (Lx <= 0 || Ly <= 0 || Lx * Ly > 16) {
    std::cerr << "bench_ftlm_nmu_rect_k: requires 1 <= Lx*Ly <= 16\n";
    return 2;
  }

  const double beta = parse_double_arg(argc, argv, "beta", 20.0);
  const double mu_min = parse_double_arg(argc, argv, "mu-min", -5.0);
  const double mu_max = parse_double_arg(argc, argv, "mu-max", 25.0);
  const int n_mu = std::max(2, parse_int_arg(argc, argv, "n-mu", 221));
  const int n_rand = parse_int_arg(argc, argv, "ftlm-random", 8);
  // Default momentum-Lanczos depth follows user heuristic:
  //   a_k = a_nonmom / (N/2) = 2 * a_nonmom / N, with a_nonmom = 72.
  const int n_sites = Lx * Ly;
  const int a_nonmom = 72;
  const int lz_default_k = std::max(20, (2 * a_nonmom) / std::max(1, n_sites));
  const int lz_steps = has_arg(argc, argv, "lanczos-steps")
                           ? parse_int_arg(argc, argv, "lanczos-steps", lz_default_k)
                           : lz_default_k;
  const unsigned fseed = static_cast<unsigned>(parse_int_arg(argc, argv, "seed", 7));
  const char* out_path = parse_string_arg(argc, argv, "out");
  const int no_monitor = parse_int_arg(argc, argv, "no-monitor", 0);
  const int no_log_k_dims = parse_int_arg(argc, argv, "no-log-k-dims", 0);
  const bool log_k_dims = (no_log_k_dims == 0);
  int ed_cutoff = parse_int_arg(argc, argv, "ed-cutoff", 64);
  if (solver_set && solver == NmuSolver::Ed && !has_arg(argc, argv, "ed-cutoff")) {
    ed_cutoff = std::numeric_limits<int>::max() / 4;
  }
  const int mem_report = parse_int_arg(argc, argv, "mem-report", 0);
  const int mem_report_detail = parse_int_arg(argc, argv, "mem-report-detail", 0);
  const int lanczos_ws_report = parse_int_arg(argc, argv, "lanczos-ws-report", 0);
  const int mem_instrument = parse_int_arg(argc, argv, "mem-instrument", 0);
  /// 0 = reuse Gram-build scratch across K (default). 1 = release dense build buffers after each K (A/B experiment).
  const int gram_build_no_reuse = parse_int_arg(argc, argv, "gram-build-no-reuse", 0);
  const int kblock_prototype_diagnostics = parse_int_arg(argc, argv, "kblock-prototype-diagnostics", 0);
  const int kblock_diagnostics_prod_dim = parse_int_arg(argc, argv, "kblock-diagnostics-prod-dim", 0);
  // Prototype: `--kblock-prototype=pkhpk-mf` uses matrix-free P_K H P_K on full sector (dk = dim_full).
  // `--kblock-prototype=im-pk-reduced`: `momentum_phi_gram_k_out_only` (same G+zheev+tol as production) for k_out; no packed V.
  // If k_out>0 and build_tiny_im_pk_reduced has r==k_out, use H_red=U†HU (no HubbardMomentumBlock). Else fall back.
  // rank(P) alone is unsafe vs k_out (see tests/test_k_gram_mismatch_scan.cpp).
  // `--kblock-prototype=orbit-matrix-free`: bridged T H_orb T^{-1} matvec for seed parity with Gram path when dims match.
  // `--kblock-prototype=orbit-direct`: Lanczos/FTLM in orbit coords via apply_orbit_k_block only (no Gram/Phi/T).
  // Default (no --solver): translation-orbit K-block first when sector-safe; Gram-based HubbardMomentumBlock as fallback.
  // `--solver=k-gram`: Gram + Phi + zheev only (no orbit-first).
  const char* kblock_prototype_arg = parse_string_arg(argc, argv, "kblock-prototype");
  bool kblock_proto_pkhpk_mf =
      kblock_prototype_arg != nullptr && std::strcmp(kblock_prototype_arg, "pkhpk-mf") == 0;
  bool kblock_proto_im_pk_reduced =
      kblock_prototype_arg != nullptr && std::strcmp(kblock_prototype_arg, "im-pk-reduced") == 0;
  bool kblock_proto_orbit_mf =
      kblock_prototype_arg != nullptr && std::strcmp(kblock_prototype_arg, "orbit-matrix-free") == 0;
  bool kblock_proto_orbit_direct =
      kblock_prototype_arg != nullptr && std::strcmp(kblock_prototype_arg, "orbit-direct") == 0;
  constexpr int kPkhpkMfMaxDim = 64;
  constexpr int kImPkReducedMaxDim = 64;
  constexpr int kOrbitMfMaxDim = 512;

  if (solver_set) {
    if (solver == NmuSolver::KGram) {
      if (kblock_prototype_arg != nullptr) {
        std::cerr << "bench_ftlm_nmu_rect_k: --solver=k-gram cannot be combined with --kblock-prototype\n";
        return 2;
      }
      kblock_proto_pkhpk_mf = false;
      kblock_proto_im_pk_reduced = false;
      kblock_proto_orbit_mf = false;
      kblock_proto_orbit_direct = false;
    } else if (solver == NmuSolver::KOrbit) {
      if (kblock_prototype_arg != nullptr && std::strcmp(kblock_prototype_arg, "orbit-direct") != 0) {
        std::cerr << "bench_ftlm_nmu_rect_k: --solver=k-orbit only allows --kblock-prototype=orbit-direct (or omit it)\n";
        return 2;
      }
      kblock_proto_pkhpk_mf = false;
      kblock_proto_im_pk_reduced = false;
      kblock_proto_orbit_mf = false;
      kblock_proto_orbit_direct = true;
    } else if (solver == NmuSolver::Ed) {
      if (kblock_prototype_arg != nullptr) {
        std::cerr << "bench_ftlm_nmu_rect_k: --solver=ed cannot be combined with --kblock-prototype\n";
        return 2;
      }
      kblock_proto_pkhpk_mf = false;
      kblock_proto_im_pk_reduced = false;
      kblock_proto_orbit_mf = false;
      kblock_proto_orbit_direct = false;
    }
  }

  if (kblock_prototype_arg != nullptr && !kblock_proto_pkhpk_mf && !kblock_proto_im_pk_reduced && !kblock_proto_orbit_mf &&
      !kblock_proto_orbit_direct) {
    std::cerr << "bench_ftlm_nmu_rect_k: unknown --kblock-prototype=" << kblock_prototype_arg
              << " (supported: pkhpk-mf, im-pk-reduced, orbit-matrix-free, orbit-direct)\n";
    return 2;
  }

  // Translation-orbit representatives + matrix-free H_K (no Gram, no zheev); see docs/ORBIT_K_BLOCK_DESIGN.md.
  // When --solver is omitted, prefer this path (Fortran-style parent/orbit indexing). Use --solver=k-gram for
  // HubbardMomentumBlock (Gram + Bloch Φ + eigendecomposition).
  const bool orbit_direct_allowed =
      kblock_proto_orbit_direct || (!solver_set && solver == NmuSolver::KGram);
#if !defined(_WIN32)
  if (mem_report_detail != 0) {
    (void)setenv("FTLM_MEM_REPORT_DETAIL", "1", 1);
  }
#endif
  std::size_t max_gram_storage_bytes = 0;
  std::uint64_t im_pk_gram_prefilter_k_in_zero_skip = 0;
  std::uint64_t orbit_mf_blocks_used = 0;
  std::uint64_t orbit_mf_blocks_skipped_zero = 0;
  std::uint64_t orbit_mf_dim_mismatch_rejected = 0;
  std::uint64_t orbit_direct_blocks_used = 0;
  std::uint64_t orbit_direct_skipped_zero = 0;
  std::uint64_t diag_orbit_mf_dk_sum = 0;
  std::uint64_t diag_orbit_direct_dk_sum = 0;
  std::uint64_t diag_prod_dk_sum = 0;
  std::uint64_t diag_prod_blocks = 0;
  std::uint64_t diag_im_pk_dk_sum = 0;
  std::uint64_t diag_im_pk_blocks = 0;
  std::uint64_t diag_pkhpk_dk_sum = 0;
  std::uint64_t diag_pkhpk_blocks = 0;
  double diag_max_rss_at_block = 0.0;
  int diag_max_rss_nu = -1;
  int diag_max_rss_nd = -1;
  int diag_max_rss_kx = -1;
  int diag_max_rss_ky = -1;

  const auto t_wall0 = clock::now();

  ftlm::HubbardParams p;
  p.Lx = Lx;
  p.Ly = Ly;
  p.t = parse_double_arg(argc, argv, "t", 1.0);
  p.tp = parse_double_arg(argc, argv, "tp", -0.35);
  p.U = parse_double_arg(argc, argv, "U", 5.75);
  p.V = parse_double_arg(argc, argv, "V", 0.9);
  p.phi_x = 0.0;
  p.phi_y = 0.0;

  const int nsec = (n_sites + 1) * (n_sites + 1);
  std::vector<double> logZ(static_cast<size_t>(nsec), -std::numeric_limits<double>::infinity());
  std::vector<int> Nelec(static_cast<size_t>(nsec), 0);

  ftlm::FtlmParams fpar;
  fpar.n_random = n_rand;
  fpar.lanczos_steps = lz_steps;
  fpar.seed = fseed;

  std::cout << "=== FTLM grand-canonical n(mu) " << Lx << "x" << Ly << " with momentum blocks ===\n";
  if (solver_set) {
    std::cout << "[solver] " << solver_label(solver);
    if (solver == NmuSolver::Ed) {
      std::cout << "  ed_cutoff=" << ed_cutoff;
    }
    std::cout << "\n";
  }
  std::cout << "beta=" << beta << "  t=" << p.t << " tp=" << p.tp << " U=" << p.U << " V=" << p.V << "\n";
  std::cout << "ftlm_random=" << n_rand << " lanczos_steps=" << lz_steps << "  sectors=" << nsec << "\n";
  if (!no_monitor) {
    std::cout << "[monitor] wall time + peak RSS reported at exit (disable with --no-monitor=1)\n";
  }
  if (gram_build_no_reuse != 0) {
    std::cout << "[ab] gram_build_no_reuse=1: release Gram-build scratch capacity after each K block\n";
  }
  if (orbit_direct_allowed && !kblock_proto_orbit_direct) {
    std::cout << "[k-path] translation-orbit K-block tried first (no Gram); --solver=k-gram for Gram+Phi+zheev\n";
  }
  if (kblock_proto_pkhpk_mf) {
    std::cout << "[kblock-prototype] pkhpk-mf: matrix-free P_K H P_K on full sector when dim_full<=" << kPkhpkMfMaxDim
              << " (no HubbardMomentumBlock / Gram build for those blocks)\n";
  }
  if (kblock_proto_im_pk_reduced) {
    std::cout << "[kblock-prototype] im-pk-reduced: after Gram k_out check, U from dense P_K, H_red=U†HU when dim_full<="
              << kImPkReducedMaxDim << " and r==k_out; else HubbardMomentumBlock\n";
  }
  if (kblock_proto_orbit_mf) {
    std::cout << "[kblock-prototype] orbit-matrix-free: when dk_orbit==dk_prod, T=Phi^dagger O build + y=T H_orb T^-1 x "
                 "(FTLM random starts stay in Gram coords); else HubbardMomentumBlock; dim_full<="
              << kOrbitMfMaxDim << "\n";
  }
  if (kblock_proto_orbit_direct) {
    std::cout << "[kblock-prototype] orbit-direct: OrbitKBlockBasis + apply_orbit_k_block in orbit coords (no Phi Gram, "
                 "no T bridge); dim_full<="
              << kOrbitMfMaxDim << "\n";
  }
  if ((kblock_proto_orbit_mf || kblock_proto_orbit_direct) && kblock_prototype_diagnostics != 0) {
    std::cout << "[kblock-diagnostics] per-K lines on stderr; kblock-diagnostics-prod-dim="
              << kblock_diagnostics_prod_dim
              << " (1 runs full Gram dim on orbit-mf / orbit-direct blocks — slow)\n";
  }

  const auto t0 = clock::now();
  ftlm::symmetry::HubbardMomentumAction hub(p);
  double rss_peak_run = 0.0;
  for (int nu = 0; nu <= n_sites; ++nu) {
    for (int nd = 0; nd <= n_sites; ++nd) {
      const int idx = sector_index(nu, nd, n_sites);
      Nelec[static_cast<size_t>(idx)] = nu + nd;
      ftlm::FockBasis fb(n_sites, nu, nd);
      const int dim_full = fb.dim();
      if (dim_full <= 0) {
        continue;
      }

      std::vector<ftlm::symmetry::RawState> universe;
      universe.reserve(static_cast<size_t>(dim_full));
      for (int i = 0; i < dim_full; ++i) {
        universe.push_back(ftlm::symmetry::RawState{static_cast<std::uint16_t>(fb.up_mask(i)),
                                                    static_cast<std::uint16_t>(fb.down_mask(i))});
      }

      const auto map = ftlm::symmetry::build_momentum_sector_map_rect(std::move(universe), Lx, Ly);
      std::size_t map_orbit_bytes_est = map.orbits.capacity() * sizeof(ftlm::symmetry::OrbitRecord);
      for (const auto& o : map.orbits) {
        map_orbit_bytes_est += o.stabilizer.capacity() * sizeof(std::pair<int, int>);
      }
      std::size_t map_index_bytes_est = map.orbit_indices_by_momentum.capacity() * sizeof(std::vector<std::size_t>);
      for (const auto& v : map.orbit_indices_by_momentum) {
        map_index_bytes_est += v.capacity() * sizeof(std::size_t);
      }
      const std::size_t fb_bytes_est = static_cast<std::size_t>(fb.dim()) * (2 * sizeof(std::uint64_t));
      const double rss_before_sector = peak_rss_bytes_self();
      double logZ_sector = -std::numeric_limits<double>::infinity();
      int dim_k_total = 0;
      int n_ed_blocks = 0;
      int n_ftlm_blocks = 0;
      int max_dk_sector = 0;
      std::size_t max_k_in_sector = 0;
      std::size_t max_k_out_sector = 0;
      std::size_t max_phi_bytes_sector = 0;
      std::size_t max_gram_g_dense_b = 0;
      std::size_t max_gram_v_b = 0;
      std::size_t max_gram_seed_b = 0;
      std::size_t max_gram_eval_b = 0;
      ExactEdMemStats ed_mem_stats;
      ExactEdScratch ed_scratch;
      // One scratch shared across all K for this (nu,nd): only one HubbardMomentumBlock alive at a time.
      ftlm::symmetry::MomentumBlockScratch sector_k_scratch;
      std::vector<std::complex<double>> pkhpk_w1;
      std::vector<std::complex<double>> pkhpk_w2;
      if (kblock_proto_pkhpk_mf && dim_full <= kPkhpkMfMaxDim) {
        pkhpk_w1.resize(static_cast<std::size_t>(dim_full));
        pkhpk_w2.resize(static_cast<std::size_t>(dim_full));
      }
      ftlm::LanczosComplexWorkspace lanczos_ws;
      ftlm::FtlmTridiagonalQuadratureScratch sector_quad_scratch;
      // Size follows each k-block dim (ftlm_log_partition_complex → lanczos_tridiagonal → ensure(dk)), not dim_full.
      fpar.lanczos_ws = &lanczos_ws;
      fpar.quad_scratch = &sector_quad_scratch;
      for (int ky = 0; ky < Ly; ++ky) {
        for (int kx = 0; kx < Lx; ++kx) {
          const auto t_block0 = clock::now();
          const ftlm::symmetry::MomentumSector K{kx, ky, Lx, Ly};
          const auto kb = ftlm::symmetry::KBasis::build(map, K);
          const bool use_pkhpk_mf_here = kblock_proto_pkhpk_mf && dim_full > 0 && dim_full <= kPkhpkMfMaxDim;
          const bool try_im_pk_reduced_here =
              kblock_proto_im_pk_reduced && dim_full > 0 && dim_full <= kImPkReducedMaxDim;
          // Conservative Step-3 gate: only mixed-spin, non-empty/non-full sectors where tiny parity tests are strongest.
          const bool orbit_mf_sector_safe = (nu > 0 && nd > 0 && nu < n_sites && nd < n_sites);
          const bool try_orbit_direct_here =
              orbit_direct_allowed && orbit_mf_sector_safe && dim_full > 0 && dim_full <= kOrbitMfMaxDim;
          const bool try_orbit_mf_here =
              kblock_proto_orbit_mf && orbit_mf_sector_safe && dim_full > 0 && dim_full <= kOrbitMfMaxDim;

          int dk = 0;
          std::function<void(const std::complex<double>*, std::complex<double>*)> apply_h;
          std::unique_ptr<ftlm::symmetry::HubbardMomentumBlock> kblock_hold;
          std::unique_ptr<ftlm::symmetry::TinyImPkReducedHamiltonian> im_pk_hold;
          std::unique_ptr<ftlm::symmetry::OrbitKBlockBasis> orbit_basis_hold;
          bool use_im_pk_here = false;
          bool use_orbit_mf_here = false;
          bool use_orbit_direct_here = false;

          if (use_pkhpk_mf_here) {
            dk = dim_full;
            max_dk_sector = std::max(max_dk_sector, dk);
            apply_h = [&, K](const std::complex<double>* x, std::complex<double>* y) {
              ftlm::symmetry::tiny_pk::apply_symmetrized_PK_H_PK_vector_matrix_free(
                  p, fb, Lx, Ly, K, hub.hoppings, hub.nn_pairs, x, y, &pkhpk_w1, &pkhpk_w2);
            };
          } else if (try_orbit_direct_here) {
            orbit_basis_hold = std::make_unique<ftlm::symmetry::OrbitKBlockBasis>();
            orbit_basis_hold->build_from_map(map, K);
            const int dk_orbit = orbit_basis_hold->dim();
            if (dk_orbit > 0) {
              dk = dk_orbit;
              ++orbit_direct_blocks_used;
              use_orbit_direct_here = true;
              max_dk_sector = std::max(max_dk_sector, dk);
              apply_h = [ptr = orbit_basis_hold.get(), &p, &fb, &hub](const std::complex<double>* x,
                                                                     std::complex<double>* y) {
                ftlm::symmetry::apply_orbit_k_block(p, fb, *ptr, hub.hoppings, hub.nn_pairs, x, y);
              };
            } else {
              ++orbit_direct_skipped_zero;
              orbit_basis_hold.reset();
            }
          } else if (try_orbit_mf_here) {
            orbit_basis_hold = std::make_unique<ftlm::symmetry::OrbitKBlockBasis>();
            orbit_basis_hold->build_from_map(map, K);
            const int dk_orbit = orbit_basis_hold->dim();
            const std::size_t dk_prod_dim = hub.momentum_block_dim(map, K, nu, nd);
            if (dk_orbit > 0 && static_cast<std::size_t>(dk_orbit) == dk_prod_dim) {
              // FTLM draws random vectors in Gram ONB coordinates. apply_orbit_k_block expects orbit coefficients for
              // the raw Bloch columns O. Use H_Gram = T H_orb T^{-1} with T = Phi^dagger O so Lanczos matches production.
              auto bridge_block =
                  std::make_unique<ftlm::symmetry::HubbardMomentumBlock>(hub, map, K, nu, nd, &sector_k_scratch);
              const int dk_bridge = static_cast<int>(bridge_block->dim());
              if (dk_bridge == dk_orbit) {
                sector_k_scratch.ensure_d_full(dim_full);
                std::vector<C> T(static_cast<std::size_t>(dk_orbit * dk_orbit), C(0.0, 0.0));
                for (int jc = 0; jc < dk_orbit; ++jc) {
                  const auto& rep = orbit_basis_hold->entries()[static_cast<std::size_t>(jc)].rep;
                  ftlm::symmetry::detail::fill_phi_orbit_bloch_from_seed(map, K, Lx, Ly, fb, rep,
                                                                        sector_k_scratch.vin.data());
                  bridge_block->project_full_to_block(sector_k_scratch.vin.data(), T.data() + jc * dk_orbit);
                }
                std::vector<C> Tinv = T;
                if (invert_colmajor_gauss_jordan(dk_orbit, &Tinv)) {
                  bridge_block.reset();
                  auto T_sh = std::make_shared<std::vector<C>>(std::move(T));
                  auto Tinv_sh = std::make_shared<std::vector<C>>(std::move(Tinv));
                  auto xo_sh = std::make_shared<std::vector<C>>(static_cast<std::size_t>(dk_orbit));
                  auto yo_sh = std::make_shared<std::vector<C>>(static_cast<std::size_t>(dk_orbit));
                  dk = dk_orbit;
                  ++orbit_mf_blocks_used;
                  use_orbit_mf_here = true;
                  max_dk_sector = std::max(max_dk_sector, dk);
                  apply_h = [T_sh, Tinv_sh, xo_sh, yo_sh, dk_orbit, ptr = orbit_basis_hold.get(), &p, &fb, &hub](
                                const std::complex<double>* x, std::complex<double>* y) {
                    C* xo = xo_sh->data();
                    C* yo = yo_sh->data();
                    for (int i = 0; i < dk_orbit; ++i) {
                      C acc(0.0, 0.0);
                      for (int k = 0; k < dk_orbit; ++k) {
                        acc += (*Tinv_sh)[static_cast<std::size_t>(i + k * dk_orbit)] * x[k];
                      }
                      xo[i] = acc;
                    }
                    ftlm::symmetry::apply_orbit_k_block(p, fb, *ptr, hub.hoppings, hub.nn_pairs, xo, yo);
                    for (int i = 0; i < dk_orbit; ++i) {
                      C acc(0.0, 0.0);
                      for (int k = 0; k < dk_orbit; ++k) {
                        acc += (*T_sh)[static_cast<std::size_t>(i + k * dk_orbit)] * yo[k];
                      }
                      y[i] = acc;
                    }
                  };
                } else {
                  kblock_hold = std::move(bridge_block);
                  dk = static_cast<int>(kblock_hold->dim());
                  max_dk_sector = std::max(max_dk_sector, dk);
                  max_k_in_sector = std::max(max_k_in_sector, kblock_hold->gram_k_in());
                  max_k_out_sector = std::max(max_k_out_sector, kblock_hold->gram_k_out());
                  max_phi_bytes_sector = std::max(max_phi_bytes_sector, kblock_hold->phi_bytes());
                  max_gram_g_dense_b = std::max(max_gram_g_dense_b, kblock_hold->gram_g_dense_bytes_estimate());
                  max_gram_v_b = std::max(max_gram_v_b, kblock_hold->gram_v_bytes());
                  max_gram_seed_b = std::max(max_gram_seed_b, kblock_hold->gram_seed_bytes());
                  max_gram_eval_b = std::max(max_gram_eval_b, kblock_hold->gram_eval_bytes());
                  if (mem_report != 0) {
                    max_gram_storage_bytes = std::max(max_gram_storage_bytes, kblock_hold->phi_bytes());
                  }
                  apply_h = [ptr = kblock_hold.get()](const std::complex<double>* x, std::complex<double>* y) {
                    ptr->apply(x, y);
                  };
                  orbit_basis_hold.reset();
                }
              } else {
                kblock_hold = std::move(bridge_block);
                dk = static_cast<int>(kblock_hold->dim());
                max_dk_sector = std::max(max_dk_sector, dk);
                max_k_in_sector = std::max(max_k_in_sector, kblock_hold->gram_k_in());
                max_k_out_sector = std::max(max_k_out_sector, kblock_hold->gram_k_out());
                max_phi_bytes_sector = std::max(max_phi_bytes_sector, kblock_hold->phi_bytes());
                max_gram_g_dense_b = std::max(max_gram_g_dense_b, kblock_hold->gram_g_dense_bytes_estimate());
                max_gram_v_b = std::max(max_gram_v_b, kblock_hold->gram_v_bytes());
                max_gram_seed_b = std::max(max_gram_seed_b, kblock_hold->gram_seed_bytes());
                max_gram_eval_b = std::max(max_gram_eval_b, kblock_hold->gram_eval_bytes());
                if (mem_report != 0) {
                  max_gram_storage_bytes = std::max(max_gram_storage_bytes, kblock_hold->phi_bytes());
                }
                apply_h = [ptr = kblock_hold.get()](const std::complex<double>* x, std::complex<double>* y) {
                  ptr->apply(x, y);
                };
                orbit_basis_hold.reset();
              }
            } else {
              if (dk_orbit > 0) {
                ++orbit_mf_dim_mismatch_rejected;
                if (kblock_prototype_diagnostics != 0) {
                  std::cerr << "[orbit-mf-dim-gate] reject nu=" << nu << " nd=" << nd << " k=(" << kx << "," << ky << ")"
                            << " dk_orbit=" << dk_orbit << " dk_prod=" << dk_prod_dim << "\n";
                }
              } else {
                ++orbit_mf_blocks_skipped_zero;
              }
              orbit_basis_hold.reset();
            }
          } else if (try_im_pk_reduced_here) {
            // Same k_out as production (G + zheev + tol_ev) via momentum_phi_gram_k_out_only — no packed V/evals.
            // When k_in==0, fill_gram_zheev_keep_indices fails immediately; skip fill+zheev (same outcome as gram_ok==false).
            std::vector<ftlm::symmetry::RawState> seeds = ftlm::symmetry::momentum_phi_seeds(map, K);
            bool gram_ok = false;
            ftlm::symmetry::GramKOutDiagnostics gram_dim{};
            if (seeds.empty()) {
              ++im_pk_gram_prefilter_k_in_zero_skip;
            } else {
              gram_ok = ftlm::symmetry::momentum_phi_gram_k_out_only(map, K, Lx, Ly, fb, &gram_dim,
                                                                     &sector_k_scratch.zheev,
                                                                     &sector_k_scratch.gram_build, &seeds);
            }
            // When k_out>0 and r==k_out, apply uses H_red=U†HU only (no HubbardMomentumBlock).
            if (gram_ok && gram_dim.k_out > 0) {
              im_pk_hold = std::make_unique<ftlm::symmetry::TinyImPkReducedHamiltonian>();
              if (ftlm::symmetry::build_tiny_im_pk_reduced_hamiltonian(p, fb, Lx, Ly, K, hub.hoppings, hub.nn_pairs,
                                                                       kImPkReducedMaxDim, im_pk_hold.get()) &&
                  im_pk_hold->r > 0 &&
                  static_cast<std::size_t>(im_pk_hold->r) == gram_dim.k_out) {
                dk = im_pk_hold->r;
                use_im_pk_here = true;
                max_dk_sector = std::max(max_dk_sector, dk);
                apply_h = [ptr = im_pk_hold.get()](const std::complex<double>* x, std::complex<double>* y) {
                  ptr->apply(x, y);
                };
              } else {
                im_pk_hold.reset();
              }
            }
          }

          if (dk == 0 && !use_pkhpk_mf_here && !use_orbit_mf_here && !use_orbit_direct_here) {
            kblock_hold = std::make_unique<ftlm::symmetry::HubbardMomentumBlock>(hub, map, K, nu, nd, &sector_k_scratch);
            dk = static_cast<int>(kblock_hold->dim());
            max_dk_sector = std::max(max_dk_sector, dk);
            max_k_in_sector = std::max(max_k_in_sector, kblock_hold->gram_k_in());
            max_k_out_sector = std::max(max_k_out_sector, kblock_hold->gram_k_out());
            max_phi_bytes_sector = std::max(max_phi_bytes_sector, kblock_hold->phi_bytes());
            max_gram_g_dense_b = std::max(max_gram_g_dense_b, kblock_hold->gram_g_dense_bytes_estimate());
            max_gram_v_b = std::max(max_gram_v_b, kblock_hold->gram_v_bytes());
            max_gram_seed_b = std::max(max_gram_seed_b, kblock_hold->gram_seed_bytes());
            max_gram_eval_b = std::max(max_gram_eval_b, kblock_hold->gram_eval_bytes());
            if (mem_report != 0) {
              max_gram_storage_bytes = std::max(max_gram_storage_bytes, kblock_hold->phi_bytes());
            }
            apply_h = [ptr = kblock_hold.get()](const std::complex<double>* x, std::complex<double>* y) {
              ptr->apply(x, y);
            };
          }

          dim_k_total += dk;
          if (log_k_dims) {
            std::cerr << "  k=(" << kx << "," << ky << ") dim=" << dk << " kb_dim=" << kb.dim()
                      << (use_pkhpk_mf_here ? " [pkhpk-mf]" : "")
                      << (use_im_pk_here ? " [im-pk-reduced r=rank(Im P_K) Gram-bypass]" : "")
                      << (use_orbit_mf_here ? " [orbit-matrix-free]" : "")
                      << (use_orbit_direct_here ? " [orbit-direct]" : "") << "\n";
          }

          std::size_t prod_dim_compare = 0;
          if (kblock_prototype_diagnostics != 0 && kblock_diagnostics_prod_dim != 0 && dk > 0 &&
              ((kblock_proto_orbit_mf && use_orbit_mf_here) || (kblock_proto_orbit_direct && use_orbit_direct_here))) {
            prod_dim_compare = hub.momentum_block_dim(map, K, nu, nd);
          }

          const char* path_tag = "none";
          if (use_pkhpk_mf_here) {
            path_tag = "pkhpk-mf";
          } else if (use_orbit_mf_here) {
            path_tag = "orbit-mf";
          } else if (use_orbit_direct_here) {
            path_tag = "orbit-direct";
          } else if (use_im_pk_here) {
            path_tag = "im-pk-reduced";
          } else if (kblock_hold) {
            path_tag = "production";
          }
          const std::size_t orbit_dim = orbit_basis_hold ? orbit_basis_hold->dim() : 0;
          const std::size_t prod_dim_from_block = kblock_hold ? static_cast<std::size_t>(kblock_hold->dim()) : 0;

          if ((kblock_proto_orbit_mf || kblock_proto_orbit_direct) && kblock_prototype_diagnostics != 0) {
            if (use_pkhpk_mf_here) {
              ++diag_pkhpk_blocks;
              diag_pkhpk_dk_sum += static_cast<std::uint64_t>(dk);
            } else if (use_orbit_mf_here) {
              diag_orbit_mf_dk_sum += static_cast<std::uint64_t>(dk);
            } else if (use_orbit_direct_here) {
              diag_orbit_direct_dk_sum += static_cast<std::uint64_t>(dk);
            } else if (use_im_pk_here) {
              ++diag_im_pk_blocks;
              diag_im_pk_dk_sum += static_cast<std::uint64_t>(dk);
            } else if (kblock_hold) {
              ++diag_prod_blocks;
              diag_prod_dk_sum += static_cast<std::uint64_t>(dk);
            }
            if (dk <= 0) {
              const double rss_now = peak_rss_bytes_self();
              if (rss_now > diag_max_rss_at_block) {
                diag_max_rss_at_block = rss_now;
                diag_max_rss_nu = nu;
                diag_max_rss_nd = nd;
                diag_max_rss_kx = kx;
                diag_max_rss_ky = ky;
              }
              const double wall_s = std::chrono::duration<double>(clock::now() - t_block0).count();
              std::cerr << "[kblock-diag] nu=" << nu << " nd=" << nd << " k=(" << kx << "," << ky << ")"
                        << " dim_full=" << dim_full << " kb_dim=" << kb.dim() << " dk=" << dk << " path=" << path_tag
                        << " orbit_dim=" << orbit_dim << " prod_dim_block=" << prod_dim_from_block
                        << " prod_dim_gram_compare=" << prod_dim_compare << " wall_s=" << wall_s
                        << " rss_peak_B=" << rss_now << " (no_lz)\n";
              continue;
            }
          } else if (dk <= 0) {
            continue;
          }
          if (lanczos_ws_report != 0) {
            std::cerr << "[lanczos_ws] sector_idx=" << idx << " k=(" << kx << "," << ky << ") d_K=" << dk
                      << " shared_pool_bytes=" << lanczos_ws.bytes_capacity() << "\n";
          }

          double lz_k = -std::numeric_limits<double>::infinity();
          if (dk <= ed_cutoff) {
            ++n_ed_blocks;
            lz_k = exact_log_partition_small_complex(dk, apply_h, beta, &ed_mem_stats, &ed_scratch);
          } else {
            ++n_ftlm_blocks;
            const unsigned seed_k = fseed + static_cast<unsigned>(idx * 257 + ky * 17 + kx);
            fpar.seed = seed_k;
            if (kblock_hold) {
              lz_k = ftlm::symmetry::ftlm_log_partition_hubbard_momentum_block_ref(*kblock_hold, beta, fpar);
            } else {
              lz_k = ftlm::ftlm_log_partition_complex(dk, apply_h, beta, fpar);
            }
          }
          logZ_sector = logsumexp2(logZ_sector, lz_k);
          if (!use_pkhpk_mf_here && !use_im_pk_here && !use_orbit_mf_here && !use_orbit_direct_here &&
              gram_build_no_reuse != 0) {
            sector_k_scratch.gram_build.release_capacity();
          }
          if ((kblock_proto_orbit_mf || kblock_proto_orbit_direct) && kblock_prototype_diagnostics != 0) {
            const double rss_now = peak_rss_bytes_self();
            if (rss_now > diag_max_rss_at_block) {
              diag_max_rss_at_block = rss_now;
              diag_max_rss_nu = nu;
              diag_max_rss_nd = nd;
              diag_max_rss_kx = kx;
              diag_max_rss_ky = ky;
            }
            const double wall_s = std::chrono::duration<double>(clock::now() - t_block0).count();
            std::cerr << "[kblock-diag] nu=" << nu << " nd=" << nd << " k=(" << kx << "," << ky << ")"
                      << " dim_full=" << dim_full << " kb_dim=" << kb.dim() << " dk=" << dk << " path=" << path_tag
                      << " orbit_dim=" << orbit_dim << " prod_dim_block=" << prod_dim_from_block
                      << " prod_dim_gram_compare=" << prod_dim_compare << " wall_s=" << wall_s
                      << " lz_k=" << lz_k << " rss_peak_B=" << rss_now << "\n";
          }
        }
      }
      sector_k_scratch.shrink_after_sector();
      lanczos_ws.shrink_to_fit();
      sector_quad_scratch.shrink_to_fit();
      fpar.lanczos_ws = nullptr;
      fpar.quad_scratch = nullptr;

      const double rss_after_sector = peak_rss_bytes_self();
      rss_peak_run = std::max(rss_peak_run, rss_after_sector);
      if (mem_instrument != 0) {
        const std::size_t zwc = sector_k_scratch.zheev.work.capacity();
        const std::size_t zrwc = sector_k_scratch.zheev.rwork.capacity();
        std::cerr << "[mem-instr] sector(" << nu << "," << nd << ") idx=" << idx << " max_d_k=" << max_dk_sector
                  << " max_k_in=" << max_k_in_sector << " max_k_out=" << max_k_out_sector
                  << " gram_phi_cap_B=" << max_phi_bytes_sector << " gram_G_dense_est_B=" << max_gram_g_dense_b
                  << " gram_V_B=" << max_gram_v_b << " gram_seed_B=" << max_gram_seed_b
                  << " gram_eval_B=" << max_gram_eval_b << " ed_ran=" << (ed_mem_stats.ran ? 1 : 0)
                  << " ed_max_d_k=" << ed_mem_stats.max_dim << " ed_H_dense_B=" << ed_mem_stats.max_H_dense_B
                  << " ed_R_dense_B=" << ed_mem_stats.max_R_dense_B << " ed_aux_B=" << ed_mem_stats.max_aux_vecs_B
                  << " ed_total_dense_B=" << ed_mem_stats.max_total_B
                  << " ed_reused_buffers=" << (ed_mem_stats.reused_buffers ? 1 : 0)
                  << " map_orbits_B_est=" << map_orbit_bytes_est << " map_indices_B_est=" << map_index_bytes_est
                  << " fb_major_B_est=" << fb_bytes_est
                  << " zheev_work_cap_B=" << (zwc * sizeof(std::complex<double>))
                  << " zheev_work_size=" << sector_k_scratch.zheev.work.size()
                  << " zheev_rwork_cap_B=" << (zrwc * sizeof(double))
                  << " zheev_rwork_size=" << sector_k_scratch.zheev.rwork.size()
                  << " quad_T_cap_B=" << (sector_quad_scratch.T_flat.capacity() * sizeof(double))
                  << " quad_T_size=" << sector_quad_scratch.T_flat.size()
                  << " quad_V_cap_B=" << (sector_quad_scratch.V_flat.capacity() * sizeof(double))
                  << " quad_V_size=" << sector_quad_scratch.V_flat.size()
                  << " lanczos_pool_B=" << lanczos_ws.bytes_capacity() << " rss_before_B=" << rss_before_sector
                  << " rss_after_B=" << rss_after_sector << " peak_run_B=" << rss_peak_run << "\n";
      }

      logZ[static_cast<size_t>(idx)] = logZ_sector;
      std::cerr << "sector (" << nu << "," << nd << ") dim_full=" << dim_full << " dim_k_total=" << dim_k_total
                << " ed_blocks=" << n_ed_blocks << " ftlm_blocks=" << n_ftlm_blocks << " logZ=" << logZ_sector
                << "\n";
    }
  }
  const double t_sectors = std::chrono::duration<double>(clock::now() - t0).count();
  std::cout << "[timing] all sectors logZ: " << t_sectors << " s\n";
  if (mem_instrument != 0) {
    std::cout << "[mem-instr] rss_peak_run_B=" << rss_peak_run << " rss_peak_run_MiB=" << (rss_peak_run / (1024.0 * 1024.0))
              << "\n";
  }

  std::ostream* out = &std::cout;
  std::ofstream file;
  if (out_path) {
    file.open(out_path);
    if (!file) {
      std::cerr << "bench_ftlm_nmu_rect_k: cannot open " << out_path << "\n";
      return 2;
    }
    out = &file;
  }

  *out << "# FTLM grand canonical (rect k-block) Lx=" << Lx << " Ly=" << Ly << " beta=" << beta << " t=" << p.t << " tp=" << p.tp
       << " U=" << p.U << " V=" << p.V << "\n";
  *out << "# ftlm_random=" << n_rand << " lanczos_steps=" << lz_steps << " sector_time_s=" << t_sectors << "\n";

  for (int k = 0; k < n_mu; ++k) {
    const double tmu = static_cast<double>(k) / static_cast<double>(n_mu - 1);
    const double mu = mu_min + tmu * (mu_max - mu_min);
    double mx = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < nsec; ++i) {
      const double lz = logZ[static_cast<size_t>(i)];
      if (!std::isfinite(lz)) {
        continue;
      }
      const int N = Nelec[static_cast<size_t>(i)];
      const double ex = lz + beta * mu * static_cast<double>(N);
      mx = std::max(mx, ex);
    }
    if (!std::isfinite(mx)) {
      *out << mu << "\t0\n";
      continue;
    }
    double sum_w = 0.0;
    double sum_Nw = 0.0;
    for (int i = 0; i < nsec; ++i) {
      const double lz = logZ[static_cast<size_t>(i)];
      if (!std::isfinite(lz)) {
        continue;
      }
      const int N = Nelec[static_cast<size_t>(i)];
      const double ex = lz + beta * mu * static_cast<double>(N) - mx;
      const double w = std::exp(ex);
      sum_w += w;
      sum_Nw += static_cast<double>(N) * w;
    }
    const double n_avg = (sum_w > 0.0) ? (sum_Nw / sum_w) / static_cast<double>(n_sites) : 0.0;
    *out << mu << "\t" << n_avg << "\n";
  }

  if (out_path) {
    std::cout << "wrote " << out_path << "\n";
  }
  const double t_wall = std::chrono::duration<double>(clock::now() - t_wall0).count();
  const double peak_b = peak_rss_bytes_self();
  const double peak_mib = peak_b / (1024.0 * 1024.0);
  if (mem_report != 0) {
    std::cout << "[mem] max_gram_storage_bytes=" << max_gram_storage_bytes << " peak_rss_mib=" << peak_mib
              << " peak_rss_bytes=" << peak_b << "\n";
  }
  if (!no_monitor) {
    std::cout << "[monitor] wall_time_s=" << t_wall << "  wall_sector_logZ_s=" << t_sectors
              << "  peak_rss_mib=" << peak_mib << "  peak_rss_bytes=" << peak_b << "\n";
    if (out_path && file.is_open()) {
      *out << "# wall_time_s=" << t_wall << " peak_rss_bytes=" << peak_b << " peak_rss_mib=" << peak_mib << "\n";
      file.flush();
    }
  }
  const char* kblock_proto_metric = "none";
  if (kblock_proto_pkhpk_mf) {
    kblock_proto_metric = "pkhpk-mf";
  } else if (kblock_proto_im_pk_reduced) {
    kblock_proto_metric = "im-pk-reduced";
  } else if (kblock_proto_orbit_mf) {
    kblock_proto_metric = "orbit-matrix-free";
  } else if (kblock_proto_orbit_direct) {
    kblock_proto_metric = "orbit-direct";
  } else if (orbit_direct_allowed) {
    kblock_proto_metric = "orbit-direct-default";
  }
  std::cout << "METRIC kind=FTLM_nmu_rect_k solver=" << (solver_set ? solver_label(solver) : "default")
            << " wall_time_s=" << t_wall << " wall_sector_logZ_s=" << t_sectors << " peak_rss_bytes=" << peak_b
            << " peak_rss_mib=" << peak_mib << " n_mu=" << n_mu << " gram_build_no_reuse=" << gram_build_no_reuse
            << " kblock_prototype=" << kblock_proto_metric;
  if (kblock_proto_im_pk_reduced) {
    std::cout << " im_pk_gram_prefilter_k_in_zero_skip=" << im_pk_gram_prefilter_k_in_zero_skip;
  }
  if (kblock_proto_orbit_mf) {
    std::cout << " orbit_mf_blocks_used=" << orbit_mf_blocks_used
              << " orbit_mf_blocks_skipped_zero=" << orbit_mf_blocks_skipped_zero
              << " orbit_mf_dim_mismatch_rejected=" << orbit_mf_dim_mismatch_rejected;
  }
  if (orbit_direct_blocks_used > 0 || orbit_direct_skipped_zero > 0) {
    std::cout << " orbit_direct_blocks_used=" << orbit_direct_blocks_used
              << " orbit_direct_skipped_zero=" << orbit_direct_skipped_zero;
  }
  if ((kblock_proto_orbit_mf || kblock_proto_orbit_direct) && kblock_prototype_diagnostics != 0) {
    std::cout << " diag_prod_blocks=" << diag_prod_blocks << " diag_prod_dk_sum=" << diag_prod_dk_sum
              << " diag_orbit_mf_dk_sum=" << diag_orbit_mf_dk_sum
              << " diag_orbit_direct_dk_sum=" << diag_orbit_direct_dk_sum << " diag_pkhpk_blocks=" << diag_pkhpk_blocks
              << " diag_pkhpk_dk_sum=" << diag_pkhpk_dk_sum << " diag_im_pk_blocks=" << diag_im_pk_blocks
              << " diag_im_pk_dk_sum=" << diag_im_pk_dk_sum << " diag_max_rss_block_nu_nd=(" << diag_max_rss_nu << ","
              << diag_max_rss_nd << ") diag_max_rss_block_k=(" << diag_max_rss_kx << "," << diag_max_rss_ky
              << ") diag_peak_rss_B_at_block=" << diag_max_rss_at_block;
    std::cout << std::flush;
    std::cerr << "[kblock-diag-summary] orbit_mf_blocks_used=" << orbit_mf_blocks_used
              << " orbit_mf_skipped_zero=" << orbit_mf_blocks_skipped_zero
              << " orbit_mf_dim_mismatch_rejected=" << orbit_mf_dim_mismatch_rejected
              << " orbit_direct_blocks_used=" << orbit_direct_blocks_used
              << " orbit_direct_skipped_zero=" << orbit_direct_skipped_zero
              << " prod_blocks=" << diag_prod_blocks << " sum_dk_prod=" << diag_prod_dk_sum
              << " sum_dk_orbit_mf=" << diag_orbit_mf_dk_sum << " sum_dk_orbit_direct=" << diag_orbit_direct_dk_sum
              << " pkhpk_blocks=" << diag_pkhpk_blocks << " im_pk_blocks=" << diag_im_pk_blocks
              << " max_rss_at_block_nu_nd=(" << diag_max_rss_nu << "," << diag_max_rss_nd << ") k=(" << diag_max_rss_kx
              << "," << diag_max_rss_ky << ") diag_peak_rss_B=" << diag_max_rss_at_block << "\n";
  }
  std::cout << "\n";
  return 0;
}
