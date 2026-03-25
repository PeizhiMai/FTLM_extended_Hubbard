// Grand-canonical n(mu) at fixed beta via FTLM using translation-symmetry momentum blocks.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <sys/resource.h>
#endif

#include "ftlm/fock_basis.hpp"
#include "ftlm/ftlm_thermo.hpp"
#include "ftlm/hubbard_params.hpp"
#include "ftlm/symmetry/hubbard_momentum_action.hpp"
#include "ftlm/symmetry/k_basis.hpp"
#include "ftlm/symmetry/momentum_sector.hpp"
#include "ftlm/symmetry/raw_state.hpp"

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

double exact_log_partition_small_complex(
    int dim, const std::function<void(const std::complex<double>*, std::complex<double>*)>& apply_h, double beta) {
  if (dim <= 0) {
    return -std::numeric_limits<double>::infinity();
  }
  if (dim == 1) {
    std::complex<double> x[1] = {std::complex<double>(1.0, 0.0)};
    std::complex<double> y[1] = {std::complex<double>(0.0, 0.0)};
    apply_h(x, y);
    return -beta * y[0].real();
  }

  std::vector<std::vector<std::complex<double>>> H(
      static_cast<size_t>(dim), std::vector<std::complex<double>>(static_cast<size_t>(dim), {0.0, 0.0}));
  std::vector<std::complex<double>> e(static_cast<size_t>(dim), {0.0, 0.0});
  std::vector<std::complex<double>> y(static_cast<size_t>(dim), {0.0, 0.0});
  for (int j = 0; j < dim; ++j) {
    std::fill(e.begin(), e.end(), std::complex<double>(0.0, 0.0));
    e[static_cast<size_t>(j)] = {1.0, 0.0};
    apply_h(e.data(), y.data());
    for (int i = 0; i < dim; ++i) {
      H[static_cast<size_t>(i)][static_cast<size_t>(j)] = y[static_cast<size_t>(i)];
    }
  }

  const int m = 2 * dim;
  std::vector<std::vector<double>> R(static_cast<size_t>(m), std::vector<double>(static_cast<size_t>(m), 0.0));
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

}  // namespace

int main(int argc, char** argv) {
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
  const int lz_default = std::max(20, (2 * a_nonmom) / std::max(1, n_sites));
  const int lz_steps = has_arg(argc, argv, "lanczos-steps")
                           ? parse_int_arg(argc, argv, "lanczos-steps", lz_default)
                           : lz_default;
  const unsigned fseed = static_cast<unsigned>(parse_int_arg(argc, argv, "seed", 7));
  const char* out_path = parse_string_arg(argc, argv, "out");
  const int no_monitor = parse_int_arg(argc, argv, "no-monitor", 0);
  const int no_log_k_dims = parse_int_arg(argc, argv, "no-log-k-dims", 0);
  const bool log_k_dims = (no_log_k_dims == 0);
  const int ed_cutoff = parse_int_arg(argc, argv, "ed-cutoff", 64);
  const int mem_report = parse_int_arg(argc, argv, "mem-report", 0);
  const int mem_report_detail = parse_int_arg(argc, argv, "mem-report-detail", 0);
  const int lanczos_ws_report = parse_int_arg(argc, argv, "lanczos-ws-report", 0);
#if !defined(_WIN32)
  if (mem_report_detail != 0) {
    (void)setenv("FTLM_MEM_REPORT_DETAIL", "1", 1);
  }
#endif
  std::size_t max_gram_storage_bytes = 0;

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
  std::cout << "beta=" << beta << "  t=" << p.t << " tp=" << p.tp << " U=" << p.U << " V=" << p.V << "\n";
  std::cout << "ftlm_random=" << n_rand << " lanczos_steps=" << lz_steps << "  sectors=" << nsec << "\n";
  if (!no_monitor) {
    std::cout << "[monitor] wall time + peak RSS reported at exit (disable with --no-monitor=1)\n";
  }

  const auto t0 = clock::now();
  ftlm::symmetry::HubbardMomentumAction hub(p);
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
      double logZ_sector = -std::numeric_limits<double>::infinity();
      int dim_k_total = 0;
      int n_ed_blocks = 0;
      int n_ftlm_blocks = 0;
      // One scratch shared across all K for this (nu,nd): only one HubbardMomentumBlock alive at a time.
      ftlm::symmetry::MomentumBlockScratch sector_k_scratch;
      ftlm::LanczosComplexWorkspace lanczos_ws;
      // Size follows each k-block dim (ftlm_log_partition_complex → lanczos_tridiagonal → ensure(dk)), not dim_full.
      fpar.lanczos_ws = &lanczos_ws;
      for (int ky = 0; ky < Ly; ++ky) {
        for (int kx = 0; kx < Lx; ++kx) {
          const ftlm::symmetry::MomentumSector K{kx, ky, Lx, Ly};
          const auto kb = ftlm::symmetry::KBasis::build(map, K);
          ftlm::symmetry::HubbardMomentumBlock kblock(hub, map, K, nu, nd, &sector_k_scratch);
          const int dk = static_cast<int>(kblock.dim());
          if (mem_report != 0) {
            max_gram_storage_bytes = std::max(max_gram_storage_bytes, kblock.phi_bytes());
          }
          dim_k_total += dk;
          if (log_k_dims) {
            std::cerr << "  k=(" << kx << "," << ky << ") dim=" << dk << " kb_dim=" << kb.dim() << "\n";
          }
          if (dk <= 0) {
            continue;
          }
          if (lanczos_ws_report != 0) {
            std::cerr << "[lanczos_ws] sector_idx=" << idx << " k=(" << kx << "," << ky << ") d_K=" << dk
                      << " shared_pool_bytes=" << lanczos_ws.bytes_capacity() << "\n";
          }

          auto apply_h = [&](const std::complex<double>* x, std::complex<double>* y) { kblock.apply(x, y); };

          double lz_k = -std::numeric_limits<double>::infinity();
          if (dk <= ed_cutoff) {
            ++n_ed_blocks;
            lz_k = exact_log_partition_small_complex(dk, apply_h, beta);
          } else {
            ++n_ftlm_blocks;
            const unsigned seed_k = fseed + static_cast<unsigned>(idx * 257 + ky * 17 + kx);
            fpar.seed = seed_k;
            lz_k = ftlm::ftlm_log_partition_complex(dk, apply_h, beta, fpar);
          }
          logZ_sector = logsumexp2(logZ_sector, lz_k);
        }
      }
      sector_k_scratch.shrink_after_sector();
      fpar.lanczos_ws = nullptr;

      logZ[static_cast<size_t>(idx)] = logZ_sector;
      std::cerr << "sector (" << nu << "," << nd << ") dim_full=" << dim_full << " dim_k_total=" << dim_k_total
                << " ed_blocks=" << n_ed_blocks << " ftlm_blocks=" << n_ftlm_blocks << " logZ=" << logZ_sector
                << "\n";
    }
  }
  const double t_sectors = std::chrono::duration<double>(clock::now() - t0).count();
  std::cout << "[timing] all sectors logZ: " << t_sectors << " s\n";

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
  std::cout << "METRIC kind=FTLM_nmu_rect_k wall_time_s=" << t_wall << " wall_sector_logZ_s=" << t_sectors
            << " peak_rss_bytes=" << peak_b << " peak_rss_mib=" << peak_mib << " n_mu=" << n_mu << "\n";
  return 0;
}
