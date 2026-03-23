// Grand-canonical n(μ) at fixed β via FTLM: stochastic Lanczos estimate of Tr_s e^{-βH} per
// (N_up,N_down) sector, then exact combination over sectors (same as ED bookkeeping).
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
#include <vector>

#if !defined(_WIN32)
#include <sys/resource.h>
#endif

#include "ftlm/fock_basis.hpp"
#include "ftlm/ftlm_thermo.hpp"
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

int sector_index(int nu, int nd, int n_sites) { return nu * (n_sites + 1) + nd; }

/// Peak RSS for this process in bytes (best-effort; 0 if unavailable).
double peak_rss_bytes_self() {
#if defined(_WIN32)
  (void)0;
  return 0.0;
#else
  struct rusage ru {};
  if (getrusage(RUSAGE_SELF, &ru) != 0) {
    return 0.0;
  }
#if defined(__APPLE__)
  // macOS: ru_maxrss is bytes.
  return static_cast<double>(ru.ru_maxrss);
#else
  // Linux and typical BSD: ru_maxrss is kilobytes.
  return static_cast<double>(ru.ru_maxrss) * 1024.0;
#endif
#endif
}

}  // namespace

int main(int argc, char** argv) {
  const int Lx = parse_int_arg(argc, argv, "Lx", 4);
  const int Ly = parse_int_arg(argc, argv, "Ly", 2);
  const double beta = parse_double_arg(argc, argv, "beta", 20.0);
  const double mu_min = parse_double_arg(argc, argv, "mu-min", -5.0);
  const double mu_max = parse_double_arg(argc, argv, "mu-max", 20.0);
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

  std::cout << "=== FTLM grand-canonical n(mu)  " << Lx << "x" << Ly << "  beta=" << beta << " ===\n";
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

      const double lz =
          ftlm::ftlm_log_partition_real(dim, apply_real, beta, fpar);
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
      std::cerr << "bench_ftlm_nmu_rect: cannot open " << out_path << "\n";
      return 2;
    }
    out = &file;
  }

  *out << "# FTLM grand canonical  Lx=" << Lx << " Ly=" << Ly << "  beta=" << beta << "  t=" << p.t
       << " tp=" << p.tp << " U=" << p.U << " V=" << p.V << "\n";
  *out << "# ftlm_random=" << n_rand << " lanczos_steps=" << lz_steps << "  sector_time_s=" << t_sectors
       << "\n";

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

  std::cout << "METRIC kind=FTLM_nmu_rect Lx=" << Lx << " Ly=" << Ly << " beta=" << beta
            << " wall_time_s=" << t_wall << " wall_sector_logZ_s=" << t_sectors << " peak_rss_bytes=" << peak_b
            << " peak_rss_mib=" << peak_mib << " n_mu=" << n_mu;
  if (no_monitor) {
    std::cout << " no_monitor=1";
  }
  std::cout << "\n";
  return 0;
}
