// 4×2 cluster: matrix-free stochastic Lanczos workload (FTLM precursor) — time + peak RSS + H|v> counts.
// Full finite-temperature FTLM estimators are not implemented yet; this measures the Krylov/matvec cost
// profile you would embed in an FTLM driver (half-filled sector by default).
#include <chrono>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>

#include "ftlm/fock_basis.hpp"
#include "ftlm/hubbard_hamiltonian.hpp"
#include "ftlm/hubbard_params.hpp"
#include "ftlm/lanczos.hpp"
#include "ftlm/lattice.hpp"

#if defined(__APPLE__) || defined(__FreeBSD__)
#include <sys/resource.h>
#include <sys/time.h>
#else
#include <sys/resource.h>
#include <sys/time.h>
#endif

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

/// ru_maxrss: bytes on macOS, kilobytes on Linux (getrusage(2)).
double max_rss_mib() {
  struct rusage ru {};
  if (getrusage(RUSAGE_SELF, &ru) != 0) {
    return 0.0;
  }
#if defined(__APPLE__) || defined(__FreeBSD__)
  return static_cast<double>(ru.ru_maxrss) / (1024.0 * 1024.0);
#else
  return static_cast<double>(ru.ru_maxrss) / 1024.0;
#endif
}

}  // namespace

int main(int argc, char** argv) {
  const int samples = parse_int_arg(argc, argv, "samples", 32);
  const int lanczos_steps = parse_int_arg(argc, argv, "lanczos-steps", 128);
  const unsigned seed = static_cast<unsigned>(parse_int_arg(argc, argv, "seed", 42));
  const int n_up = parse_int_arg(argc, argv, "n-up", 4);
  const int n_dn = parse_int_arg(argc, argv, "n-dn", 4);

  ftlm::HubbardParams p;
  p.Lx = 4;
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

  ftlm::FockBasis basis(n_sites, n_up, n_dn);
  const int dim = basis.dim();

  long long matvec_count = 0;
  auto apply_counted = [&](const std::complex<double>* x, std::complex<double>* y) {
    ++matvec_count;
    ftlm::apply_extended_hubbard(p, basis, hops, pairs, x, y);
  };

  std::cout << "=== 4×2 FTLM precursor: stochastic Lanczos (matrix-free) ===\n";
  std::cout << "sector N_up=" << n_up << " N_dn=" << n_dn << " dim=" << dim << "  samples=" << samples
            << "  lanczos-steps=" << lanczos_steps << "\n";
  std::cout << "t=" << p.t << " t'=" << p.tp << " U=" << p.U << " V=" << p.V << "\n";

  const double rss0 = max_rss_mib();
  const auto t0 = clock::now();
  long long sum_steps = 0;
  double sum_e_min = 0.0;
  double sum_e_max = 0.0;

  for (int s = 0; s < samples; ++s) {
    const ftlm::LanczosExtrema lz = ftlm::lanczos_extrema(
        dim, apply_counted, lanczos_steps, seed + static_cast<unsigned>(s) * 7919u);
    sum_steps += lz.steps_used;
    sum_e_min += lz.min_eval;
    sum_e_max += lz.max_eval;
  }

  const double wall_s = std::chrono::duration<double>(clock::now() - t0).count();
  const double rss_peak = max_rss_mib();

  std::cout << "avg Ritz E_min=" << (sum_e_min / samples) << "  E_max=" << (sum_e_max / samples)
            << " (not thermodynamic observables; diagnostic only)\n";
  std::cout << "METRIC kind=FTLM_precursor_lanczos Lx=" << p.Lx << " Ly=" << p.Ly << " dim=" << dim
            << " n_up=" << n_up << " n_dn=" << n_dn << " samples=" << samples
            << " lanczos_steps_req=" << lanczos_steps << " sum_lanczos_steps_used=" << sum_steps
            << " matvec_count=" << matvec_count << " wall_s=" << wall_s << " max_rss_mib=" << rss_peak
            << " rss_baseline_mib=" << rss0 << "\n";
  return 0;
}
