// 2×2 cluster (4 sites, periodic torus): matvec throughput + Lanczos ground energy vs dense reference.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "ftlm/fock_basis.hpp"
#include "ftlm/hubbard_hamiltonian.hpp"
#include "ftlm/hubbard_params.hpp"
#include "ftlm/lanczos.hpp"
#include "ftlm/lattice.hpp"

namespace {

using clock = std::chrono::high_resolution_clock;

// Dense reference: classic Jacobi needs many sweeps for n≈36; tridiagonal matrices converge faster.
constexpr int kDenseJacobiSweeps = 1600;

double seconds_since(clock::time_point t0) {
  return std::chrono::duration<double>(clock::now() - t0).count();
}

std::pair<double, double> jacobi_min_max(std::vector<std::vector<double>> A) {
  const int n = static_cast<int>(A.size());
  if (n == 0) {
    return {0.0, 0.0};
  }
  if (n == 1) {
    return {A[0][0], A[0][0]};
  }
  for (int sweep = 0; sweep < kDenseJacobiSweeps; ++sweep) {
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

/// Full symmetric spectrum: Jacobi until max |off-diag| < tol or max_sweeps, then sort diagonal.
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
      const double im = out[static_cast<size_t>(i)].imag();
      if (std::abs(im) > 1e-10) {
        std::cerr << "bench_2x2: expected real-symmetric H (set phi_x=phi_y=0); imag residual " << im
                  << " at (" << i << "," << j << ")\n";
        std::exit(2);
      }
      (*Hreal)[static_cast<size_t>(i)][static_cast<size_t>(j)] = out[static_cast<size_t>(i)].real();
    }
  }
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

std::complex<double> dotc_bench(const std::vector<std::complex<double>>& a,
                               const std::vector<std::complex<double>>& b) {
  std::complex<double> s(0.0, 0.0);
  for (size_t i = 0; i < a.size(); ++i) {
    s += std::conj(a[i]) * b[i];
  }
  return s;
}

double norm2_bench(const std::vector<std::complex<double>>& v) {
  double s = 0.0;
  for (const auto& z : v) {
    s += std::norm(z);
  }
  return s;
}

/// Hermitian Lanczos (same convention as `ftlm::lanczos_extrema`): returns α, β for Krylov dimension `used`.
void lanczos_tridiagonal(const std::function<void(const std::complex<double>*, std::complex<double>*)>& apply_h,
                         int dim, int max_steps, unsigned seed, std::vector<double>* alpha_out,
                         std::vector<double>* beta_out, int* used_out) {
  alpha_out->clear();
  beta_out->clear();
  *used_out = 0;
  if (dim <= 0 || max_steps <= 0) {
    return;
  }
  std::mt19937 rng(seed);
  std::normal_distribution<double> gauss(0.0, 1.0);
  std::vector<std::complex<double>> q(static_cast<size_t>(dim));
  std::vector<std::complex<double>> q_prev(static_cast<size_t>(dim), std::complex<double>(0.0, 0.0));
  std::vector<std::complex<double>> w(static_cast<size_t>(dim));
  for (int i = 0; i < dim; ++i) {
    q[static_cast<size_t>(i)] = {gauss(rng), gauss(rng)};
  }
  double n0 = std::sqrt(norm2_bench(q));
  if (n0 < 1e-18) {
    q[0] = {1.0, 0.0};
    n0 = 1.0;
  } else {
    for (int i = 0; i < dim; ++i) {
      q[static_cast<size_t>(i)] /= n0;
    }
  }
  double beta_prev = 0.0;
  for (int k = 0; k < max_steps; ++k) {
    apply_h(q.data(), w.data());
    const std::complex<double> z = dotc_bench(q, w);
    const double a_k = z.real();
    for (int i = 0; i < dim; ++i) {
      w[static_cast<size_t>(i)] -= z * q[static_cast<size_t>(i)];
    }
    if (k > 0) {
      for (int i = 0; i < dim; ++i) {
        w[static_cast<size_t>(i)] -= beta_prev * q_prev[static_cast<size_t>(i)];
      }
    }
    const double nw2 = norm2_bench(w);
    alpha_out->push_back(a_k);
    if (nw2 < 1e-24) {
      *used_out = k + 1;
      return;
    }
    const double b_k = std::sqrt(nw2);
    beta_out->push_back(b_k);
    q_prev = q;
    for (int i = 0; i < dim; ++i) {
      q[static_cast<size_t>(i)] = w[static_cast<size_t>(i)] / b_k;
    }
    beta_prev = b_k;
    *used_out = k + 1;
  }
}

std::vector<double> sorted_ritz_from_tridiagonal(const std::vector<double>& alpha,
                                                 const std::vector<double>& beta, int used) {
  if (used <= 0) {
    return {};
  }
  std::vector<std::vector<double>> T(static_cast<size_t>(used), std::vector<double>(static_cast<size_t>(used), 0.0));
  for (int i = 0; i < used; ++i) {
    T[static_cast<size_t>(i)][static_cast<size_t>(i)] = alpha[static_cast<size_t>(i)];
  }
  const int nb = static_cast<int>(beta.size());
  for (int i = 0; i < used - 1 && i < nb; ++i) {
    const double b = beta[static_cast<size_t>(i)];
    T[static_cast<size_t>(i)][static_cast<size_t>(i + 1)] = b;
    T[static_cast<size_t>(i + 1)][static_cast<size_t>(i)] = b;
  }
  return sorted_eigenvalues_jacobi(std::move(T), 6000, 1e-12);
}

void write_plot_data(const std::string& dir, const std::vector<double>& dense_evals,
                     const std::vector<double>& ritz_evals, const ftlm::HubbardParams& p, int n_up, int n_dn,
                     int lanczos_used) {
  namespace fs = std::filesystem;
  fs::create_directories(dir);
  {
    std::ofstream f(dir + "/dense_evals.txt");
    for (double e : dense_evals) {
      f << e << "\n";
    }
  }
  {
    std::ofstream f(dir + "/lanczos_ritz.txt");
    for (double e : ritz_evals) {
      f << e << "\n";
    }
  }
  {
    std::ofstream f(dir + "/meta.txt");
    f << "Lx=" << p.Lx << " Ly=" << p.Ly << " t=" << p.t << " tp=" << p.tp << " U=" << p.U << " V=" << p.V
      << " phi_x=" << p.phi_x << " phi_y=" << p.phi_y << "\n";
    f << "N_up=" << n_up << " N_dn=" << n_dn << " dim_dense=" << dense_evals.size()
      << " lanczos_krylov_dim=" << lanczos_used << " n_ritz=" << ritz_evals.size() << "\n";
  }
  std::cout << "[plot-data] wrote " << dir << "/{dense_evals.txt,lanczos_ritz.txt,meta.txt}\n";
}

}  // namespace

int main(int argc, char** argv) {
  const long long matvec_iters = parse_int_arg(argc, argv, "matvec-iters", 80000);
  const int lanczos_steps_req = parse_int_arg(argc, argv, "lanczos-steps", 36);
  const unsigned lanczos_seed = static_cast<unsigned>(parse_int_arg(argc, argv, "lanczos-seed", 7));
  const char* plot_dir_c = parse_string_arg(argc, argv, "write-plot-data");
  const std::string plot_dir = plot_dir_c ? std::string(plot_dir_c) : std::string();

  ftlm::HubbardParams p;
  p.Lx = 2;
  p.Ly = 2;
  p.t = 1.0;
  p.tp = -0.35;
  p.U = 5.75;
  p.V = 0.9;
  p.phi_x = 0.0;
  p.phi_y = 0.0;

  ftlm::RectLattice lat{p.Lx, p.Ly};
  std::vector<ftlm::SpinfulHopping> hops;
  std::vector<ftlm::NearestPair> pairs;
  ftlm::build_hubbard_geometry(p, lat, &hops, &pairs);

  const int n = ftlm::nsites(p);
  const int n_up = n / 2;
  const int n_dn = n - n_up;
  ftlm::FockBasis basis(n, n_up, n_dn);
  const int dim = basis.dim();
  const int lanczos_steps = std::min(lanczos_steps_req, dim);
  if (lanczos_steps_req > dim) {
    std::cerr << "bench_2x2: lanczos-steps=" << lanczos_steps_req << " capped to Hilbert dim=" << dim
              << " (no reorthogonalization)\n";
  }

  std::cout << "=== ftlm_extended_hubbard: 2×2 (4-site) PBC benchmark ===\n";
  std::cout << "t=" << p.t << " t'=" << p.tp << " U=" << p.U << " V=" << p.V << "  sector N_up=" << n_up
            << " N_dn=" << n_dn << "  dim=" << dim << "\n";

  std::vector<std::vector<double>> Hdense;
  auto t0 = clock::now();
  build_dense_real(p, basis, hops, pairs, &Hdense);
  const double t_dense = seconds_since(t0);
  const auto mm = jacobi_min_max(Hdense);
  const double e0_exact = mm.first;
  const double e1_exact = mm.second;
  std::cout << "[reference] dense build " << (t_dense * 1e3) << " ms  jacobi_sweeps=" << kDenseJacobiSweeps
            << "  E_min=" << e0_exact << "  E_max=" << e1_exact << "\n";

  std::vector<double> dense_evals_full;
  if (!plot_dir.empty()) {
    dense_evals_full = sorted_eigenvalues_jacobi(Hdense, 8000, 1e-12);
    std::cout << "[plot-data] dense spectrum: " << dense_evals_full.size()
              << " eigenvalues (full Jacobi)\n";
  }

  // Consistency: ||H_dense v - apply_H v|| should be ~0.
  {
    std::mt19937 rng2(99);
    std::normal_distribution<double> g2(0.0, 1.0);
    std::vector<std::complex<double>> x(static_cast<size_t>(dim)), ha(static_cast<size_t>(dim)),
        hd(static_cast<size_t>(dim));
    for (int i = 0; i < dim; ++i) {
      x[static_cast<size_t>(i)] = {g2(rng2), g2(rng2)};
    }
    ftlm::apply_extended_hubbard(p, basis, hops, pairs, x.data(), ha.data());
    for (int i = 0; i < dim; ++i) {
      std::complex<double> s(0.0, 0.0);
      for (int j = 0; j < dim; ++j) {
        s += Hdense[static_cast<size_t>(i)][static_cast<size_t>(j)] * x[static_cast<size_t>(j)];
      }
      hd[static_cast<size_t>(i)] = s;
    }
    double err = 0.0;
    for (int i = 0; i < dim; ++i) {
      err = std::max(err, std::abs(hd[static_cast<size_t>(i)] - ha[static_cast<size_t>(i)]));
    }
    std::cout << "[check] max|H_dense v - apply_H v| = " << err << "\n";
    if (err > 1e-10) {
      std::cerr << "bench_2x2: dense vs apply mismatch\n";
      return 3;
    }
  }

  std::vector<std::complex<double>> v(static_cast<size_t>(dim)), w(static_cast<size_t>(dim));
  std::mt19937 rng(123);
  std::normal_distribution<double> gauss(0.0, 1.0);
  for (int i = 0; i < dim; ++i) {
    v[static_cast<size_t>(i)] = {gauss(rng), gauss(rng)};
  }
  double nv = 0.0;
  for (int i = 0; i < dim; ++i) {
    nv += std::norm(v[static_cast<size_t>(i)]);
  }
  nv = std::sqrt(nv);
  for (int i = 0; i < dim; ++i) {
    v[static_cast<size_t>(i)] /= nv;
  }

  const int warmup = std::max(200, dim * 4);
  for (int i = 0; i < warmup; ++i) {
    ftlm::apply_extended_hubbard(p, basis, hops, pairs, v.data(), w.data());
    v.swap(w);
  }

  if (matvec_iters > 0) {
    t0 = clock::now();
    for (long long k = 0; k < matvec_iters; ++k) {
      ftlm::apply_extended_hubbard(p, basis, hops, pairs, v.data(), w.data());
      v.swap(w);
    }
    const double t_mat = seconds_since(t0);
    const double rate = static_cast<double>(matvec_iters) / t_mat;
    std::cout << "[bench] apply_H  iters=" << matvec_iters << "  time=" << (t_mat * 1e3) << " ms  throughput="
              << (rate * 1e-6) << " Mapply/s\n";
  } else {
    std::cout << "[bench] apply_H  skipped (matvec-iters=0)\n";
  }

  auto apply = [&](const std::complex<double>* x, std::complex<double>* y) {
    ftlm::apply_extended_hubbard(p, basis, hops, pairs, x, y);
  };
  std::vector<double> alpha;
  std::vector<double> beta;
  int lz_used = 0;
  t0 = clock::now();
  lanczos_tridiagonal(apply, dim, lanczos_steps, lanczos_seed, &alpha, &beta, &lz_used);
  const double t_lz = seconds_since(t0);
  std::vector<double> beta_trim;
  if (lz_used >= 2) {
    beta_trim.assign(beta.begin(), beta.begin() + (lz_used - 1));
  }
  const auto lz_mm = ftlm::tridiagonal_extrema(alpha, beta_trim, lz_used);
  std::cout << "[bench] lanczos  steps=" << lanczos_steps << "  used=" << lz_used << "  time=" << (t_lz * 1e3)
            << " ms  E_min≈" << lz_mm.first << "  E_max≈" << lz_mm.second
            << "  Δ_min=E_min(lz)-E_min(dense)=" << (lz_mm.first - e0_exact) << "  (no reorthogonalization)\n";

  if (!plot_dir.empty()) {
    const std::vector<double> ritz_sorted = sorted_ritz_from_tridiagonal(alpha, beta, lz_used);
    if (dense_evals_full.empty()) {
      dense_evals_full = sorted_eigenvalues_jacobi(Hdense, 8000, 1e-12);
    }
    write_plot_data(plot_dir, dense_evals_full, ritz_sorted, p, n_up, n_dn, lz_used);
  }

  return 0;
}
