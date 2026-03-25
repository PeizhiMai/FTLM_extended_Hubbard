// 2x2: compare sorted eigenvalues of full sector H vs union of momentum-block spectra for every (N_up,N_dn).
// Writes TSV (pairwise + per-k) and prints a short summary to stdout.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "ftlm/fock_basis.hpp"
#include "ftlm/hubbard_hamiltonian.hpp"
#include "ftlm/hubbard_params.hpp"
#include "ftlm/lattice.hpp"
#include "ftlm/symmetry/hubbard_momentum_action.hpp"
#include "ftlm/symmetry/k_basis.hpp"
#include "ftlm/symmetry/momentum_sector.hpp"
#include "ftlm/symmetry/raw_state.hpp"

#if defined(__APPLE__)
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK 1
#endif
#include <Accelerate/Accelerate.h>
#else
extern "C" {
void zheev_(char* jobz, char* uplo, int* n, std::complex<double>* a, int* lda, double* w,
            std::complex<double>* work, int* lwork, double* rwork, int* info);
}
#endif

namespace {

constexpr int Lx = 2;
constexpr int Ly = 2;
constexpr int Ns = 4;
constexpr int Nmax = 4;

std::vector<std::vector<std::complex<double>>> build_full_H_dense(const ftlm::HubbardParams& p, const ftlm::FockBasis& fb,
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
    for (int i = 0; i < dim; ++i) {
      H[static_cast<size_t>(i)][static_cast<size_t>(j)] = y[static_cast<size_t>(i)];
    }
  }
  return H;
}

std::vector<double> zheev_evals_only(const std::vector<std::vector<std::complex<double>>>& H) {
  const int n = static_cast<int>(H.size());
  if (n == 0) {
    return {};
  }
#if defined(__APPLE__)
  std::vector<std::complex<double>> a(static_cast<size_t>(n) * static_cast<size_t>(n));
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      a[static_cast<size_t>(j * n + i)] = H[static_cast<size_t>(i)][static_cast<size_t>(j)];
    }
  }
  char jobz = 'N';
  char uplo = 'U';
  __LAPACK_int nn = static_cast<__LAPACK_int>(n);
  __LAPACK_int lda = static_cast<__LAPACK_int>(n);
  std::vector<double> w(static_cast<size_t>(n));
  std::vector<std::complex<double>> work(1);
  __LAPACK_int lwork = -1;
  std::vector<double> rwork(static_cast<size_t>(std::max(1, 3 * n - 2)));
  __LAPACK_int info = 0;
  zheev_(&jobz, &uplo, &nn, reinterpret_cast<__LAPACK_double_complex*>(a.data()), &lda, w.data(),
         reinterpret_cast<__LAPACK_double_complex*>(work.data()), &lwork, rwork.data(), &info);
  if (info != 0) {
    throw std::runtime_error("zheev workspace query failed (info=" + std::to_string(static_cast<int>(info)) + ")");
  }
  lwork = static_cast<__LAPACK_int>(std::llround(work[0].real()));
  if (lwork < 1) {
    lwork = 1;
  }
  work.resize(static_cast<size_t>(lwork));
  zheev_(&jobz, &uplo, &nn, reinterpret_cast<__LAPACK_double_complex*>(a.data()), &lda, w.data(),
         reinterpret_cast<__LAPACK_double_complex*>(work.data()), &lwork, rwork.data(), &info);
  if (info != 0) {
    throw std::runtime_error("zheev failed (info=" + std::to_string(static_cast<int>(info)) + ")");
  }
  return w;
#else
  std::vector<std::complex<double>> a(static_cast<size_t>(n) * static_cast<size_t>(n));
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      a[static_cast<size_t>(j * n + i)] = H[static_cast<size_t>(i)][static_cast<size_t>(j)];
    }
  }
  char jobz = 'N';
  char uplo = 'U';
  int nn = n;
  int lda = n;
  std::vector<double> w(static_cast<size_t>(n));
  std::vector<std::complex<double>> work(1);
  int lwork = -1;
  std::vector<double> rwork(static_cast<size_t>(std::max(1, 3 * n - 2)));
  int info = 0;
  zheev_(&jobz, &uplo, &nn, a.data(), &lda, w.data(), work.data(), &lwork, rwork.data(), &info);
  if (info != 0) {
    throw std::runtime_error("zheev workspace query failed (info=" + std::to_string(info) + ")");
  }
  lwork = static_cast<int>(std::llround(work[0].real()));
  if (lwork < 1) {
    lwork = 1;
  }
  work.resize(static_cast<size_t>(lwork));
  zheev_(&jobz, &uplo, &nn, a.data(), &lda, w.data(), work.data(), &lwork, rwork.data(), &info);
  if (info != 0) {
    throw std::runtime_error("zheev failed (info=" + std::to_string(info) + ")");
  }
  return w;
#endif
}

double max_abs_diff_sorted(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) {
    return std::numeric_limits<double>::infinity();
  }
  double m = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    m = std::max(m, std::abs(a[i] - b[i]));
  }
  return m;
}

struct KBlockEvals {
  int kx{};
  int ky{};
  int dk{};
  std::vector<double> evals_asc{};
};

struct SectorEigs {
  int nup{};
  int ndn{};
  int dim{};
  int sum_dk{};
  double max_im_H{};
  double max_err{};
  double tol{};
  bool ok_dim{};
  bool ok_eigs{};
  std::vector<double> eval_full_sorted{};
  std::vector<double> eval_k_merged_sorted{};
  std::vector<KBlockEvals> k_blocks{};
};

SectorEigs compute_sector(const ftlm::HubbardParams& p, const std::vector<ftlm::SpinfulHopping>& hops,
                          const std::vector<ftlm::NearestPair>& pairs, int nup, int ndn,
                          ftlm::symmetry::HubbardMomentumAction& hub) {
  SectorEigs out;
  out.nup = nup;
  out.ndn = ndn;
  ftlm::FockBasis fb(Ns, nup, ndn);
  out.dim = fb.dim();

  std::vector<ftlm::symmetry::RawState> universe;
  universe.reserve(static_cast<size_t>(out.dim));
  for (int i = 0; i < out.dim; ++i) {
    universe.push_back(ftlm::symmetry::RawState{static_cast<std::uint16_t>(fb.up_mask(i)),
                                                static_cast<std::uint16_t>(fb.down_mask(i))});
  }
  const auto map = ftlm::symmetry::build_momentum_sector_map_rect(std::move(universe), Lx, Ly);

  const auto H_full = build_full_H_dense(p, fb, hops, pairs);
  for (int i = 0; i < out.dim; ++i) {
    for (int j = 0; j < out.dim; ++j) {
      out.max_im_H = std::max(out.max_im_H, std::abs(H_full[static_cast<size_t>(i)][static_cast<size_t>(j)].imag()));
    }
  }

  out.eval_full_sorted = zheev_evals_only(H_full);
  std::sort(out.eval_full_sorted.begin(), out.eval_full_sorted.end());

  std::vector<double> eval_k_all;
  eval_k_all.reserve(static_cast<size_t>(out.dim));

  ftlm::symmetry::MomentumBlockScratch k_scratch;
  for (int ky = 0; ky < Ly; ++ky) {
    for (int kx = 0; kx < Lx; ++kx) {
      const ftlm::symmetry::MomentumSector K{kx, ky, Lx, Ly};
      const auto kb = ftlm::symmetry::KBasis::build(map, K);
      ftlm::symmetry::HubbardMomentumBlock block(hub, map, K, nup, ndn, &k_scratch);
      const int dk = static_cast<int>(block.dim());
      if (dk <= 0) {
        continue;
      }
      out.sum_dk += dk;

      std::vector<std::vector<std::complex<double>>> Hk(static_cast<size_t>(dk),
                                                         std::vector<std::complex<double>>(static_cast<size_t>(dk), 0.0));
      std::vector<std::complex<double>> x(static_cast<size_t>(dk), 0.0), y(static_cast<size_t>(dk), 0.0);
      for (int j = 0; j < dk; ++j) {
        std::fill(x.begin(), x.end(), std::complex<double>(0.0, 0.0));
        x[static_cast<size_t>(j)] = {1.0, 0.0};
        block.apply(x.data(), y.data());
        for (int i = 0; i < dk; ++i) {
          Hk[static_cast<size_t>(i)][static_cast<size_t>(j)] = y[static_cast<size_t>(i)];
        }
      }

      std::vector<double> w = zheev_evals_only(Hk);
      KBlockEvals blk;
      blk.kx = kx;
      blk.ky = ky;
      blk.dk = dk;
      blk.evals_asc = std::move(w);
      for (double e : blk.evals_asc) {
        eval_k_all.push_back(e);
      }
      out.k_blocks.push_back(std::move(blk));
    }
  }
  k_scratch.vin.clear();
  k_scratch.vin.shrink_to_fit();
  k_scratch.wout.clear();
  k_scratch.wout.shrink_to_fit();

  std::sort(eval_k_all.begin(), eval_k_all.end());
  out.eval_k_merged_sorted = std::move(eval_k_all);

  out.ok_dim = (out.sum_dk == out.dim);
  out.ok_eigs = (out.eval_full_sorted.size() == out.eval_k_merged_sorted.size());
  out.tol = 1e-8;
  if (!out.eval_full_sorted.empty()) {
    out.tol *= std::max(1.0, std::max(std::abs(out.eval_full_sorted.back()), std::abs(out.eval_full_sorted.front())));
  }
  out.max_err = max_abs_diff_sorted(out.eval_full_sorted, out.eval_k_merged_sorted);
  if (!out.ok_dim || !out.ok_eigs) {
    out.ok_eigs = false;
  } else if (out.max_err > out.tol) {
    out.ok_eigs = false;
  }

  return out;
}

void write_tsv(const std::string& path, const ftlm::HubbardParams& p,
               const std::vector<SectorEigs>& sectors) {
  namespace fs = std::filesystem;
  fs::path fp(path);
  if (fp.has_parent_path()) {
    fs::create_directories(fp.parent_path());
  }
  std::ofstream os(path);
  if (!os) {
    throw std::runtime_error("failed to open for write: " + path);
  }
  os << std::setprecision(17);
  os << "# 2x2 full vs K-block eigenvalues; sorted full vs sorted merged k-block union\n";
  os << "# K blocks: rank(P_k) spectral basis (same as HubbardMomentumAction for dim<=512)\n";
  os << "# Lx=" << Lx << " Ly=" << Ly << " t=" << p.t << " tp=" << p.tp << " U=" << p.U << " V=" << p.V << " phi_x=" << p.phi_x
     << " phi_y=" << p.phi_y << "\n";
  os << "\n";

  os << "# sector_summary\n";
  os << "n_up\tn_dn\tdim\tsum_k_dim_k\tmax_imag_H\tmax_abs_sorted_diff\ttol\tpass_dim\tpass_eigs\n";
  for (const auto& s : sectors) {
    os << s.nup << "\t" << s.ndn << "\t" << s.dim << "\t" << s.sum_dk << "\t" << s.max_im_H << "\t" << s.max_err << "\t" << s.tol
       << "\t" << (s.ok_dim ? 1 : 0) << "\t" << (s.ok_eigs ? 1 : 0) << "\n";
  }
  os << "\n";

  os << "# eigs_pairwise (same idx: full diagonalization vs merged K-block spectra, both sorted ascending)\n";
  os << "n_up\tn_dn\tdim\tidx\teval_full_sorted\teval_k_merged_sorted\tabsdiff\n";
  for (const auto& s : sectors) {
    const size_t n = s.eval_full_sorted.size();
    for (size_t i = 0; i < n; ++i) {
      const double ef = s.eval_full_sorted[i];
      const double ek = s.eval_k_merged_sorted[i];
      os << s.nup << "\t" << s.ndn << "\t" << s.dim << "\t" << i << "\t" << ef << "\t" << ek << "\t" << std::abs(ef - ek)
         << "\n";
    }
  }
  os << "\n";

  os << "# eigs_per_k (LAPACK ascending order within each momentum block)\n";
  os << "n_up\tn_dn\tk_x\tk_y\td_k\tidx_in_block\teval_k_block_ascending\n";
  for (const auto& s : sectors) {
    for (const auto& blk : s.k_blocks) {
      for (size_t i = 0; i < blk.evals_asc.size(); ++i) {
        os << s.nup << "\t" << s.ndn << "\t" << blk.kx << "\t" << blk.ky << "\t" << blk.dk << "\t" << i << "\t"
           << blk.evals_asc[i] << "\n";
      }
    }
  }
}

void print_usage(const char* prog) {
  std::cerr << "Usage: " << prog << " [-o|--output PATH.tsv] [--strict] [t] [tp] [U] [V]\n";
  std::cerr << "  Default output: benchmarks/data/eigs_full_vs_k_2x2.tsv (relative to cwd)\n";
  std::cerr << "  All particle sectors (n_up,n_dn) in [0,4] on 4 sites.\n";
  std::cerr << "  Exit 0 after writing TSV unless --strict and some sector fails eigenvalue match.\n";
}

}  // namespace

int main(int argc, char** argv) {
  ftlm::HubbardParams p;
  p.Lx = Lx;
  p.Ly = Ly;
  p.t = 1.0;
  p.tp = -0.35;
  p.U = 5.75;
  p.V = 0.9;
  p.phi_x = 0.0;
  p.phi_y = 0.0;

  std::string out_path = "benchmarks/data/eigs_full_vs_k_2x2.tsv";
  bool strict = false;
  std::vector<double> pos;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      print_usage(argv[0]);
      return 0;
    }
    if (a == "--strict") {
      strict = true;
      continue;
    }
    if (a == "-o" || a == "--output") {
      if (i + 1 >= argc) {
        print_usage(argv[0]);
        return 2;
      }
      out_path = argv[++i];
      continue;
    }
    try {
      pos.push_back(std::stod(a));
    } catch (...) {
      std::cerr << "Bad argument: " << a << "\n";
      print_usage(argv[0]);
      return 2;
    }
  }
  if (pos.size() > 0) {
    p.t = pos[0];
  }
  if (pos.size() > 1) {
    p.tp = pos[1];
  }
  if (pos.size() > 2) {
    p.U = pos[2];
  }
  if (pos.size() > 3) {
    p.V = pos[3];
  }

  ftlm::RectLattice lat{Lx, Ly};
  std::vector<ftlm::SpinfulHopping> hops;
  std::vector<ftlm::NearestPair> pairs;
  ftlm::build_hubbard_geometry(p, lat, &hops, &pairs);
  ftlm::symmetry::HubbardMomentumAction hub(p);

  std::vector<SectorEigs> sectors;
  sectors.reserve(static_cast<size_t>((Nmax + 1) * (Nmax + 1)));

  std::cout << "bench_2x2_eigs_full_vs_k: all sectors n_up,n_dn in [0," << Nmax << "]; HubbardMomentumAction (dense P_k)\n";
  std::cout << "params: t=" << p.t << " tp=" << p.tp << " U=" << p.U << " V=" << p.V << " phi=(" << p.phi_x << "," << p.phi_y
            << ")\n";
  std::cout << "TSV output: " << out_path << "\n";

  bool all_ok = true;
  for (int nup = 0; nup <= Nmax; ++nup) {
    for (int ndn = 0; ndn <= Nmax; ++ndn) {
      SectorEigs s = compute_sector(p, hops, pairs, nup, ndn, hub);
      sectors.push_back(std::move(s));
      const SectorEigs& r = sectors.back();
      std::cout << "  (" << r.nup << "," << r.ndn << ") dim=" << r.dim << " sum_dk=" << r.sum_dk << " max_err=" << r.max_err
                << " pass=" << (r.ok_dim && r.ok_eigs ? "yes" : "NO") << "\n";
      if (!r.ok_dim || !r.ok_eigs) {
        all_ok = false;
      }
    }
  }

  try {
    write_tsv(out_path, p, sectors);
  } catch (const std::exception& e) {
    std::cerr << "TSV write failed: " << e.what() << "\n";
    return 2;
  }

  if (!all_ok) {
    std::cerr << "Warning: one or more sectors failed full vs K-block eigenvalue match (see pass_eigs in TSV).\n";
    if (strict) {
      return 1;
    }
    return 0;
  }
  std::cout << "PASS (all sectors)\n";
  return 0;
}
