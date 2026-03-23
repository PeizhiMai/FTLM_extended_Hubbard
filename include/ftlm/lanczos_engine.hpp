#pragma once

#include <complex>
#include <functional>
#include <vector>

#include "ftlm/lanczos.hpp"

namespace ftlm {

/// Matrix-free Hermitian Lanczos façade: keeps **three** workspace vectors (`v_cur`, `v_prev`, `work`)
/// sized to the Hilbert dimension for upcoming FTLM / in-place loops. `run_extrema` currently delegates
/// to `lanczos_extrema` (which allocates its own scratch); a later refactor can thread these buffers
/// through to avoid duplicate allocation.
struct LanczosEngine {
  using ApplyH = std::function<void(const std::complex<double>*, std::complex<double>*)>;

  int dim = 0;
  ApplyH apply_h{};
  std::vector<std::complex<double>> v_cur{};
  std::vector<std::complex<double>> v_prev{};
  std::vector<std::complex<double>> work{};

  LanczosEngine() = default;

  LanczosEngine(int d, ApplyH fn) : dim(d), apply_h(std::move(fn)) { resize_buffers(d); }

  void resize_buffers(int d) {
    dim = d;
    v_cur.assign(static_cast<std::size_t>(d), std::complex<double>(0.0, 0.0));
    v_prev.assign(static_cast<std::size_t>(d), std::complex<double>(0.0, 0.0));
    work.assign(static_cast<std::size_t>(d), std::complex<double>(0.0, 0.0));
  }

  /// Smallest / largest Ritz values from the Krylov tridiagonal (same semantics as `lanczos_extrema`).
  LanczosExtrema run_extrema(int max_steps, unsigned seed = 1) const {
    return lanczos_extrema(dim, apply_h, max_steps, seed);
  }
};

}  // namespace ftlm
