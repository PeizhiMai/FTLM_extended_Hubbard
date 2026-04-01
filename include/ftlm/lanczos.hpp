#pragma once

#include <complex>
#include <cstddef>
#include <functional>
#include <vector>

namespace ftlm {

/// Reusable buffers for `lanczos_tridiagonal` (three-term recurrence: only q, q_prev, w — no full Krylov basis).
/// Contrast: Fortran FTLM reference codes often store all Lanczos columns (e.g. `phia(np,1:L)`); see
/// `docs/FTLM_HUB_COND_LANCZOS_REFERENCE.md`.
/// Pass through `FtlmParams::lanczos_ws` to reuse across multiple K-blocks in the same sector without reallocating
/// O(dim) complex vectors per FTLM call.
struct LanczosComplexWorkspace {
  std::vector<std::complex<double>> q{};
  std::vector<std::complex<double>> q_prev{};
  std::vector<std::complex<double>> w{};

  void ensure(int dim) {
    const auto d = static_cast<std::size_t>(dim);
    q.resize(d);
    q_prev.assign(d, std::complex<double>(0.0, 0.0));
    w.resize(d);
  }

  std::size_t bytes_capacity() const {
    return (q.capacity() + q_prev.capacity() + w.capacity()) * sizeof(std::complex<double>);
  }

  /// Return excess capacity to the allocator (e.g. after a sector’s largest k-block was processed).
  void shrink_to_fit() {
    q.shrink_to_fit();
    q_prev.shrink_to_fit();
    w.shrink_to_fit();
  }
};

/// Smallest and largest eigenvalues of a Hermitian operator given only `apply(v, out)` with out = H v.
/// Uses Hermitian Lanczos without full reorthogonalization (smoke / moderate step counts).
struct LanczosExtrema {
  double min_eval = 0.0;
  double max_eval = 0.0;
  int steps_used = 0;
};

LanczosExtrema lanczos_extrema(
    int dim,
    const std::function<void(const std::complex<double>* v, std::complex<double>* Hv)>& apply_h,
    int max_steps, unsigned seed = 1);

/// Hermitian Lanczos without reorthogonalization: fills `alpha` (length = steps used) and `beta`
/// (length = steps used − 1, subdiagonal). Returns Krylov dimension used (0 on failure).
///
/// If `ws != nullptr`, uses `ws` vectors (caller must `ensure(dim)` or rely on this function resizing).
/// If `ws == nullptr`, uses a thread_local buffer resized to `dim` (reuse across calls, capacity grows to max dim).
int lanczos_tridiagonal(
    int dim,
    const std::function<void(const std::complex<double>* v, std::complex<double>* Hv)>& apply_h,
    int max_steps, unsigned seed, std::vector<double>* alpha, std::vector<double>* beta,
    LanczosComplexWorkspace* ws = nullptr);

/// Extrema of a real symmetric tridiagonal matrix (main diag `alpha`, off-diag `beta`, length steps).
std::pair<double, double> tridiagonal_extrema(const std::vector<double>& alpha,
                                              const std::vector<double>& beta, int steps);

}  // namespace ftlm
