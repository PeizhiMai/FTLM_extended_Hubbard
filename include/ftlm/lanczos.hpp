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

/// Optional storage for **all** Lanczos directions \(v_0,\ldots,v_{m-1}\) (column \(j\) = \(v_j\), length `dim`),
/// layout `cols[j*dim + i]`). **Memory:** \(O(\texttt{dim}\times\texttt{max\_steps})\) complex; use for small blocks,
/// debugging, or observables that need the Krylov basis (cf. Quantum Basis `iram` layout, Fortran full `phia`).
/// Filled by `lanczos_tridiagonal` when non-null; `steps_used` is set to the Krylov dimension returned.
struct LanczosFullBasisBuffer {
  int dim = 0;
  int steps_used = 0;
  std::vector<std::complex<double>> cols{};

  void ensure(int dim_in, int max_steps) {
    dim = dim_in;
    steps_used = 0;
    cols.assign(static_cast<std::size_t>(dim) * static_cast<std::size_t>(max_steps), std::complex<double>(0.0, 0.0));
  }

  const std::complex<double>* column(int j) const {
    return cols.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(dim);
  }

  std::complex<double>* column(int j) {
    return cols.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(dim);
  }

  std::size_t bytes_capacity() const {
    return cols.capacity() * sizeof(std::complex<double>);
  }

  void shrink_to_fit() { cols.shrink_to_fit(); }
};

/// Pack Hermitian tridiagonal Lanczos coefficients into one array:
/// \([α_0, β_0, α_1, β_1, \ldots, α_{n-1}]\) with `beta.size() == n - 1`.
void lanczos_tridiagonal_pack_coeffs(const std::vector<double>& alpha, const std::vector<double>& beta,
                                     std::vector<double>* packed);

/// Inverse of `lanczos_tridiagonal_pack_coeffs`. Returns false if `packed` length is not \(2n-1\) for some \(n\ge1\).
bool lanczos_tridiagonal_unpack_coeffs(const std::vector<double>& packed, std::vector<double>* alpha,
                                       std::vector<double>* beta);

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
///
/// If `full_basis != nullptr`, it is resized with `ensure(dim, max_steps)`; on return, `full_basis->steps_used`
/// equals the return value and column `j` is \(v_j\) (orthonormal Lanczos basis).
int lanczos_tridiagonal(
    int dim,
    const std::function<void(const std::complex<double>* v, std::complex<double>* Hv)>& apply_h,
    int max_steps, unsigned seed, std::vector<double>* alpha, std::vector<double>* beta,
    LanczosComplexWorkspace* ws = nullptr, LanczosFullBasisBuffer* full_basis = nullptr);

/// Extrema of a real symmetric tridiagonal matrix (main diag `alpha`, off-diag `beta`, length steps).
std::pair<double, double> tridiagonal_extrema(const std::vector<double>& alpha,
                                              const std::vector<double>& beta, int steps);

}  // namespace ftlm
