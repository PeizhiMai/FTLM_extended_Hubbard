#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "ftlm/symmetry/k_block_matrix_free.hpp"

namespace ftlm {
namespace symmetry {
namespace k_block_matrix_free {

/// Prototype: represent the **same** block Hamiltonian as `HubbardMomentumBlock::apply` by an explicit
/// dense matrix \(H_{\mathrm{block}}\in\mathbb{C}^{d_K\times d_K}\) with
/// \[
///   (H_{\mathrm{block}})_{ij} = e_i^\dagger H_{\mathrm{block}} e_j = (H_{\mathrm{block}} e_j)_i,
/// \]
/// i.e. column \(j\) is `apply(e_j)` in the **orthonormal momentum-block coordinates** from the current
/// Gram/whitening construction (same as production).
///
/// **Vector space:** `x` and `y` are coefficients in that **same** orthonormal block basis (length `d_K`).
///
/// **What this is *not*:** it is **not** a separate construction of \(P_K H P_K\) in the full Fock space
/// followed by a change of basis. Production `apply` implements \(y=\Phi^\dagger H(\Phi x)\) with the
/// whitened orbit–Bloch basis \(\Phi\); this dense form is **algebraically** \(H_{\mathrm{block}}=\Phi^\dagger H\Phi\)
/// in that basis (up to numerical rounding from how `apply` is evaluated).
///
/// **Memory:** \(O(d_K^2)\) complex — **only** for small validation (`d_K <= max_dim`).
///
/// **No dense Gram matrix** and **no zheev** are used here; the operator is **sampled** via the existing
/// `apply` (which internally uses the production Gram data).
class DenseBlockHamiltonianFromApply final : public KBlockHamiltonianApply {
 public:
  /// Builds dense columns by calling `apply(e_j)` for `j = 0 .. d_K-1`. `apply` must be the production
  /// momentum-block apply (same signature as `HubbardMomentumBlock::apply`).
  template <typename ApplyFn>
  DenseBlockHamiltonianFromApply(int dim_k, ApplyFn&& apply_one, int max_dim = 64) : dim_(dim_k) {
    if (dim_k <= 0) {
      throw std::invalid_argument("DenseBlockHamiltonianFromApply: dim_k must be positive");
    }
    if (dim_k > max_dim) {
      throw std::invalid_argument("DenseBlockHamiltonianFromApply: dim_k exceeds max_dim (prototype only)");
    }
    cols_.assign(static_cast<std::size_t>(dim_k) * static_cast<std::size_t>(dim_k), std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> ej(static_cast<std::size_t>(dim_k), std::complex<double>(0.0, 0.0));
    std::vector<std::complex<double>> col(static_cast<std::size_t>(dim_k), std::complex<double>(0.0, 0.0));
    for (int j = 0; j < dim_k; ++j) {
      std::fill(ej.begin(), ej.end(), std::complex<double>(0.0, 0.0));
      ej[static_cast<std::size_t>(j)] = std::complex<double>(1.0, 0.0);
      apply_one(ej.data(), col.data());
      for (int i = 0; i < dim_k; ++i) {
        cols_[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(dim_k)] =
            col[static_cast<std::size_t>(i)];
      }
    }
  }

  int dim() const override { return dim_; }

  void apply(const std::complex<double>* x, std::complex<double>* y) const override {
    for (int i = 0; i < dim_; ++i) {
      std::complex<double> s(0.0, 0.0);
      for (int j = 0; j < dim_; ++j) {
        s += cols_[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(dim_)] *
             x[static_cast<std::size_t>(j)];
      }
      y[static_cast<std::size_t>(i)] = s;
    }
  }

  /// Column-major dense matrix (same layout as LAPACK-style column j).
  const std::vector<std::complex<double>>& dense_cols() const { return cols_; }

 private:
  int dim_ = 0;
  std::vector<std::complex<double>> cols_;
};

}  // namespace k_block_matrix_free
}  // namespace symmetry
}  // namespace ftlm
