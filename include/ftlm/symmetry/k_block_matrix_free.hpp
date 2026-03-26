#pragma once

#include <complex>
#include <cstddef>

namespace ftlm {
namespace symmetry {

/// Prototype hooks for a **lower-memory** momentum k-block path that avoids the current
/// dense Gram build + full `zheev` on `k_in × k_in` followed by storing a large `V` slice.
///
/// **Production path (unchanged):** `HubbardMomentumBlock` + `MomentumPhiGramBasis` (Gram + eig + packed `V`).
///
/// Future implementations may satisfy `KBlockHamiltonianApply` without materializing that pipeline.
namespace k_block_matrix_free {

/// Abstract matrix-free operator on the k-block Hilbert space: \(y = H_K x\) with
/// \(H_K = P_K H P_K\) in the chosen block representation (orthonormal basis or equivalent).
///
/// Memory target: \(O(d_K)\) workspace for a few vectors plus sparse/hopping structure, not \(O(k_{\mathrm{in}}^2)\)
/// dense Gram storage.
struct KBlockHamiltonianApply {
  virtual ~KBlockHamiltonianApply() = default;
  virtual int dim() const = 0;
  virtual void apply(const std::complex<double>* x, std::complex<double>* y) const = 0;
};

/// Thin adapter: exposes existing `HubbardMomentumBlock` through `KBlockHamiltonianApply` for
/// side-by-side tests and future replacement of the inner representation only.
template <typename Block>
class HubbardBlockAdapter final : public KBlockHamiltonianApply {
 public:
  explicit HubbardBlockAdapter(const Block* block) : block_(block) {}
  int dim() const override { return static_cast<int>(block_->dim()); }
  void apply(const std::complex<double>* x, std::complex<double>* y) const override { block_->apply(x, y); }

 private:
  const Block* block_ = nullptr;
};

}  // namespace k_block_matrix_free
}  // namespace symmetry
}  // namespace ftlm
