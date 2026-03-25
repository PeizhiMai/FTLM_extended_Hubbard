#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "ftlm/fock_basis.hpp"
#include "ftlm/hubbard_params.hpp"
#include "ftlm/symmetry/k_basis.hpp"
#include "ftlm/lattice.hpp"
#include "ftlm/symmetry/momentum_sector.hpp"
#include "ftlm/symmetry/orbit_bloch_phi.hpp"

namespace ftlm {
namespace symmetry {

/// Matrix-free extended Hubbard action on a **translation-symmetry-reduced** momentum block.
///
/// The `universe` used to build `orbit_map` must be **closed under hopping** (same
/// \(N_\uparrow,N_\downarrow\) sector and all configurations connected by `hoppings`), otherwise
/// off-diagonal pushes outside `basis` are dropped and the result is wrong.
///
/// Builds \(\Phi\) from translation-orbit Bloch sums (`momentum_phi_seeds`: one column per orbit member, fermionic
/// \(U(R)\)), then Gram whitening on the small Gram matrix. No dense \(P_k\) or \(d_{\mathrm{full}}\times d_{\mathrm{full}}\)
/// translation matrices.
struct HubbardMomentumAction {
  HubbardParams params{};
  RectLattice lat{};
  std::vector<SpinfulHopping> hoppings{};
  std::vector<NearestPair> nn_pairs{};

  /// Builds NN (+ optional NNN / Peierls) lists via `build_hubbard_geometry`.
  explicit HubbardMomentumAction(HubbardParams p);

  /// `y += H x` in the momentum-block ordering for `K`. `x`/`y` length must be `momentum_block_dim(...)`.
  void apply(const MomentumSectorMap& orbit_map, MomentumSector K, int n_up, int n_dn, const std::complex<double>* x,
             std::complex<double>* y) const;

  /// Dimension of the momentum block after orbit Bloch construction + Gram–Schmidt (dependent columns dropped).
  /// Matches `HubbardMomentumBlock::dim()`; can be less than `KBasis::dim()` if some Bloch columns were linearly dependent.
  /// Runs the same Φ build as the block (use sparingly outside setup; prefer constructing a block once).
  std::size_t momentum_block_dim(const MomentumSectorMap& orbit_map, MomentumSector K, int n_up, int n_dn) const;
};

/// Alias matching the project’s “matrix-free Hamiltonian action” naming.
using HubbardHamiltonianAction = HubbardMomentumAction;

/// Reusable full-space scratch for `HubbardMomentumBlock::apply` (lift / image of \(H\)). One instance can be shared
/// across consecutive blocks in the same thread so `vin`/`wout` are not duplicated inside each block object.
struct MomentumBlockScratch {
  std::vector<std::complex<double>> vin{};
  std::vector<std::complex<double>> wout{};
  MomentumPhiGramApplyScratch gram_apply{};
  void ensure_d_full(int d) {
    const auto u = static_cast<std::size_t>(d);
    vin.resize(u);
    wout.resize(u);
  }
  /// Drop peak allocations between particle sectors / long runs (k-bench passes one scratch per \((n_\up,n_\down)\)).
  void shrink_after_sector() {
    vin.clear();
    vin.shrink_to_fit();
    wout.clear();
    wout.shrink_to_fit();
    gram_apply.shrink_to_fit();
  }
};

/// Cached momentum block: builds **matrix-free** Gram / eigen data for the same orthonormal \(\Phi\) as the legacy
/// dense construction, then `apply` is lift → `apply_extended_hubbard` → project (no stored \(d_{\mathrm{full}}\times d_k\) \(\Phi\)).
///
/// `orbit_map` must outlive the block (same pointer as used at construction).
struct HubbardMomentumBlock {
  HubbardMomentumBlock(const HubbardMomentumAction& hub, const MomentumSectorMap& orbit_map, MomentumSector K,
                       int n_up, int n_dn, MomentumBlockScratch* scratch = nullptr);

  ~HubbardMomentumBlock();

  HubbardMomentumBlock(const HubbardMomentumBlock&) = delete;
  HubbardMomentumBlock& operator=(const HubbardMomentumBlock&) = delete;
  HubbardMomentumBlock(HubbardMomentumBlock&&) = default;
  HubbardMomentumBlock& operator=(HubbardMomentumBlock&&) = default;

  std::size_t dim() const noexcept { return dk_; }

  void apply(const std::complex<double>* x, std::complex<double>* y) const;

  /// Persistent storage for Gram eigenvectors + seeds (no dense \(\Phi\)); bytes of complex `V` etc.
  std::size_t phi_bytes() const noexcept { return gram_.storage_bytes(); }

  int d_full() const noexcept { return d_full_; }

 private:
  const HubbardMomentumAction* hub_ = nullptr;
  const MomentumSectorMap* orbit_map_ = nullptr;
  MomentumSector K_{};
  MomentumBlockScratch* scratch_ = nullptr;
  FockBasis fb_;
  int d_full_ = 0;
  std::size_t dk_ = 0;
  MomentumPhiGramBasis gram_{};
};

}  // namespace symmetry
}  // namespace ftlm
