#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "ftlm/hubbard_params.hpp"
#include "ftlm/symmetry/k_basis.hpp"
#include "ftlm/lattice.hpp"
#include "ftlm/symmetry/momentum_sector.hpp"

namespace ftlm {
namespace symmetry {

/// Matrix-free extended Hubbard action on a **translation-symmetry-reduced** momentum block.
///
/// The `universe` used to build `orbit_map` must be **closed under hopping** (same
/// \(N_\uparrow,N_\downarrow\) sector and all configurations connected by `hoppings`), otherwise
/// off-diagonal pushes outside `basis` are dropped and the result is wrong.
///
/// For Hilbert-space dimension \(\le\) 512, builds \(\Phi\) as an orthonormal basis of \(\mathrm{range}(P_k)\) with
/// \(P_k=\frac{1}{|G|}\sum_R \chi_k(R)^\dagger U(R)\) (Hermitian-symmetrized) and \(U(R)\) the fermionic translation on
/// the Fock basis (JW mode order matching `apply_extended_hubbard`). The block dimension equals \(\mathrm{rank}(P_k)\)
/// from the spectral cut (not `KBasis::dim()`, which uses stabilizer-only masks). Larger sectors use a legacy
/// orbit–Bloch path (best-effort; may disagree with dense \(P_k\) where the mask overcounts).
struct HubbardMomentumAction {
  HubbardParams params{};
  RectLattice lat{};
  std::vector<SpinfulHopping> hoppings{};
  std::vector<NearestPair> nn_pairs{};

  /// Builds NN (+ optional NNN / Peierls) lists via `build_hubbard_geometry`.
  explicit HubbardMomentumAction(HubbardParams p);

  /// `y += H x` in the momentum-block ordering for `K`. `x`/`y` length must be `momentum_block_dim(...)`.
  /// Uses `n_up,n_dn` for the Fock sector (same as in `orbit_map`); `KBasis` is not used for dimension when
  /// `dim<=512` (rank of dense \(P_k\) can differ from stabilizer-based `KBasis::dim()`).
  void apply(const MomentumSectorMap& orbit_map, MomentumSector K, int n_up, int n_dn, const std::complex<double>* x,
             std::complex<double>* y) const;

  /// Dimension of the momentum block: \(\mathrm{rank}(P_k)\) from dense spectral decomposition if `dim<=512`, else
  /// legacy orbit–Bloch column count (`KBasis::dim()`).
  std::size_t momentum_block_dim(const MomentumSectorMap& orbit_map, MomentumSector K, int n_up, int n_dn) const;
};

/// Alias matching the project’s “matrix-free Hamiltonian action” naming.
using HubbardHamiltonianAction = HubbardMomentumAction;

}  // namespace symmetry
}  // namespace ftlm
