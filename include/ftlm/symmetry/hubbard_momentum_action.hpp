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
/// Requires `p.Lx == p.Ly == 4` to match `Lattice4x4` / Z₄×Z₄ projection. The `universe` used to
/// build `orbit_map` must be **closed under hopping** (same \(N_\uparrow,N_\downarrow\) sector and
/// all configurations connected by `hoppings`), otherwise off-diagonal pushes outside `basis` are
/// dropped and the result is wrong.
///
/// Basis vectors are orthonormal \(|k,\alpha\rangle\) built from canonical representatives; hopping
/// uses the same Jordan–Wigner string as `apply_extended_hubbard` (mode order \(2\cdot\mathrm{site}+\sigma\)).
struct HubbardMomentumAction {
  HubbardParams params{};
  RectLattice lat{};
  std::vector<SpinfulHopping> hoppings{};
  std::vector<NearestPair> nn_pairs{};

  /// Builds NN (+ optional NNN / Peierls) lists via `build_hubbard_geometry`.
  explicit HubbardMomentumAction(HubbardParams p);

  /// `y += H x` in the `basis` ordering for momentum `K` (uses `orbit_map` for stabilizer sizes).
  void apply(const MomentumSectorMap& orbit_map, MomentumSector K, const KBasis& basis,
             const std::complex<double>* x, std::complex<double>* y) const;
};

/// Alias matching the project’s “matrix-free Hamiltonian action” naming.
using HubbardHamiltonianAction = HubbardMomentumAction;

}  // namespace symmetry
}  // namespace ftlm
