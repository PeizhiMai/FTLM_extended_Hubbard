#pragma once

#include <complex>

#include "ftlm/fock_basis.hpp"
#include "ftlm/hubbard_params.hpp"
#include "ftlm/lattice.hpp"

namespace ftlm {

/// y += H x for the extended Hubbard Hamiltonian in a single (N_up, N_down) sector.
/// `hoppings` / `nn_pairs` should come from `build_hubbard_geometry`.
void apply_extended_hubbard(const HubbardParams& p, const FockBasis& basis,
                            const std::vector<SpinfulHopping>& hoppings,
                            const std::vector<NearestPair>& nn_pairs,
                            const std::complex<double>* x, std::complex<double>* y);

}  // namespace ftlm
