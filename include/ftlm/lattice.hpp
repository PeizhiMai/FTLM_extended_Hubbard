#pragma once

#include <complex>
#include <vector>

#include "ftlm/hubbard_params.hpp"

namespace ftlm {

/// Directed kinetic term: H += coeff * c^\dagger_{to} c_{from} (spin index separate).
struct SpinfulHopping {
  int from = 0;
  int to = 0;
  int spin = 0;  ///< 0 = up, 1 = down
  std::complex<double> coeff{0.0, 0.0};
};

/// Nearest-neighbor pair (unordered) for the V interaction: n_i n_j with i < j.
struct NearestPair {
  int i = 0;
  int j = 0;
};

/// Square/rectangular lattice with periodic boundaries and Peierls phases on wraps.
struct RectLattice {
  int Lx = 1;
  int Ly = 1;

  int n_sites() const { return Lx * Ly; }
  int site_index(int x, int y) const { return y * Lx + x; }
  void coords(int site, int* x, int* y) const {
    *x = site % Lx;
    *y = site / Lx;
  }
};

/// Build all directed spinful hoppings and NN pairs for the extended Hubbard kinetic + V graph.
void build_hubbard_geometry(const HubbardParams& p, const RectLattice& lat,
                            std::vector<SpinfulHopping>* hoppings,
                            std::vector<NearestPair>* nn_pairs);

}  // namespace ftlm
