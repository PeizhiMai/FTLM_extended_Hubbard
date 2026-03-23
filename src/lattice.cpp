#include "ftlm/lattice.hpp"

#include <cmath>

namespace ftlm {

namespace detail {

std::complex<double> peierls(int wx, int wy, double phi_x, double phi_y) {
  return std::polar(1.0, static_cast<double>(wx) * phi_x + static_cast<double>(wy) * phi_y);
}

void add_directed(const RectLattice& lat, int from, int to, const std::complex<double>& phase,
                  double t_strength, std::vector<SpinfulHopping>* hoppings) {
  if (std::abs(t_strength) < 1e-15) {
    return;
  }
  const std::complex<double> c_ij = -t_strength * phase;
  const std::complex<double> c_ji = -t_strength * std::conj(phase);
  for (int spin : {0, 1}) {
    hoppings->push_back(SpinfulHopping{from, to, spin, c_ij});
    hoppings->push_back(SpinfulHopping{to, from, spin, c_ji});
  }
}

void append_nn_pair(std::vector<NearestPair>* pairs, int a, int b) {
  if (a == b) {
    return;
  }
  if (a > b) {
    std::swap(a, b);
  }
  pairs->push_back(NearestPair{a, b});
}

}  // namespace detail

void build_hubbard_geometry(const HubbardParams& p, const RectLattice& lat,
                            std::vector<SpinfulHopping>* hoppings,
                            std::vector<NearestPair>* nn_pairs) {
  hoppings->clear();
  nn_pairs->clear();
  const int Lx = lat.Lx;
  const int Ly = lat.Ly;

  for (int y = 0; y < Ly; ++y) {
    for (int x = 0; x < Lx; ++x) {
      const int i = lat.site_index(x, y);

      // East (skip if Lx==1 — periodic wrap would be an on-site loop)
      if (Lx > 1) {
        int wx = 0;
        int nx = x + 1;
        if (nx >= Lx) {
          nx = 0;
          wx = 1;
        }
        const int j = lat.site_index(nx, y);
        if (i != j) {
          detail::append_nn_pair(nn_pairs, i, j);
          detail::add_directed(lat, i, j, detail::peierls(wx, 0, p.phi_x, p.phi_y), p.t, hoppings);
        }
      }
      // North
      if (Ly > 1) {
        int wy = 0;
        int ny = y + 1;
        if (ny >= Ly) {
          ny = 0;
          wy = 1;
        }
        const int j = lat.site_index(x, ny);
        if (i != j) {
          detail::append_nn_pair(nn_pairs, i, j);
          detail::add_directed(lat, i, j, detail::peierls(0, wy, p.phi_x, p.phi_y), p.t, hoppings);
        }
      }

      // Next-nearest (diagonals)
      auto diag = [&](int dx, int dy) {
        int wx = 0;
        int wy = 0;
        int nx = x + dx;
        int ny = y + dy;
        if (nx >= Lx) {
          nx -= Lx;
          wx = 1;
        } else if (nx < 0) {
          nx += Lx;
          wx = -1;
        }
        if (ny >= Ly) {
          ny -= Ly;
          wy = 1;
        } else if (ny < 0) {
          ny += Ly;
          wy = -1;
        }
        const int j = lat.site_index(nx, ny);
        if (i != j) {
          detail::add_directed(lat, i, j, detail::peierls(wx, wy, p.phi_x, p.phi_y), p.tp, hoppings);
        }
      };
      diag(1, 1);
      diag(1, -1);
      diag(-1, 1);
      diag(-1, -1);
    }
  }
}

}  // namespace ftlm
