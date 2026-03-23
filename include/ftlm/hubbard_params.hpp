#pragma once

namespace ftlm {

/// Physical and lattice parameters for the extended Hubbard model on a rectangle.
struct HubbardParams {
  int Lx = 2;
  int Ly = 2;
  double t = 1.0;
  double tp = 0.0;
  double U = 0.0;
  double V = 0.0;
  double phi_x = 0.0;
  double phi_y = 0.0;
};

int nsites(const HubbardParams& p);

}  // namespace ftlm
