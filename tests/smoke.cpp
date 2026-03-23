#include <cassert>

#include "ftlm/hubbard_params.hpp"

int main() {
  ftlm::HubbardParams p;
  p.Lx = 4;
  p.Ly = 4;
  assert(ftlm::nsites(p) == 16);
  return 0;
}
