#include "ftlm/hubbard_params.hpp"

namespace ftlm {

int nsites(const HubbardParams& p) { return p.Lx * p.Ly; }

}  // namespace ftlm
