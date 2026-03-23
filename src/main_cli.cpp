#include <iostream>

#include "ftlm/fock_basis.hpp"
#include "ftlm/hubbard_params.hpp"
#include "ftlm/version.hpp"

int main() {
  ftlm::HubbardParams p;
  std::cout << "ftlm_extended_hubbard " << ftlm::version_string() << "\n";
  std::cout << "Lx=" << p.Lx << " Ly=" << p.Ly << " sites=" << ftlm::nsites(p) << "\n";
  const int n = ftlm::nsites(p);
  const int n_up = n / 2;
  const int n_dn = n - n_up;
  ftlm::FockBasis basis(n, n_up, n_dn);
  std::cout << "example sector N_up=" << n_up << " N_down=" << n_dn << " dim=" << basis.dim() << "\n";
  return 0;
}
