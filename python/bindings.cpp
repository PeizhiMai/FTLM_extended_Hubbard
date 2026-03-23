#include <pybind11/pybind11.h>

#include "ftlm/hubbard_params.hpp"
#include "ftlm/version.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_ftlm_core, m) {
  m.doc() = "ftlm_extended_hubbard C++ core (minimal bindings)";

  py::class_<ftlm::HubbardParams>(m, "HubbardParams")
      .def(py::init<>())
      .def(py::init([](int Lx, int Ly, double t, double tp, double U, double V, double phi_x,
                      double phi_y) {
             ftlm::HubbardParams p;
             p.Lx = Lx;
             p.Ly = Ly;
             p.t = t;
             p.tp = tp;
             p.U = U;
             p.V = V;
             p.phi_x = phi_x;
             p.phi_y = phi_y;
             return p;
           }),
           py::arg("Lx") = 2, py::arg("Ly") = 2, py::arg("t") = 1.0, py::arg("tp") = 0.0,
           py::arg("U") = 0.0, py::arg("V") = 0.0, py::arg("phi_x") = 0.0, py::arg("phi_y") = 0.0)
      .def_readwrite("Lx", &ftlm::HubbardParams::Lx)
      .def_readwrite("Ly", &ftlm::HubbardParams::Ly)
      .def_readwrite("t", &ftlm::HubbardParams::t)
      .def_readwrite("tp", &ftlm::HubbardParams::tp)
      .def_readwrite("U", &ftlm::HubbardParams::U)
      .def_readwrite("V", &ftlm::HubbardParams::V)
      .def_readwrite("phi_x", &ftlm::HubbardParams::phi_x)
      .def_readwrite("phi_y", &ftlm::HubbardParams::phi_y);

  m.def("version_string", &ftlm::version_string);
  m.def("nsites", &ftlm::nsites, py::arg("params"));
}
