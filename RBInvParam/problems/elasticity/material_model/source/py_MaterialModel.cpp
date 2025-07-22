#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream>

#include "MaterialModel.hpp"

// -------- PYTHON BINDINGS -----------------------------------------------------------------------

namespace py = pybind11;

PYBIND11_MODULE(material_model, m) {
     py::module::import("pymor_dealii_bindings");

     py::class_<MaterialModel>(m, "MaterialModel")
          .def(py::init<const MaterialModelConfig&>())
          .def("make_grid", &MaterialModel::make_grid)
          .def("setup_system", &MaterialModel::setup_system)
          .def_readwrite("m_q", &MaterialModel::m_q)
          .def_readwrite("m_d", &MaterialModel::m_d)
          .def("assemble_system_matrix", &MaterialModel::assemble_system_matrix)
          .def("assemble_l2_matrix", &MaterialModel::assemble_l2_matrix)
          .def("assemble_l2_0_matrix", &MaterialModel::assemble_l2_0_matrix)
          .def("assemble_h1_semi_matrix", &MaterialModel::assemble_h1_semi_matrix)
          .def("assemble_h1_0_semi_matrix", &MaterialModel::assemble_h1_0_semi_matrix)
          .def("assemble_h1_matrix", &MaterialModel::assemble_h1_matrix)
          .def("assemble_h1_0_matrix", &MaterialModel::assemble_h1_0_matrix)
          .def("assemble_mass_matrix", &MaterialModel::assemble_mass_matrix)
          .def("sparsity_pattern", &MaterialModel::sparsity_pattern, py::return_value_policy::reference_internal);

     py::class_<MaterialModelConfig>(m, "MaterialModelConfig")
          .def(py::init<>())
          .def_readwrite("T_initial", &MaterialModelConfig::T_initial)
          .def_readwrite("T_final", &MaterialModelConfig::T_final)
          .def_readwrite("delta_t", &MaterialModelConfig::delta_t)
          .def_readwrite("polynomial_degree", &MaterialModelConfig::polynomial_degree)
          .def_readwrite("par_dim", &MaterialModelConfig::par_dim)
          .def_readwrite("refine_global", &MaterialModelConfig::refine_global);
}