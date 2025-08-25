#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream>

#include "MaterialModel.hpp"
#include "MaterialMatricesFactory.hpp"
#include "BodyForce.hpp"
#include "ObservationOperatorFactory.hpp"
#include "StateProductFactory.hpp"

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
         //  .def("assemble_l2_matrix", &MaterialModel::assemble_l2_matrix)
         //  .def("assemble_l2_0_matrix", &MaterialModel::assemble_l2_0_matrix)
         //  .def("assemble_h1_semi_matrix", &MaterialModel::assemble_h1_semi_matrix)
         //  .def("assemble_h1_0_semi_matrix", &MaterialModel::assemble_h1_0_semi_matrix)
         //  .def("assemble_h1_matrix", &MaterialModel::assemble_h1_matrix)
         //  .def("assemble_h1_0_matrix", &MaterialModel::assemble_h1_0_matrix)
          .def("assemble_state_product", &MaterialModel::assemble_state_product)
          .def("assemble_mass_matrix", &MaterialModel::assemble_mass_matrix)
          .def("sparsity_pattern", &MaterialModel::sparsity_pattern, py::return_value_policy::reference_internal)
          .def("n_dofs", &MaterialModel::n_dofs, py::return_value_policy::reference_internal)
          .def("get_force_list", &MaterialModel::get_force_list, py::return_value_policy::reference_internal)
          //.def("assemble_euclidian_matrix", &MaterialModel::assemble_euclidian_matrix, py::return_value_policy::reference_internal)
          .def("assemble_observation_operator_matrix", &MaterialModel::assemble_observation_operator_matrix, py::return_value_policy::reference_internal)
          .def("assemble_bilinear_cost_matrix", &MaterialModel::assemble_bilinear_cost_matrix, py::return_value_policy::reference_internal)
          .def("clear_rhs_boundary_dofs", &MaterialModel::clear_rhs_boundary_dofs, py::return_value_policy::reference_internal)
          .def("assemble_system_matrix_derivative", &MaterialModel::assemble_system_matrix_derivative, py::return_value_policy::reference_internal);

      py::enum_<BodyForceType>(m, "BodyForceType")
         .value("CenterExcite", BodyForceType::CenterExcite)
         .value("Dummy", BodyForceType::Dummy)
         .export_values();

      py::enum_<SystemMatrixType>(m, "SystemMatrixType")
         .value("Cosserat", SystemMatrixType::Cosserat)
         .value("CosseratDelamination", SystemMatrixType::CosseratDelamination)
         .export_values();

      py::enum_<ObservationOperatorType>(m, "ObservationOperatorType")
         .value("Identity", ObservationOperatorType::Identity)
         .value("Boundary", ObservationOperatorType::Boundary)
         .value("SensorsR9d", ObservationOperatorType::SensorsR9d)
         .value("SensorsR8d", ObservationOperatorType::SensorsR8d)
         .export_values();

      py::enum_<StateProductType>(m, "StateProductType")
         .value("L2", StateProductType::L2)
         .value("L2_0", StateProductType::L2_0)
         .value("H1_semi", StateProductType::H1_semi)
         .value("H1_0_semi", StateProductType::H1_0_semi)
         .value("H1", StateProductType::H1)
         .value("H1_0", StateProductType::H1_0)
         .value("Mass", StateProductType::Mass)
         .value("BoundaryMass", StateProductType::BoundaryMass)
         .export_values();

     py::class_<MaterialModelConfig>(m, "MaterialModelConfig")
          .def(py::init<>())
          .def_readwrite("nt", &MaterialModelConfig::nt)
          .def_readwrite("T_initial", &MaterialModelConfig::T_initial)
          .def_readwrite("T_final", &MaterialModelConfig::T_final)
          .def_readwrite("delta_t", &MaterialModelConfig::delta_t)
          .def_readwrite("spatial_resolution", &MaterialModelConfig::spatial_resolution)
          .def_readwrite("body_force_type", &MaterialModelConfig::body_force_type)
          .def_readwrite("system_matrix_type", &MaterialModelConfig::system_matrix_type)
          .def_readwrite("system_matrix_hyperparameter", &MaterialModelConfig::system_matrix_hyperparameter);
          
}