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
#include "ObservationSpaceProductFactory.hpp"

// -------- PYTHON BINDINGS -----------------------------------------------------------------------

namespace py = pybind11;

PYBIND11_MODULE(material_model, m) {
     py::module::import("pymor_dealii_bindings");

     py::class_<MaterialModel>(m, "MaterialModel")
          .def(py::init<const MaterialModelConfig&>())
          .def("make_grid", &MaterialModel::make_grid)
          .def("setup_system", &MaterialModel::setup_system)
         
          .def_readonly("param_space_dim", &MaterialModel::m_param_space_dim)
          .def_readonly("state_space_dim", &MaterialModel::m_state_space_dim)
          .def_readonly("observation_space_dim", &MaterialModel::m_observation_space_dim)

          .def_readonly("product_V", &MaterialModel::m_product_V)
          .def_readonly("product_H", &MaterialModel::m_product_H)
          .def_readonly("product_C", &MaterialModel::m_product_C)

          .def("assemble_product_V", &MaterialModel::assemble_product_V)
          .def("assemble_product_H", &MaterialModel::assemble_product_H)
          .def("assemble_product_C", &MaterialModel::assemble_product_C)

          .def("assemble_system_matrix", &MaterialModel::assemble_system_matrix)
          .def("assemble_mass_matrix", &MaterialModel::assemble_mass_matrix)          
          .def("assemble_observation_operator_matrix", &MaterialModel::assemble_observation_operator_matrix, py::return_value_policy::reference_internal)
          .def("assemble_bilinear_cost_matrix", &MaterialModel::assemble_bilinear_cost_matrix, py::return_value_policy::reference_internal)
          .def("assemble_system_matrix_derivative", &MaterialModel::assemble_system_matrix_derivative, py::return_value_policy::reference_internal)
  
          .def_readwrite("m_q", &MaterialModel::m_q)
          .def_readonly("mass_matrix", &MaterialModel::m_mass_matrix)
          .def_readonly("system_matrix", &MaterialModel::m_system_matrix)
          .def_readonly("system_matrix_derivative", &MaterialModel::m_system_matrix_derivative)
          .def_readonly("observation_operator", &MaterialModel::m_observation_operator)
          .def_readonly("bilinear_cost_operator", &MaterialModel::m_bilinear_cost_operator)
          .def_readonly("force_list", &MaterialModel::m_force_list)
          .def_readonly("system_matrix_sp", &MaterialModel::m_system_matrix_sp)
          
          .def("clear_rhs_boundary_dofs", &MaterialModel::clear_rhs_boundary_dofs, py::return_value_policy::reference_internal);

      py::enum_<BodyForceType>(m, "BodyForceType")
         .value("CenterExcite", BodyForceType::CenterExcite)
         .value("Dummy", BodyForceType::Dummy)
         .export_values();

      py::enum_<SystemMatrixType>(m, "SystemMatrixType")
         .value("Cosserat", SystemMatrixType::Cosserat)
         .value("CosseratDelamination", SystemMatrixType::CosseratDelamination)
         .value("CosseratSpatial", SystemMatrixType::CosseratSpatial)
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
      
      py::enum_<ObservationSpaceProductType>(m, "ObservationSpaceProductType")
         .value("EUCLID", ObservationSpaceProductType::EUCLID)
         .value("STATE_L2", ObservationSpaceProductType::STATE_L2)
         .value("STATE_L2_0", ObservationSpaceProductType::STATE_L2_0)
         .value("STATE_H1_semi", ObservationSpaceProductType::STATE_H1_semi)
         .value("STATE_H1_0_semi", ObservationSpaceProductType::STATE_H1_0_semi)
         .value("STATE_H1", ObservationSpaceProductType::STATE_H1)
         .value("STATE_H1_0", ObservationSpaceProductType::STATE_H1_0)
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