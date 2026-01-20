#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream>

#include "MaterialModel.hpp"
#include "BodyForceFactory.hpp"
#include "ObservationOperatorFactory.hpp"
#include "StateProductFactory.hpp"
#include "ObservationSpaceProductFactory.hpp"

#include "MaterialOperatorFactory.hpp"

// -------- PYTHON BINDINGS -----------------------------------------------------------------------

namespace py = pybind11;

PYBIND11_MODULE(material_model, m) {
     py::module::import("pymor_dealii_bindings");

     py::class_<MaterialModel>(m, "MaterialModel")
          .def(py::init<const MaterialModelBaseConfig&>())
          .def("make_grid", &MaterialModel::make_grid)
          .def("setup_system", &MaterialModel::setup_system)
         
          .def_readonly("param_space_dim", &MaterialModel::m_param_space_dim)
          .def_readonly("state_space_dim", &MaterialModel::m_state_space_dim)
          .def_readonly("observation_space_dim", &MaterialModel::m_observation_space_dim)
          .def_readonly("m_has_translation_operator", &MaterialModel::m_has_translation_operator)

          .def_readonly("product_V", &MaterialModel::m_product_V)
          .def_readonly("product_H", &MaterialModel::m_product_H)
          .def_readonly("product_C", &MaterialModel::m_product_C)
          .def_readonly("product_L2", &MaterialModel::m_product_L2)
          .def_readonly("product_H1", &MaterialModel::m_product_H1)

          .def("assemble_product_V", &MaterialModel::assemble_product_V)
          .def("assemble_product_H", &MaterialModel::assemble_product_H)
          .def("assemble_product_C", &MaterialModel::assemble_product_C)

          .def("assemble_mass_matrix", &MaterialModel::assemble_mass_matrix)          
          .def("assemble_observation_operator_matrix", &MaterialModel::assemble_observation_operator_matrix, py::return_value_policy::reference_internal)
          .def("assemble_bilinear_cost_matrix", &MaterialModel::assemble_bilinear_cost_matrix, py::return_value_policy::reference_internal)
  
          .def_readonly("mass_matrix", &MaterialModel::m_mass_matrix)
          .def_readonly("observation_operator", &MaterialModel::m_observation_operator)
          .def_readonly("bilinear_cost_operator", &MaterialModel::m_bilinear_cost_operator)
          .def_readonly("force_list", &MaterialModel::m_force_list)
          .def_readonly("system_matrix_sp", &MaterialModel::m_system_matrix_sp)
          
          .def("get_component_dofs", &MaterialModel::get_component_dofs, py::return_value_policy::reference_internal)
          .def("clear_rhs_boundary_dofs", &MaterialModel::clear_rhs_boundary_dofs, py::return_value_policy::reference_internal)
          .def("save_state", &MaterialModel::save_state, py::return_value_policy::reference_internal)
          .def("save_time_series", &MaterialModel::save_time_series, py::return_value_policy::reference_internal);

      py::enum_<MaterialOperatorType>(m, "MaterialOperatorType")
         .value("Cosserat", MaterialOperatorType::Cosserat)
         .value("CosseratDelamination", MaterialOperatorType::CosseratDelamination)
         .value("CosseratSpatial", MaterialOperatorType::CosseratSpatial)
         .export_values();
      
      py::enum_<BodyForceType>(m, "BodyForceType")
         .value("CenterExcite", BodyForceType::CenterExcite)
         .value("Gaussian", BodyForceType::Gaussian)
         .export_values();

      py::enum_<ObservationOperatorType>(m, "ObservationOperatorType")
         .value("Identity", ObservationOperatorType::Identity)
         .value("Boundary", ObservationOperatorType::Boundary)
         .value("Sensors", ObservationOperatorType::Sensors)
         .value("SensorsGrid", ObservationOperatorType::SensorsGrid)         
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

     py::class_<MaterialModelBaseConfig>(m, "MaterialModelBaseConfig")
          .def(py::init<>())
          .def_readwrite("nt", &MaterialModelBaseConfig::nt)
          .def_readwrite("T_initial", &MaterialModelBaseConfig::T_initial)
          .def_readwrite("T_final", &MaterialModelBaseConfig::T_final)
          .def_readwrite("delta_t", &MaterialModelBaseConfig::delta_t)
          .def_readwrite("spatial_resolution", &MaterialModelBaseConfig::spatial_resolution)
          .def_readwrite("body_force_type", &MaterialModelBaseConfig::body_force_type)
          .def_readwrite("body_force_hyperparameter", &MaterialModelBaseConfig::body_force_hyperparameter);
}