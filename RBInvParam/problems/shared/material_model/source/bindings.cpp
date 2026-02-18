#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <deal.II/base/mpi.h>

#include <fstream>

#include "MaterialModel.hpp"
#include "BodyForceFactory.hpp"
#include "BoundaryConditionFactory.hpp"
#include "ObservationOperatorFactory.hpp"
#include "StateProductFactory.hpp"
#include "ObservationSpaceProductFactory.hpp"
#include "MaterialOperatorFactory.hpp"

// -------- PYTHON BINDINGS -----------------------------------------------------------------------

namespace py = pybind11;

PYBIND11_MODULE(material_model, m) {
      py::module::import("pymor_dealii_bindings");
      // m.def("_mpi_info", []() {
      //    return py::make_tuple(
      //       dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD),
      //       dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
      //    );
      // });

      py::class_<MaterialModel>(m, "MaterialModel")
         //.def(py::init<const MaterialModelBaseConfig&>())
         .def("setup_system", &MaterialModel::setup_system)

         .def_readonly("delta_t", &MaterialModel::delta_t)
         .def_readonly("param_space_dim", &MaterialModel::m_param_dim)
         .def_readonly("state_space_dim", &MaterialModel::m_state_dim)
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

         //.def("get_component_dofs", &MaterialModel::get_component_dofs, py::return_value_policy::reference_internal)
         .def("clear_rhs_boundary_dofs", &MaterialModel::clear_rhs_boundary_dofs, py::return_value_policy::reference_internal)
         .def("save_state", &MaterialModel::save_state, py::return_value_policy::reference_internal)
         .def("save_time_series", &MaterialModel::save_time_series, py::return_value_policy::reference_internal)
         
         .def("state_space_context", &MaterialModel::state_space_context, py::return_value_policy::reference_internal)
         .def("param_space_context", &MaterialModel::param_space_context, py::return_value_policy::reference_internal)
         .def("q_time_dep", &MaterialModel::q_time_dep, py::return_value_policy::reference_internal)
         .def("A_affine", &MaterialModel::A_affine, py::return_value_policy::reference_internal)
         .def("A_q_linear", &MaterialModel::A_q_linear, py::return_value_policy::reference_internal)
         ;

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

      py::enum_<BoundaryConditionType>(m, "BoundaryConditionType")
         .value("AllNeumann", BoundaryConditionType::AllNeumann)
         .value("DirichletOnYandZ", BoundaryConditionType::DirichletOnYandZ)
         .export_values();

     py::class_<MaterialModelBaseConfig>(m, "MaterialModelBaseConfig")
          .def(py::init<>())
          .def_readwrite("nt", &MaterialModelBaseConfig::nt)
          .def_readwrite("T_initial", &MaterialModelBaseConfig::T_initial)
          .def_readwrite("T_final", &MaterialModelBaseConfig::T_final)
          .def_readwrite("p1", &MaterialModelBaseConfig::p1)
          .def_readwrite("p2", &MaterialModelBaseConfig::p2)
          .def_readwrite("param_grid_resolution", &MaterialModelBaseConfig::param_grid_resolution)
          .def_readwrite("state_grid_resolution", &MaterialModelBaseConfig::state_grid_resolution)
          .def_readwrite("body_force_type", &MaterialModelBaseConfig::body_force_type)
          .def_readwrite("body_force_hyperparameter", &MaterialModelBaseConfig::body_force_hyperparameter)
          .def_readwrite("BC_type", &MaterialModelBaseConfig::BC_type)
          .def_readwrite("BC_hyperparameter", &MaterialModelBaseConfig::BC_hyperparameter);
}